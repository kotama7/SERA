# SERA 要件定義書 — SLURM実行パイプライン

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 24. SLURM実行パイプライン（Local LLM + 実験実行）

SERAはSLURMクラスタ上で2つの異なる実行パターンをサポートする：

1. **実験のSLURM実行** — 生成された実験スクリプトをSLURMジョブとして計算ノードに投入
2. **Local LLM（vLLM）のGPU管理** — ヘッドノード上でvLLMを動作させ、PPO学習とGPUメモリを協調管理

### 24.1 全体アーキテクチャ

```text
┌─────────────── ヘッドノード / ログインノード ───────────────┐
│                                                              │
│  ┌─ AgentLLM ────────────────────────────────────────────┐  │
│  │  provider="local", inference.engine="vllm"            │  │
│  │  ┌─ VLLMInferenceEngine ──────────────────────────┐   │  │
│  │  │  vllm.LLM (offline mode)                       │   │  │
│  │  │  LoRA hot-swap via LoRARequest                  │   │  │
│  │  │  sleep(level=2) / wake_up()                     │   │  │
│  │  └────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                    │                                         │
│  SearchManager ────┤                                         │
│  TreeOps           │                                         │
│  PPOTrainer ───────┘  ← GPU共有: vLLM sleep → PPO → wake   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
          │
          │ submitit.AutoExecutor → sbatch
          ▼
┌─── SLURM 計算ノード ───┐
│  _run_experiment()      │
│  ・module load          │
│  ・experiment.py 実行   │
│  ・metrics.json 出力    │
└─────────────────────────┘
```

**重要な設計判断**: vLLMはSLURMジョブ内ではなくヘッドノード上で動作する。これによりジョブキュー待ちなしでの即座の仮説生成が可能。

### 24.2 実験のSLURM実行

#### 24.2.1 設定（ResourceSpec）

`resource_spec.yaml` で実行バックエンドとSLURM設定を指定する：

```yaml
# resource_spec.yaml
compute:
  executor_type: slurm        # "local" | "slurm" | "docker"
  gpu_required: true
  gpu_type: A100
  gpu_count: 2
  cpu_cores: 16
  memory_gb: 128

slurm:
  partition: gpu               # SLURMパーティション名
  account: my_project          # SLURMアカウント / プロジェクト
  time_limit: "04:00:00"       # 壁時計時間制限 (HH:MM:SS or D-HH:MM:SS)
  modules:                     # ジョブ内でロードするモジュール
    - cuda/12.1
    - pytorch/2.0
  sbatch_extra:                # 追加sbatchディレクティブ
    - "--gres=gpu:a100:2"
    - "--constraint=gpu80"
```

**Specモデル定義**（`src/sera/specs/resource_spec.py`）：

| クラス | フィールド | 型 | デフォルト | 説明 |
|--------|-----------|-----|-----------|------|
| `ComputeConfig` | `executor_type` | `str` | `"local"` | 実行バックエンド選択 |
| `ComputeConfig` | `gpu_required` | `bool` | `True` | GPUが必要か |
| `ComputeConfig` | `gpu_type` | `str` | `""` | GPU種別制約（例: `"A100"`） |
| `ComputeConfig` | `gpu_count` | `int` | `1` | GPU数 |
| `ComputeConfig` | `cpu_cores` | `int` | `8` | CPUコア数 |
| `ComputeConfig` | `memory_gb` | `int` | `32` | メモリ（GB） |
| `SlurmConfig` | `partition` | `str` | `"gpu"` | SLURMパーティション |
| `SlurmConfig` | `account` | `str` | `""` | SLURMアカウント |
| `SlurmConfig` | `time_limit` | `str` | `"04:00:00"` | 壁時計時間制限 |
| `SlurmConfig` | `modules` | `list[str]` | `[]` | ロードする環境モジュール |
| `SlurmConfig` | `sbatch_extra` | `list[str]` | `[]` | 追加sbatchディレクティブ |

**ComputeConfig → submitit 自動マッピング**: `ComputeConfig` のフィールドは `SlurmExecutor` によって submitit のネイティブパラメータに自動変換される。`sbatch_extra` で同等のパラメータが明示指定された場合は `sbatch_extra` が優先される（ユーザー指定が常に勝つ）。

| ComputeConfig フィールド | submitit パラメータ | 優先度競合時の動作 |
|--------------------------|--------------------|--------------------|
| `gpu_count`（`gpu_required=True` の場合のみ） | `slurm_gpus_per_node` | `sbatch_extra` に `gres` / `gpus-per-node` があれば削除 |
| `memory_gb` | `slurm_mem`（例: `"64G"`） | `sbatch_extra` に `mem` / `mem-per-cpu` があれば削除 |
| `cpu_cores` | `slurm_cpus_per_task` | `sbatch_extra` に `cpus-per-task` があれば削除 |
| `gpu_type` | `slurm_additional_parameters.constraint` | `sbatch_extra` に `gres` があれば設定しない |

#### 24.2.2 Executor選択フロー

`research_cmd.py:64-90` でspec値に基づきExecutorが動的に選択される：

```python
executor_type = getattr(specs.resource.compute, "executor_type", "local")

if executor_type == "slurm":
    from sera.execution.slurm_executor import SlurmExecutor
    executor = SlurmExecutor(
        work_dir=workspace,
        slurm_config=specs.resource.slurm,
        compute_config=specs.resource.compute,  # ComputeConfig → submitit 自動マッピング
        interpreter_command=interpreter_cmd,     # 多言語対応
        seed_arg_format=seed_arg_fmt,
    )
```

`compute_config` はオプション（`None` 可）で後方互換性を維持する。`replay_cmd.py` でも同様に `ComputeConfig` が渡される。

#### 24.2.3 SlurmExecutor 実行フロー

`SlurmExecutor.run()`（`src/sera/execution/slurm_executor.py:108-256`）の処理手順：

```text
SlurmExecutor.run(node_id, script_path, seed, timeout_sec)
  │
  ├─ 1. ディレクトリ準備
  │     runs/{node_id}/ を作成
  │     stdout.log, stderr.log, metrics.json パスを設定
  │     slurm_logs/ サブディレクトリを作成
  │
  ├─ 2. submitit設定
  │     submitit.AutoExecutor(folder=slurm_logs/)
  │     slurm_partition, slurm_time, slurm_job_name を設定
  │     ComputeConfig → submitit パラメータ自動マッピング（低優先度）
  │     sbatch_extra → slurm_additional_parameters に変換（高優先度、競合時はこちらが勝つ）
  │
  ├─ 3. ジョブ投入
  │     executor.submit(_run_experiment, interpreter, script, seed, run_dir, modules)
  │     → sbatchジョブとしてSLURMに投入される
  │
  ├─ 4. ポーリング（完了待ち）
  │     sacct利用可能 → submitit経由でjob.stateを確認
  │     sacct利用不可 → squeue -j <job_id> -h -o "%T" で確認
  │     timeout_sec超過 → scancel + TimeoutError
  │
  ├─ 5. ログ収集
  │     submitit出力を stdout.log / stderr.log にコピー（未出力の場合）
  │
  ├─ 6. OOM検出（多層アプローチ）
  │     ① SLURM job state == "OUT_OF_MEMORY"
  │     ② exit_code == 137 or -9 + stderrパターンマッチ
  │     ③ stderr内の "MemoryError" / "OutOfMemoryError" 検出
  │
  └─ 7. RunResult返却
        success, exit_code, stdout_path, stderr_path, metrics_path, wall_time_sec, seed
```

#### 24.2.4 SLURMジョブ内の実行

`_run_experiment()`（`src/sera/execution/slurm_executor.py:26-63`）はSLURMジョブ内部で実行されるcallable：

```python
def _run_experiment(interpreter_command, script_path, seed, run_dir, modules, seed_arg_format):
    # 1. 環境モジュールロード: module load <mod>
    # 2. コマンド構築: [interpreter, script_path, "--seed", str(seed)]
    # 3. subprocess.Popen で実行（stdout/stderrをファイルにリダイレクト）
    # 4. exit code を返却
```

#### 24.2.5 sbatch_extra のパース規則

`sbatch_extra` リスト内の各ディレクティブは以下の形式をサポート：

| 入力形式 | パース結果 |
|---------|-----------|
| `"--gres=gpu:1"` | `{"gres": "gpu:1"}` |
| `"--constraint A100"` | `{"constraint": "A100"}` |
| `"#SBATCH --mem=128G"` | `{"mem": "128G"}` |

これらは `submitit` の `slurm_additional_parameters` に渡される。

#### 24.2.6 タイムアウト制御（二重レイヤー）

| レイヤー | 制御元 | 動作 |
|---------|--------|------|
| **SLURM time_limit** | `SlurmConfig.time_limit` | スケジューラによる強制終了。`state="TIMEOUT"` |
| **Python timeout_sec** | `SlurmExecutor.run()` の引数 | ポーリングループで検出 → `scancel` → `exit_code=-9` |

Python側タイムアウトはSLURMのtime_limitより短い値を設定して早期終了に使用する。

#### 24.2.7 終了コードマッピング

| SLURM State | exit_code | SERAでの意味 | SearchNode.status |
|------------|-----------|-------------|-------------------|
| `COMPLETED` | `job.result()` (通常0) | 成功 | `"evaluated"` |
| `FAILED` | `1` | スクリプトエラー | `"failed"` |
| `TIMEOUT` | `-9` | 時間制限超過 | `"timeout"` |
| `OUT_OF_MEMORY` | `-7` (SERA独自センチネル) | メモリ不足 | `"oom"` |
| `CANCELLED` | `-15` | ユーザーまたは自動キャンセル | `"failed"` |

#### 24.2.8 依存ライブラリ

`submitit`（Meta Research製）をSLURMジョブ管理に使用。オプション依存：

```bash
pip install "sera[slurm]"  # or: pip install submitit
```

### 24.3 Local LLM（vLLM）のSLURMクラスタ上での動作

#### 24.3.1 設定（ModelSpec）

`model_spec.yaml` でvLLMエンジンと推論設定を指定する：

```yaml
# model_spec.yaml
base_model:
  id: meta-llama/Llama-3.1-70B
  revision: ""
  dtype: bf16
  load_method: auto
  max_seq_len: 8192

agent_llm:
  provider: local              # "local" | "openai" | "anthropic"
  temperature: 0.7
  max_tokens: 4096

inference:
  engine: vllm                 # "vllm" | "transformers"
  gpu_memory_utilization: 0.5  # vLLMのGPUメモリ使用率
  max_lora_rank: 64            # vLLMのLoRAプリアロケーション最大ランク
  adapter_cache_dir: /dev/shm/sera_adapters  # tmpfs上のアダプタキャッシュ
  swap_space_gb: 4.0           # vLLMのCPUスワップ領域
  enforce_eager: false         # CUDAグラフ無効化（デバッグ用）
```

**Specモデル定義**（`src/sera/specs/model_spec.py:60-68`）：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `engine` | `str` | `"transformers"` | 推論エンジン選択 |
| `gpu_memory_utilization` | `float` | `0.5` | vLLMのGPUメモリ割当率 |
| `max_lora_rank` | `int` | `64` | LoRAプリアロケーション最大ランク |
| `adapter_cache_dir` | `str` | `"/dev/shm/sera_adapters"` | アダプタのtmpfsキャッシュ |
| `swap_space_gb` | `float` | `4.0` | CPUスワップ領域（GB） |
| `enforce_eager` | `bool` | `False` | Eagerモード強制（デバッグ） |

#### 24.3.2 vLLMエンジン初期化フロー

`AgentLLM`（`src/sera/agent/agent_llm.py`）は遅延初期化パターンを使用：

```text
AgentLLM.__init__()
  │ _inference_engine = model_spec.inference.engine  （"vllm" or "transformers"）
  │ _vllm_engine = None  （遅延初期化）
  │
AgentLLM.generate(prompt, purpose, adapter_node_id)
  │
  ├─ provider == "local" AND engine == "vllm"
  │   ├─ _init_vllm_engine()  （初回のみ）
  │   │   └─ VLLMInferenceEngine(model_spec)
  │   │       └─ vllm.LLM(model=..., enable_lora=True, ...)
  │   │
  │   └─ _vllm_engine.generate(prompt, temp, max_tok, adapter_node_id, lineage_manager)
  │
  └─ provider == "local" AND engine == "transformers"
      └─ transformers + peft による従来の推論パス
```

#### 24.3.3 VLLMInferenceEngine

`src/sera/agent/vllm_engine.py` の主要コンポーネント：

**初期化**（`__init__`）:
```python
self._llm = LLM(
    model=model_spec.base_model.id,
    revision=model_spec.base_model.revision,
    dtype=model_spec.base_model.dtype,
    enable_lora=True,                      # LoRA事前有効化
    max_lora_rank=inf.max_lora_rank,       # プリアロケーション
    gpu_memory_utilization=inf.gpu_memory_utilization,
    max_model_len=model_spec.base_model.max_seq_len,
    swap_space=inf.swap_space_gb,
    enforce_eager=inf.enforce_eager,
)
```

**LoRA Hot-Swap**（`_get_lora_request`）:
```text
_get_lora_request(adapter_node_id, lineage_manager)
  ├─ キャッシュ確認: adapter_cache_dir/{adapter_node_id}/adapter_model.safetensors
  │   存在しない場合:
  │   └─ lineage_manager.export_for_vllm(adapter_node_id, adapter_dir, model_spec)
  │       ├─ materialize(): root→nodeまでのデルタ重みを累積復元
  │       ├─ save_file(): adapter_model.safetensors を出力
  │       └─ adapter_config.json を出力（peft互換フォーマット）
  │
  ├─ vLLM用整数IDの割当: adapter_id_map[adapter_node_id] → int_id
  └─ LoRARequest(adapter_node_id, int_id, str(adapter_dir)) を返却
```

**生成**（`generate`）:
```python
outputs = self._llm.generate(
    [prompt],
    SamplingParams(temperature=temperature, max_tokens=max_tokens),
    lora_request=lora_request,  # リクエスト単位でアダプタを指定
)
```

#### 24.3.4 GPUメモリ協調管理（Sleep/Wake プロトコル）

**課題**: vLLMとPyTorch（PPO学習）は同一GPU上で共存できない。

**解決策**: vLLMの `sleep(level=2)` / `wake_up()` APIによる明示的なメモリ管理。

```text
SearchManager.run() ループ
  │
  ├─ Phase 2-4: vLLM使用中（通常推論）
  │   AgentLLM → VLLMInferenceEngine.generate()
  │   GPU: vLLMがメモリ占有
  │
  ├─ Phase 5: PPO更新トリガー
  │   PPOTrainer.update()
  │     │
  │     ├─ vllm_engine.sleep()          ← GPUメモリ解放
  │     │   └─ self._llm.sleep(level=2)   level=2 = 完全解放
  │     │
  │     ├─ _ppo_update_core()           ← PyTorchがGPUを使用
  │     │   ├─ GAE計算
  │     │   ├─ PPOクリッピング損失
  │     │   ├─ LoRAパラメータのみ更新
  │     │   └─ デルタ抽出 → lineage保存
  │     │
  │     └─ vllm_engine.wake()           ← GPUメモリ復帰（finally句で保証）
  │         └─ self._llm.wake_up()
  │
  └─ 次のイテレーションでvLLM再利用
```

`ppo_trainer.py:184-203` で `try/finally` パターンにより、PPO更新の成否にかかわらず `wake()` が必ず呼ばれる：

```python
vllm_engine = getattr(agent_llm, "_vllm_engine", None)
if vllm_engine is not None:
    vllm_engine.sleep()
try:
    return await self._ppo_update_core(rollouts, agent_llm, specs, ...)
finally:
    if vllm_engine is not None:
        vllm_engine.wake()
```

#### 24.3.5 アダプタキャッシュ戦略

| 要素 | 詳細 |
|------|------|
| **キャッシュ場所** | `/dev/shm/sera_adapters`（tmpfs = RAMディスク） |
| **キャッシュ単位** | `{adapter_node_id}/adapter_model.safetensors` + `adapter_config.json` |
| **キャッシュ判定** | safetensorsファイルの存在チェック |
| **材料化** | `LineageManager.materialize()`: root→nodeパスのデルタ累積 |
| **出力形式** | peft互換: `adapter_model.safetensors` + `adapter_config.json` |
| **vLLM ID管理** | `adapter_id_map: dict[str, int]` で文字列ID→整数IDマッピング |

#### 24.3.6 export_for_vllm の出力

`LineageManager.export_for_vllm()`（`src/sera/lineage/lineage_manager.py:331-381`）が生成するファイル：

```text
{adapter_cache_dir}/{adapter_node_id}/
  ├─ adapter_model.safetensors    # 材料化されたLoRA重み（safetensors形式）
  └─ adapter_config.json          # peft設定
      {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": <rank>,
        "lora_alpha": <alpha>,
        "target_modules": [...],
        "lora_dropout": <dropout>,
        "bias": "none",
        "base_model_name_or_path": "<model_id>"
      }
```

### 24.4 統合パイプライン：SLURMクラスタ上の全体フロー

```text
research_cmd.run_research()
  │
  ├─ 1. Spec読み込み・検証
  │     executor_type = specs.resource.compute.executor_type  ("slurm")
  │     engine = specs.model.inference.engine                 ("vllm")
  │
  ├─ 2. コンポーネント初期化
  │     AgentLLM(model_spec, resource_spec)      → vLLMエンジン（遅延初期化）
  │     SlurmExecutor(work_dir, slurm_config, compute_config)  → SLURM実験実行器
  │     StatisticalEvaluator(executor, ...)       → 評価器（executor経由で実験実行）
  │     PPOTrainer(exec_spec, model_spec, ...)   → PPO学習器（オプション）
  │     LineageManager(lineage_dir)              → LoRA系譜管理
  │
  ├─ 3. 探索ループ（SearchManager.run()）
  │     │
  │     ├─ Phase 2: ノード生成
  │     │   AgentLLM.generate() → vLLMで仮説/コード生成（ヘッドノード上）
  │     │
  │     ├─ Phase 3: 実験実行
  │     │   StatisticalEvaluator → SlurmExecutor.run()
  │     │     → sbatchでSLURMジョブ投入（計算ノード）
  │     │     → ポーリングで完了待ち
  │     │     → RunResult返却
  │     │
  │     ├─ Phase 4: 統計評価
  │     │   mu, se, lcb 計算
  │     │   逐次評価（repeats回繰り返し、各回がSLURMジョブ）
  │     │
  │     ├─ Phase 5: PPO学習（条件付き）
  │     │   learning.enabled=True AND provider="local" の場合のみ
  │     │   vLLM sleep → PPO更新 → vLLM wake
  │     │
  │     └─ Phase 6: 系譜管理・剪定
  │         デルタsquash、深いノードの剪定
  │
  └─ 4. 結果出力
        best_node の情報表示
        export-best でアーティファクトエクスポート
```

### 24.5 PPO/Lineageの有効化条件

`research_cmd.py:113-136` において、PPOとLineageは以下の条件でのみ有効化：

```python
learning_enabled = getattr(specs.execution.learning, "enabled", True)
# AND agent_llm.provider == "local" （暗黙の前提：PPOはローカルモデルでのみ可能）
```

| 条件 | PPO | Lineage | vLLM Sleep/Wake |
|------|-----|---------|-----------------|
| `learning.enabled=True` + `provider="local"` | 有効 | 有効 | 有効 |
| `learning.enabled=True` + `provider="openai"` | 無効（例外でfallback） | 無効 | N/A |
| `learning.enabled=False` | 無効 | 無効 | N/A |

PPO/Lineage無効時でも探索ループ（Phase 2-4）は正常に動作する。

### 24.7 非同期パイプライン：SLURM実行時のフェーズ重複回避

#### 24.7.1 課題：逐次ボトルネック

現在の `SearchManager.run()` ループは逐次処理であり、1ノードずつ以下を直列に実行する：

```text
[Phase 2: 仮説生成 (vLLM)] → [Phase 3: 実験投入+待機 (SLURM)] → [Phase 4: 評価] → [Phase 5: PPO] → ...
```

SLURM使用時の問題点：
- `SlurmExecutor.run()` はブロッキング呼び出し（ポーリングで完了待ち）
- SLURM実験がキュー待ち＋計算中の間、ヘッドノードのvLLMは**遊休状態**
- 仮説生成中はSLURMクラスタのGPUが**遊休状態**
- 1ノードの実験が`repeats`回すべて完了するまで次のノード生成に進めない

#### 24.7.2 設計方針：非同期バッチパイプライン

SLURM実行時は、ヘッドノード処理（LLM推論・PPO学習）とSLURMジョブ実行を**時間的に重複させない**パイプラインを構築する。

```text
┌─ ヘッドノード（GPU使用）─────────────────────────────────────────────┐
│                                                                      │
│  ┌─ Phase A: バッチ生成 ─┐  ┌─ Phase C: バッチ評価+PPO ─┐          │
│  │  vLLM推論で複数ノード  │  │  結果収集・統計評価       │          │
│  │  の仮説を一括生成      │  │  PPO更新 (vLLM sleep)    │          │
│  │  実験コード一括生成    │  │  vLLM wake               │          │
│  └────────┬───────────────┘  └───────────┬───────────────┘          │
│           │                              │                           │
└───────────┼──────────────────────────────┼───────────────────────────┘
            │                              ▲
            ▼                              │
┌─ SLURM 計算ノード ──────────────────────────────────────────────────┐
│                                                                      │
│  ┌─ Phase B: バッチ実験実行 ─────────────────────────────────────┐  │
│  │  複数SLURMジョブを同時投入 → 並列実行 → 結果回収             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**3フェーズの時系列**：

```text
時間 →
Phase A ████░░░░░░░░░░░░████░░░░░░░░░░░░████  (ヘッドノードGPU: vLLM)
Phase B ░░░░████████████░░░░████████████░░░░   (SLURMクラスタ: 実験実行)
Phase C ░░░░░░░░░░░░████░░░░░░░░░░░░████░░░░  (ヘッドノードGPU: 評価+PPO)

█ = 実行中, ░ = 待機/未使用
```

**原則**: ヘッドノードGPU使用フェーズ（A, C）とSLURMジョブ実行フェーズ（B）は**同時に走ってよい**が、Phase AとPhase Cは**同一GPUを使うため排他**である。PPO実行中はvLLMをsleepさせる既存の仕組み（§24.3.4）をそのまま活用する。

#### 24.7.3 パイプラインの各フェーズ詳細

**Phase A: バッチ生成**（ヘッドノードGPU — vLLM推論）

```text
SearchManager._batch_generate(batch_size)
  │
  ├─ 1. open_listから最大 batch_size 個のノードを選択
  │     （pending → evaluate, evaluated → improve, failed → debug）
  │
  ├─ 2. 各ノードに対して TreeOps で仮説/コード生成
  │     ・Draft: 新仮説をバッチ生成
  │     ・Debug: 失敗ノードの修正案を生成
  │     ・Improve: 成功ノードの改善案を生成
  │
  ├─ 3. ExperimentGenerator で実験スクリプトを一括生成
  │
  └─ 4. 生成したノード群を「投入待ちキュー」(submit_queue) に積む
```

**Phase B: バッチ実験実行**（SLURM計算ノード — ヘッドノードGPUは不要）

```text
SearchManager._batch_submit_and_wait(submit_queue)
  │
  ├─ 1. submit_queue の全ノード × repeats 分のSLURMジョブを投入
  │     SlurmExecutor.submit_async(node_id, script, seed)
  │       → submitit.submit() でジョブ投入（結果を待たない）
  │       → Job オブジェクトを pending_jobs リストに保持
  │
  ├─ 2. 全ジョブの完了を一括ポーリング
  │     SlurmExecutor.wait_all(pending_jobs, timeout_sec)
  │       → 定期的に全ジョブの state を確認
  │       → 完了したジョブから順に RunResult を収集
  │       → タイムアウト超過ジョブは scancel
  │
  └─ 3. RunResult リストを返却
```

**Phase C: バッチ評価 + PPO更新**（ヘッドノードGPU — 統計計算 + PPO学習）

```text
SearchManager._batch_evaluate_and_learn(results)
  │
  ├─ 1. 各ノードの RunResult を集約、統計量 (mu, se, lcb) を計算
  │
  ├─ 2. feasibility チェック、best_node 更新
  │
  ├─ 3. PPOバッファに追加 → 条件を満たせばPPO更新
  │     ・vLLM sleep → PPO更新 → vLLM wake（既存プロトコル）
  │
  ├─ 4. 剪定（pruning）
  │
  └─ 5. チェックポイント保存
```

#### 24.7.4 SlurmExecutor の拡張インターフェース

非同期バッチ投入をサポートするため、`SlurmExecutor` に以下のメソッドを追加する：

```python
class SlurmExecutor(Executor):
    # 既存（同期・単一ジョブ）
    def run(self, node_id, script_path, seed, timeout_sec) -> RunResult: ...

    # 新規（非同期・バッチ対応）
    def submit_async(self, node_id: str, script_path: Path, seed: int) -> SlurmJobHandle:
        """SLURMジョブを投入し、即座にハンドルを返す（ブロックしない）。"""
        ...

    def poll_jobs(self, handles: list[SlurmJobHandle]) -> list[SlurmJobStatus]:
        """投入済みジョブの現在のステータスを一括取得する。"""
        ...

    def wait_all(
        self,
        handles: list[SlurmJobHandle],
        timeout_sec: int | None = None,
    ) -> list[RunResult]:
        """全ジョブの完了を待ち、RunResult リストを返す。
        タイムアウト超過ジョブは scancel して exit_code=-9 を返す。
        """
        ...

    def cancel_all(self, handles: list[SlurmJobHandle]) -> None:
        """投入済みジョブを一括キャンセルする（SIGINT時のクリーンアップ用）。"""
        ...
```

**`SlurmJobHandle`** データクラス：

```python
@dataclass
class SlurmJobHandle:
    node_id: str
    seed: int
    job: Any           # submitit Job オブジェクト
    run_dir: Path
    start_time: float  # time.monotonic()
```

#### 24.7.5 SearchManager の変更

`SearchManager.run()` のメインループをSLURM使用時に分岐する：

```python
async def run(self) -> SearchNode | None:
    # 初期ノード生成
    initial_nodes = await self.tree_ops.draft(n_initial, self.all_nodes)
    for node in initial_nodes:
        self._add_node(node)

    if self._is_slurm_executor():
        return await self._run_batched_pipeline()
    else:
        return await self._run_sequential_pipeline()  # 既存ロジック
```

**バッチパイプラインループ**：

```python
async def _run_batched_pipeline(self) -> SearchNode | None:
    batch_size = getattr(self.specs.execution.search, "slurm_batch_size", 5)

    while not self._should_terminate():
        # Phase A: バッチ生成（ヘッドノードGPU使用）
        submit_queue = await self._batch_generate(batch_size)
        if not submit_queue:
            break

        # Phase B: バッチ投入 + 完了待ち（ヘッドノードGPU未使用）
        results = self.executor.wait_all(
            [self.executor.submit_async(n.node_id, n.script_path, seed)
             for n, seed in submit_queue],
            timeout_sec=...,
        )

        # Phase C: バッチ評価 + PPO（ヘッドノードGPU使用）
        await self._batch_evaluate_and_learn(results)

        self._checkpoint_if_needed()
```

#### 24.7.6 設定パラメータ

`execution_spec.yaml` に追加するバッチパイプライン関連の設定：

```yaml
# execution_spec.yaml
search:
  slurm_batch_size: 5        # 1バッチあたりの最大ノード数
  slurm_max_concurrent: 10   # 同時投入SLURMジョブ数の上限
  slurm_poll_interval_sec: 15 # バッチポーリング間隔（秒）
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `slurm_batch_size` | `int` | `5` | Phase Aで一度に生成するノード数 |
| `slurm_max_concurrent` | `int` | `10` | SLURMに同時投入するジョブ数の上限（`repeats` × `batch_size` が上限を超える場合はチャンク分割） |
| `slurm_poll_interval_sec` | `float` | `15.0` | `wait_all` のポーリング間隔 |

**`slurm_max_concurrent` の制約理由**：
- SLURMパーティションのジョブ数制限（`MaxSubmitJobs`）を超えない
- 他ユーザーのジョブとの公平性を維持
- ファイルシステムへの同時書き込み負荷を制限

#### 24.7.7 SIGINT時のクリーンアップ

バッチパイプラインでは投入済みSLURMジョブのキャンセルが必要：

```text
SIGINT受信
  ├─ 1. 投入済みジョブを一括 scancel（SlurmExecutor.cancel_all）
  ├─ 2. チェックポイント保存（既存ロジック）
  └─ 3. exit(20)
```

`SearchManager._setup_signal_handler()` で `cancel_all` を呼び出すよう拡張する。

#### 24.7.8 パイプラインとGPU使用の排他制御まとめ

| フェーズ | ヘッドノードGPU | SLURMクラスタ | 備考 |
|---------|----------------|--------------|------|
| Phase A: バッチ生成 | **vLLM使用** | 未使用 | LLM推論でGPU占有 |
| Phase B: バッチ実験 | **未使用** | **ジョブ実行中** | ヘッドノードはポーリングのみ（CPU） |
| Phase C: 評価+PPO | **PPO使用** | 未使用 | vLLM sleep → PPO → vLLM wake |

**重要**: Phase BではヘッドノードのGPUは使用されないため、`vLLM.sleep(level=2)` でGPUメモリを解放することも可能。ただしPhase B完了後にPhase Cで再度vLLMが必要な場合、wake_upの再初期化コストとのトレードオフを考慮する。通常はvLLMをsleepせず待機させるほうが効率的である（Phase Bの後にPhase Aが来る可能性もあるため）。

#### 24.7.9 ローカル実行時の動作

`executor_type="local"` の場合はバッチパイプラインを使用せず、既存の逐次ループをそのまま使用する。ローカル実行では実験がヘッドノード上で走るため、vLLM推論と実験実行が同一マシンのGPU/CPUを競合する。逐次実行が安全かつ効率的である。

### 24.8 ファイルリファレンス

| ファイル | 主要クラス/関数 | 役割 |
|---------|---------------|------|
| `src/sera/execution/slurm_executor.py` | `SlurmExecutor`, `_run_experiment`, `_build_compute_params`, `submit_async`, `wait_all`, `cancel_all`, `SlurmJobHandle` | SLURMジョブ投入・ポーリング・OOM検出・ComputeConfig自動マッピング・非同期バッチ投入 |
| `src/sera/agent/vllm_engine.py` | `VLLMInferenceEngine` | vLLM推論 + LoRA Hot-Swap + sleep/wake |
| `src/sera/agent/agent_llm.py` | `AgentLLM` | LLMプロバイダ選択・vLLM遅延初期化 |
| `src/sera/learning/ppo_trainer.py` | `PPOTrainer` | PPO更新 + vLLM sleep/wake協調 |
| `src/sera/lineage/lineage_manager.py` | `LineageManager.export_for_vllm()` | アダプタ材料化 + peft形式エクスポート |
| `src/sera/commands/research_cmd.py` | `run_research()` | Executor選択・コンポーネント組み立て |
| `src/sera/search/search_manager.py` | `SearchManager._run_batched_pipeline()`, `_batch_generate()`, `_batch_evaluate_and_learn()` | バッチパイプライン制御（SLURM使用時） |
| `src/sera/specs/resource_spec.py` | `ComputeConfig`, `SlurmConfig` | SLURM設定スキーマ |
| `src/sera/specs/model_spec.py` | `InferenceConfig` | vLLM設定スキーマ |

---
