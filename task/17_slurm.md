# SERA 要件定義書 — SLURM実行パイプライン

> 本ファイルは TASK.md v13.3 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 23. SLURM実行パイプライン（Local LLM + 実験実行）

SERAはSLURMクラスタ上で2つの異なる実行パターンをサポートする：

1. **実験のSLURM実行** — 生成された実験スクリプトをSLURMジョブとして計算ノードに投入
2. **Local LLM（vLLM）のGPU管理** — ヘッドノード上でvLLMを動作させ、PPO学習とGPUメモリを協調管理

### 23.1 全体アーキテクチャ

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

### 23.2 実験のSLURM実行

#### 23.2.1 設定（ResourceSpec）

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
  modules: []                  # ジョブ内でロードするモジュール（デフォルト: 空リスト）
    # 例:
    # - cuda/12.1
    # - pytorch/2.0
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
| `SlurmConfig` | `modules` | `list[str]` | `[]`（空リスト） | ロードする環境モジュール（環境依存のため既定値は空） |
| `SlurmConfig` | `sbatch_extra` | `list[str]` | `[]` | 追加sbatchディレクティブ |

**ComputeConfig → submitit 自動マッピング**: `ComputeConfig` のフィールドは `SlurmExecutor` によって submitit のネイティブパラメータに自動変換される。`sbatch_extra` で同等のパラメータが明示指定された場合は `sbatch_extra` が優先される（ユーザー指定が常に勝つ）。

| ComputeConfig フィールド | submitit パラメータ | 優先度競合時の動作 |
|--------------------------|--------------------|--------------------|
| `gpu_count`（`gpu_required=True` の場合のみ） | `slurm_gpus_per_node` | `sbatch_extra` に `gres` / `gpus-per-node` があれば削除 |
| `memory_gb` | `slurm_mem`（例: `"64G"`） | `sbatch_extra` に `mem` / `mem-per-cpu` があれば削除 |
| `cpu_cores` | `slurm_cpus_per_task` | `sbatch_extra` に `cpus-per-task` があれば削除 |
| `gpu_type` | `slurm_additional_parameters.constraint` | `sbatch_extra` に `gres` があれば設定しない |

#### 23.2.2 Executor選択フロー

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

#### 23.2.3 SlurmExecutor 実行フロー

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

#### 23.2.4 SLURMジョブ内の実行

`_run_experiment()`（`src/sera/execution/slurm_executor.py:26-63`）はSLURMジョブ内部で実行されるcallable：

```python
def _run_experiment(interpreter_command, script_path, seed, run_dir, modules, seed_arg_format):
    # 1. 環境モジュールロード: module load <mod>
    # 2. コマンド構築: [interpreter, script_path, "--seed", str(seed)]
    # 3. subprocess.Popen で実行（stdout/stderrをファイルにリダイレクト）
    # 4. exit code を返却
```

#### 23.2.5 sbatch_extra のパース規則

`sbatch_extra` リスト内の各ディレクティブは以下の形式をサポート：

| 入力形式 | パース結果 |
|---------|-----------|
| `"--gres=gpu:1"` | `{"gres": "gpu:1"}` |
| `"--constraint A100"` | `{"constraint": "A100"}` |
| `"#SBATCH --mem=128G"` | `{"mem": "128G"}` |

これらは `submitit` の `slurm_additional_parameters` に渡される。

#### 23.2.6 タイムアウト制御（二重レイヤー）

| レイヤー | 制御元 | 動作 |
|---------|--------|------|
| **SLURM time_limit** | `SlurmConfig.time_limit` | スケジューラによる強制終了。`state="TIMEOUT"` |
| **Python timeout_sec** | `SlurmExecutor.run()` の引数 | ポーリングループで検出 → `scancel` → `exit_code=-9` |

Python側タイムアウトはSLURMのtime_limitより短い値を設定して早期終了に使用する。

#### 23.2.7 終了コードマッピング

| SLURM State | exit_code | SERAでの意味 | SearchNode.status |
|------------|-----------|-------------|-------------------|
| `COMPLETED` | `job.result()` (通常0) | 成功 | `"evaluated"` |
| `FAILED` | `1` | スクリプトエラー | `"failed"` |
| `TIMEOUT` | `-9` | 時間制限超過 | `"timeout"` |
| `OUT_OF_MEMORY` | `-7` (SERA独自センチネル) | メモリ不足 | `"oom"` |
| `CANCELLED` | `-15` | ユーザーまたは自動キャンセル | `"failed"` |

#### 23.2.8 依存ライブラリ

`submitit`（Meta Research製）をSLURMジョブ管理に使用。オプション依存：

```bash
pip install "sera[slurm]"  # or: pip install submitit
```

### 23.3 Local LLM（vLLM）のSLURMクラスタ上での動作

#### 23.3.1 設定（ModelSpec）

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

#### 23.3.2 vLLMエンジン初期化フロー

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

#### 23.3.3 VLLMInferenceEngine

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

#### 23.3.4 GPUメモリ協調管理（Sleep/Wake プロトコル）

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

#### 23.3.5 アダプタキャッシュ戦略

| 要素 | 詳細 |
|------|------|
| **キャッシュ場所** | `/dev/shm/sera_adapters`（tmpfs = RAMディスク） |
| **キャッシュ単位** | `{adapter_node_id}/adapter_model.safetensors` + `adapter_config.json` |
| **キャッシュ判定** | safetensorsファイルの存在チェック |
| **材料化** | `LineageManager.materialize()`: root→nodeパスのデルタ累積 |
| **出力形式** | peft互換: `adapter_model.safetensors` + `adapter_config.json` |
| **vLLM ID管理** | `adapter_id_map: dict[str, int]` で文字列ID→整数IDマッピング |

#### 23.3.6 export_for_vllm の出力

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

### 23.4 統合パイプライン：SLURMクラスタ上の全体フロー

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

### 23.5 PPO/Lineageの有効化条件

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

### 23.6 SLURM + コンテナ統合（Singularity / Apptainer / Docker）

HPC クラスタでは SLURM と Singularity/Apptainer の組み合わせが標準的な実行環境である。現在の `SlurmExecutor` は素の `subprocess` 実行のみをサポートしており、コンテナ化された実験環境に未対応である。本節では `SlurmConfig` 内に `ContainerConfig` をネストし、SLURM ジョブ内でのコンテナ実行をサポートする。

#### 23.6.1 ContainerConfig 定義

`SlurmConfig` 内にネストする。`enabled: false`（デフォルト）で既存動作に影響しない：

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `enabled` | `bool` | `False` | コンテナ実行を有効化 |
| `runtime` | `str` | `"singularity"` | コンテナランタイム（`"singularity"`, `"apptainer"`, `"docker"`） |
| `image` | `str` | `""` | コンテナイメージ（URI or `.sif` パス）。例: `"docker://nvcr.io/nvidia/pytorch:24.01-py3"`, `"/shared/images/sera.sif"` |
| `bind_mounts` | `list[str]` | `[]` | バインドマウント（例: `["/data:/data", "/scratch:/scratch"]`） |
| `env_vars` | `dict[str, str]` | `{}` | コンテナ内に渡す環境変数 |
| `gpu_enabled` | `bool` | `True` | GPU パススルー（Singularity: `--nv`, Apptainer: `--nv`, Docker: `--gpus all`） |
| `extra_flags` | `list[str]` | `[]` | ランタイム固有の追加フラグ |
| `overlay` | `str` | `""` | オーバーレイファイルシステム（Singularity/Apptainer のみ。例: `"/scratch/overlay.img"`) |
| `writable_tmpfs` | `bool` | `False` | 書き込み可能な tmpfs を有効化（Singularity/Apptainer: `--writable-tmpfs`） |

全フィールドは Frozen 層に属し、Phase 1 以降は不変である。

#### 23.6.2 YAML 設定例

**Singularity + SLURM の例**:

```yaml
slurm:
  partition: gpu
  account: my_project
  time_limit: "04:00:00"
  modules:
    - singularity/3.11
  sbatch_extra:
    - "--gres=gpu:a100:2"
  container:
    enabled: true
    runtime: singularity
    image: "/shared/images/sera_pytorch.sif"
    bind_mounts:
      - "${SERA_WORKSPACE}:/workspace"
      - "/data:/data:ro"
    env_vars:
      PYTHONPATH: "/workspace"
    gpu_enabled: true
    writable_tmpfs: true
```

**Apptainer + SLURM の例**:

```yaml
slurm:
  partition: gpu
  account: my_project
  time_limit: "04:00:00"
  modules:
    - apptainer/1.2
  container:
    enabled: true
    runtime: apptainer
    image: "docker://nvcr.io/nvidia/pytorch:24.01-py3"
    bind_mounts:
      - "${SERA_WORKSPACE}:/workspace"
      - "/scratch/${USER}:/scratch"
    env_vars:
      CUDA_VISIBLE_DEVICES: "0,1"
    gpu_enabled: true
    extra_flags:
      - "--cleanenv"
    overlay: "/scratch/${USER}/overlay.img"
```

#### 23.6.3 `_run_experiment` のコンテナラッピング

`_run_experiment()` 内で、`ContainerConfig.enabled == True` の場合、実験コマンドをコンテナ `exec` でラッピングする：

```text
_run_experiment(interpreter_command, script_path, seed, run_dir, modules, seed_arg_format, container_config)
  │
  ├─ container_config.enabled == False の場合:
  │   └─ 既存パス: [interpreter_command, script_path, seed_arg]
  │
  └─ container_config.enabled == True の場合:
      │
      ├─ 1. ベースコマンド構築
      │     runtime = container_config.runtime  # "singularity" | "apptainer" | "docker"
      │
      │     ■ Singularity / Apptainer:
      │       cmd = [runtime, "exec"]
      │       gpu_enabled → cmd += ["--nv"]
      │       writable_tmpfs → cmd += ["--writable-tmpfs"]
      │       overlay → cmd += ["--overlay", overlay]
      │       for mount in bind_mounts:
      │           cmd += ["--bind", mount]
      │       for key, val in env_vars.items():
      │           cmd += ["--env", f"{key}={val}"]
      │       cmd += extra_flags
      │       cmd += [image]
      │       cmd += [interpreter_command, script_path, seed_arg]
      │
      │     ■ Docker:
      │       cmd = ["docker", "run", "--rm"]
      │       gpu_enabled → cmd += ["--gpus", "all"]
      │       for mount in bind_mounts:
      │           cmd += ["-v", mount]
      │       for key, val in env_vars.items():
      │           cmd += ["-e", f"{key}={val}"]
      │       cmd += extra_flags
      │       cmd += [image]
      │       cmd += [interpreter_command, script_path, seed_arg]
      │
      ├─ 2. 実行
      │     subprocess.Popen(cmd, stdout=..., stderr=..., cwd=run_dir)
      │     ※ run_dir は自動的に bind_mounts に追加（ユーザー指定がない場合）
      │
      └─ 3. 結果返却
            exit code, stdout/stderr は既存の RunResult 契約に準拠
```

**生成されるコマンド例**（Singularity）:

```bash
singularity exec --nv --writable-tmpfs \
  --bind /home/user/sera_workspace/runs/abc123:/workspace \
  --bind /data:/data:ro \
  --env PYTHONPATH=/workspace \
  /shared/images/sera_pytorch.sif \
  python /workspace/experiment.py --seed 42
```

**生成されるコマンド例**（Apptainer）:

```bash
apptainer exec --nv --cleanenv \
  --overlay /scratch/user/overlay.img \
  --bind /home/user/sera_workspace/runs/abc123:/workspace \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  docker://nvcr.io/nvidia/pytorch:24.01-py3 \
  python /workspace/experiment.py --seed 42
```

#### 23.6.4 run_dir の自動バインド

コンテナ内から実験スクリプトと `metrics.json` 出力先にアクセスするため、`run_dir`（`runs/{node_id}/`）は自動的にバインドマウントに追加する：

```text
auto_bind = f"{run_dir}:{run_dir}"
if auto_bind not in container_config.bind_mounts:
    effective_bind_mounts = [auto_bind] + container_config.bind_mounts
```

これにより、コンテナ内からの `metrics.json` 書き込みがホスト側に反映され、既存の `RunResult.metrics_path` 契約が維持される。

#### 23.6.5 コンテナイメージの事前準備ガイダンス

**Singularity/Apptainer の `.sif` ファイル作成**:

```bash
# Docker Hub イメージから変換
singularity build sera_pytorch.sif docker://nvcr.io/nvidia/pytorch:24.01-py3

# Singularity レシピから
singularity build sera_pytorch.sif sera.def
```

**推奨イメージ内容**:
- Python 3.11+ + 実験に必要なライブラリ
- CUDA ランタイム（GPU 使用時）
- コンパイル型言語サポート時（§7.3.2）: 該当コンパイラ（`g++`, `cargo`, `go`）
- `metrics.json` を書き込めるディレクトリのパーミッション

**注意事項**:
- `.sif` ファイルは共有ファイルシステム上に配置（各ノードからアクセス可能であること）
- Docker URI（`docker://...`）は初回実行時にプルされるため、ネットワークアクセスが必要
- キャッシュされたイメージは `~/.singularity/cache`（Singularity）または `~/.apptainer/cache`（Apptainer）に保存

#### 23.6.6 Docker on SLURM の注意事項

SLURM クラスタで Docker を使用する場合の制約と推奨事項：

| 項目 | 注意事項 |
|------|---------|
| **権限** | Docker デーモンへのアクセスには通常 root 権限が必要。HPC 環境では Singularity/Apptainer を推奨 |
| **セキュリティ** | 多くの HPC サイトでは Docker の使用を禁止している。サイトポリシーを確認すること |
| **ネットワーク** | 計算ノードからのネットワークアクセスが制限されている場合、イメージの事前プルが必要 |
| **GPU** | `--gpus all` は NVIDIA Container Toolkit が必要 |
| **ファイルシステム** | Docker の `-v` マウントは共有ファイルシステム上のパスを使用すること |
| **推奨** | HPC 環境では `runtime: "singularity"` または `runtime: "apptainer"` を優先使用 |

#### 23.6.7 Spec モデル拡張

`SlurmConfig`（`src/sera/specs/resource_spec.py`）に `container` フィールドを追加する：

```python
class ContainerConfig(BaseModel):
    enabled: bool = False
    runtime: str = "singularity"        # "singularity" | "apptainer" | "docker"
    image: str = ""
    bind_mounts: list[str] = []
    env_vars: dict[str, str] = {}
    gpu_enabled: bool = True
    extra_flags: list[str] = []
    overlay: str = ""
    writable_tmpfs: bool = False

class SlurmConfig(BaseModel):
    partition: str = "gpu"
    account: str = ""
    time_limit: str = "04:00:00"
    modules: list[str] = []
    sbatch_extra: list[str] = []
    container: ContainerConfig = ContainerConfig()  # 新規追加
```

#### 23.6.8 SlurmExecutor の変更

`SlurmExecutor.__init__()` が `ContainerConfig` を受け取り、`_run_experiment()` に渡す：

```python
class SlurmExecutor(Executor):
    def __init__(
        self,
        work_dir: Path,
        slurm_config: SlurmConfig,
        compute_config: ComputeConfig | None = None,
        interpreter_command: str = "python",
        seed_arg_format: str | None = None,
    ):
        # 既存フィールド
        self._container_config = slurm_config.container  # 新規: SlurmConfig からネスト取得
```

`submit` 呼び出しで `_run_experiment` に `container_config` を追加引数として渡す。`ContainerConfig.enabled == False` の場合は既存動作と完全に同一である。

#### 23.6.9 テスト計画

| テストケース | 概要 |
|-------------|------|
| `test_container_config_defaults` | `enabled=False` のデフォルト値で既存動作に影響なし |
| `test_container_config_singularity` | Singularity 設定の YAML パース・バリデーション |
| `test_container_config_apptainer` | Apptainer 設定の YAML パース・バリデーション |
| `test_container_config_docker` | Docker 設定の YAML パース・バリデーション |
| `test_container_command_singularity` | Singularity exec コマンドが正しく構築されること |
| `test_container_command_apptainer` | Apptainer exec コマンドが正しく構築されること |
| `test_container_command_docker` | Docker run コマンドが正しく構築されること |
| `test_container_gpu_flag` | `gpu_enabled=True` → `--nv`（Singularity）/ `--gpus all`（Docker） |
| `test_container_gpu_disabled` | `gpu_enabled=False` → GPU フラグなし |
| `test_container_bind_mounts` | バインドマウントが正しくコマンドに反映されること |
| `test_container_auto_bind_run_dir` | `run_dir` が自動的にバインドされること |
| `test_container_env_vars` | 環境変数が正しくコマンドに反映されること |
| `test_container_overlay` | オーバーレイ設定が正しくコマンドに反映されること |
| `test_container_writable_tmpfs` | `--writable-tmpfs` フラグが正しく付与されること |
| `test_container_disabled_passthrough` | `enabled=False` 時に既存の素の実行パスが使われること |

### 23.7 非同期パイプライン：SLURM実行時のフェーズ重複回避

#### 23.7.1 課題：逐次ボトルネック

現在の `SearchManager.run()` ループは逐次処理であり、1ノードずつ以下を直列に実行する：

```text
[Phase 2: 仮説生成 (vLLM)] → [Phase 3: 実験投入+待機 (SLURM)] → [Phase 4: 評価] → [Phase 5: PPO] → ...
```

SLURM使用時の問題点：
- `SlurmExecutor.run()` はブロッキング呼び出し（ポーリングで完了待ち）
- SLURM実験がキュー待ち＋計算中の間、ヘッドノードのvLLMは**遊休状態**
- 仮説生成中はSLURMクラスタのGPUが**遊休状態**
- 1ノードの実験が`repeats`回すべて完了するまで次のノード生成に進めない

#### 23.7.2 設計方針：非同期バッチパイプライン

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

**原則**: ヘッドノードGPU使用フェーズ（A, C）とSLURMジョブ実行フェーズ（B）は**同時に走ってよい**が、Phase AとPhase Cは**同一GPUを使うため排他**である。PPO実行中はvLLMをsleepさせる既存の仕組み（§23.3.4）をそのまま活用する。

#### 23.7.3 パイプラインの各フェーズ詳細

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

#### 23.7.4 SlurmExecutor の拡張インターフェース

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

#### 23.7.5 SearchManager の変更

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

#### 23.7.6 設定パラメータ

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

#### 23.7.7 SIGINT時のクリーンアップ

バッチパイプラインでは投入済みSLURMジョブのキャンセルが必要：

```text
SIGINT受信
  ├─ 1. 投入済みジョブを一括 scancel（SlurmExecutor.cancel_all）
  ├─ 2. チェックポイント保存（既存ロジック）
  └─ 3. exit(20)
```

`SearchManager._setup_signal_handler()` で `cancel_all` を呼び出すよう拡張する。

#### 23.7.8 パイプラインとGPU使用の排他制御まとめ

| フェーズ | ヘッドノードGPU | SLURMクラスタ | 備考 |
|---------|----------------|--------------|------|
| Phase A: バッチ生成 | **vLLM使用** | 未使用 | LLM推論でGPU占有 |
| Phase B: バッチ実験 | **未使用** | **ジョブ実行中** | ヘッドノードはポーリングのみ（CPU） |
| Phase C: 評価+PPO | **PPO使用** | 未使用 | vLLM sleep → PPO → vLLM wake |

**重要**: Phase BではヘッドノードのGPUは使用されないため、`vLLM.sleep(level=2)` でGPUメモリを解放することも可能。ただしPhase B完了後にPhase Cで再度vLLMが必要な場合、wake_upの再初期化コストとのトレードオフを考慮する。通常はvLLMをsleepせず待機させるほうが効率的である（Phase Bの後にPhase Aが来る可能性もあるため）。

#### 23.7.9 ローカル実行時の動作

`executor_type="local"` の場合はバッチパイプラインを使用せず、既存の逐次ループをそのまま使用する。ローカル実行では実験がヘッドノード上で走るため、vLLM推論と実験実行が同一マシンのGPU/CPUを競合する。逐次実行が安全かつ効率的である。

### 23.8 ファイルリファレンス

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

### 23.9 分散SLURMアーキテクチャ（Distributed SLURM Architecture）

#### 23.9.1 課題：ヘッドノードGPU依存

§23.1 のアーキテクチャではvLLM推論・PPO学習がヘッドノード上で動作するため、以下の制約がある：

| 課題 | 詳細 |
|------|------|
| **ログインノードのGPU制限** | 多くのHPCサイトではログインノードにGPUがない、またはGPU使用が禁止されている |
| **リソース競合** | ヘッドノード上でvLLMとPPOのGPUメモリ協調が必要（§23.3.4 sleep/wake） |
| **スケーラビリティ** | ヘッドノードのGPUリソースがボトルネック。大規模モデルでは単ノードに収まらない |
| **ベンチマーク公正性** | 実験実行時に他ジョブの影響を排除できない |

#### 23.9.2 全体設計：ログインノード＝オーケストレーター

分散SLURMアーキテクチャでは、ログインノードはオーケストレーション（制御・調整）のみを担い、**すべてのGPU処理を計算ノード上のSLURMジョブとして実行**する。

```text
┌─────────────── ログインノード（GPUなし） ──────────────────────┐
│                                                                  │
│  ┌─ SearchManager (Orchestrator) ──────────────────────────┐    │
│  │  ・ノード選択 (open_list管理)                            │    │
│  │  ・フェーズ間調整 (A→B→C パイプライン制御)               │    │
│  │  ・チェックポイント保存・復帰                            │    │
│  │  ・SIGINT → 全ジョブ scancel + 安全停止                  │    │
│  └──────────────────────────────────────────────────────────┘    │
│           │                    │                    │             │
│    srun/sbatch            srun/sbatch          srun/sbatch       │
│           │                    │                    │             │
└───────────┼────────────────────┼────────────────────┼─────────────┘
            ▼                    ▼                    ▼
┌── 計算ノード群 ─────────────────────────────────────────────────┐
│                                                                  │
│  ┌─ vLLM推論ノード ──┐  ┌─ 実験ノード ──┐  ┌─ PPO学習ノード ┐  │
│  │ (Stage: inference) │  │ (Stage: exp)  │  │ (Stage: train) │  │
│  │                    │  │               │  │                │  │
│  │ vLLMサーバー       │  │ experiment.py │  │ PPOTrainer     │  │
│  │ (OpenAI互換API)    │  │ metrics.json  │  │ LoRAデルタ更新 │  │
│  │ LoRA hot-swap      │  │               │  │                │  │
│  └────────────────────┘  └───────────────┘  └────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**設計原則**：
- ログインノードは **CPU処理のみ**（GPU不使用）: SLURMジョブ投入・ポーリング・結果収集・探索木管理
- vLLM推論は計算ノード上で **OpenAI互換APIサーバー** として動作。ログインノードからHTTP経由でアクセス
- PPO学習は計算ノード上で **バッチジョブ** として実行。学習結果（LoRAデルタ）は共有ファイルシステム経由で回収
- 実験実行は既存の `SlurmExecutor`（§23.2）をそのまま使用

#### 23.9.3 StageNodeConfig — ステージ別ノード設定

各処理ステージ（推論・実験・学習）に対して、個別のSLURMリソースを設定できる：

```python
class StageNodeConfig(BaseModel):
    """Per-stage SLURM node configuration."""

    partition: str = Field("gpu", description="SLURM partition for this stage")
    nodes: int = Field(1, description="Number of nodes to allocate")
    gpu_count: int = Field(1, description="GPUs per node")
    gpu_type: str = Field("", description="GPU type constraint, e.g. 'A100'")
    cpu_cores: int = Field(8, description="CPU cores per node")
    memory_gb: int = Field(32, description="RAM per node in GB")
    time_limit: str = Field("04:00:00", description="Wall-clock time limit (HH:MM:SS)")
    exclusive: bool = Field(False, description="Request exclusive node allocation (--exclusive)")
    nodelist: str = Field("", description="Specific node list (--nodelist), e.g. 'gpu[01-04]'")
    sbatch_extra: list[str] = Field(default_factory=list, description="Additional sbatch directives")
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `partition` | `str` | `"gpu"` | SLURMパーティション |
| `nodes` | `int` | `1` | 確保ノード数 |
| `gpu_count` | `int` | `1` | ノードあたりGPU数 |
| `gpu_type` | `str` | `""` | GPU種別制約（例: `"A100"`） |
| `cpu_cores` | `int` | `8` | ノードあたりCPUコア数 |
| `memory_gb` | `int` | `32` | ノードあたりメモリ（GB） |
| `time_limit` | `str` | `"04:00:00"` | 壁時計時間制限 |
| `exclusive` | `bool` | `False` | 排他ノード割り当て（`--exclusive`） |
| `nodelist` | `str` | `""` | 指定ノードリスト（`--nodelist`） |
| `sbatch_extra` | `list[str]` | `[]` | 追加sbatchディレクティブ |

#### 23.9.4 DistributedSlurmConfig — 分散SLURM設定

```python
class DistributedSlurmConfig(BaseModel):
    """Distributed SLURM architecture settings (§23.9)."""

    enabled: bool = Field(False, description="Enable distributed SLURM mode")
    account: str = Field("", description="SLURM account (shared across stages)")
    modules: list[str] = Field(default_factory=list, description="Environment modules to load (shared)")
    inference: StageNodeConfig = Field(
        default_factory=lambda: StageNodeConfig(gpu_count=1, time_limit="08:00:00"),
        description="vLLM inference server node config",
    )
    experiment: StageNodeConfig = Field(
        default_factory=StageNodeConfig,
        description="Experiment execution node config",
    )
    training: StageNodeConfig = Field(
        default_factory=lambda: StageNodeConfig(gpu_count=1, time_limit="02:00:00"),
        description="PPO training node config",
    )
    vllm_mode: str = Field(
        "persistent", description="vLLM server lifecycle: 'persistent' (long-running) or 'transient' (per-batch)"
    )
```

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `enabled` | `bool` | `False` | 分散モード有効化。`False` の場合は既存の §23.1 アーキテクチャ |
| `account` | `str` | `""` | 共有SLURMアカウント |
| `modules` | `list[str]` | `[]` | 全ステージ共通の環境モジュール |
| `inference` | `StageNodeConfig` | gpu_count=1, time_limit="08:00:00" | vLLM推論ノード設定 |
| `experiment` | `StageNodeConfig` | デフォルト | 実験実行ノード設定 |
| `training` | `StageNodeConfig` | gpu_count=1, time_limit="02:00:00" | PPO学習ノード設定 |
| `vllm_mode` | `str` | `"persistent"` | vLLMサーバーライフサイクル |

**`vllm_mode` のルール**：

| `vllm_mode` | `inference.nodes` | `inference.exclusive` | 動作 |
|-------------|--------------------|-----------------------|------|
| `"persistent"` | 任意 | 任意 | vLLMサーバーを研究開始時に起動し、終了まで維持 |
| `"transient"` | `1` | `True`（推奨） | バッチごとにvLLMサーバーを起動・停止 |

`nodes=1` かつ `exclusive=True` の場合は `"transient"` が自動設定される（ノードを効率的に使い回すため）。ユーザが明示的に `"persistent"` を指定した場合はそちらを尊重する。

#### 23.9.5 resource_spec.yaml の設定例

`DistributedSlurmConfig` は `ComputeConfig` 内にネストされる：

```yaml
# resource_spec.yaml — 分散SLURMモード
compute:
  executor_type: slurm
  gpu_required: true

  slurm:
    partition: gpu                # 実験ノードのデフォルト（distributed.experimentで上書き可）
    account: my_project
    time_limit: "04:00:00"
    modules:
      - cuda/12.1

  distributed:
    enabled: true
    account: my_project           # 全ステージ共有（stageのpartitionが異なる場合でも同一アカウント）
    modules:
      - cuda/12.1
      - pytorch/2.4

    inference:                    # vLLM推論ノード
      partition: gpu-large
      nodes: 1
      gpu_count: 4
      gpu_type: A100
      memory_gb: 256
      time_limit: "12:00:00"
      nodelist: "gpu01"           # 特定ノードを指定可

    experiment:                   # 実験実行ノード
      partition: gpu
      nodes: 1
      gpu_count: 1
      gpu_type: A100
      memory_gb: 64
      time_limit: "02:00:00"

    training:                     # PPO学習ノード
      partition: gpu-large
      nodes: 1
      gpu_count: 2
      gpu_type: A100
      memory_gb: 128
      time_limit: "04:00:00"

    vllm_mode: persistent         # 研究中ずっとvLLMサーバーを維持
```

**ベンチマーク向け排他モード例**：

```yaml
  distributed:
    enabled: true
    account: benchmark_project

    experiment:
      partition: gpu-exclusive
      nodes: 1
      gpu_count: 8
      gpu_type: H100
      memory_gb: 512
      time_limit: "01:00:00"
      exclusive: true             # 他ジョブと共有しない

    vllm_mode: transient          # 排他ノード1台のため、推論時のみ起動
```

#### 23.9.6 Spec モデル配置

`DistributedSlurmConfig` と `StageNodeConfig` は `src/sera/specs/resource_spec.py` に配置する。`ComputeConfig` に `distributed` フィールドを追加する：

```python
class ComputeConfig(BaseModel):
    executor_type: str = Field("local", description="Execution backend: 'local', 'slurm', 'docker'")
    # ... 既存フィールド ...
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    distributed: DistributedSlurmConfig = Field(
        default_factory=DistributedSlurmConfig,
        description="Distributed SLURM architecture (§23.9). Only used when enabled=True and executor_type='slurm'",
    )
```

`distributed.enabled == False`（デフォルト）の場合は既存の §23.1 アーキテクチャがそのまま動作し、後方互換性を維持する。

### 23.10 vLLMリモートサーバー管理

分散SLURMモードでは、vLLMはヘッドノード上のインプロセスエンジンではなく、計算ノード上で **OpenAI互換APIサーバー** として動作する。

#### 23.10.1 VLLMServerManager クラス

```python
class VLLMServerManager:
    """Manages vLLM inference server lifecycle on SLURM compute nodes."""

    def __init__(
        self,
        distributed_config: DistributedSlurmConfig,
        model_spec: ModelSpecModel,
        workspace: Path,
    ):
        self._config = distributed_config
        self._model_spec = model_spec
        self._workspace = workspace
        self._server_job: SlurmJobHandle | None = None
        self._server_url: str | None = None

    async def start(self) -> str:
        """Submit vLLM server as SLURM job and return the API base URL."""
        ...

    async def stop(self) -> None:
        """Gracefully stop the vLLM server (scancel the SLURM job)."""
        ...

    async def health_check(self) -> bool:
        """Check if the vLLM server is responsive (GET /health)."""
        ...

    @property
    def api_base_url(self) -> str | None:
        """Return the OpenAI-compatible API base URL, or None if not running."""
        return self._server_url
```

#### 23.10.2 persistent vs transient モード

| モード | 起動タイミング | 停止タイミング | ユースケース |
|--------|---------------|---------------|-------------|
| **persistent** | 研究開始時（`SearchManager.run()` 冒頭） | 研究終了時（`finally` 句） | 長時間研究、頻繁なLLM呼び出し |
| **transient** | Phase Aバッチ生成前 | Phase A完了後 | 排他ノード利用、リソース節約 |

**persistent モードのフロー**：

```text
SearchManager.run() (分散モード)
  │
  ├─ 1. vllm_manager.start()
  │     ├─ sbatch: vLLMサーバージョブ投入（inference stage設定）
  │     ├─ srun内: vllm serve <model> --host 0.0.0.0 --port 8000
  │     ├─ ジョブ開始待ち → ノード名取得 → URL構築
  │     └─ health_check() でサーバー応答確認
  │
  ├─ 2. AgentLLM に API base URL をセット
  │     agent_llm.set_remote_endpoint(vllm_manager.api_base_url)
  │     → 以降の generate() は OpenAI互換クライアント経由
  │
  ├─ 3. 探索ループ（Phase A → B → C の繰り返し）
  │     ・Phase A: AgentLLM → HTTP → vLLMサーバー（計算ノード）
  │     ・Phase B: SlurmExecutor で実験投入（別の計算ノード）
  │     ・Phase C: PPOは RemotePPOExecutor 経由（§23.11）
  │
  └─ 4. finally: vllm_manager.stop()
        └─ scancel でvLLMサーバージョブを停止
```

**transient モードのフロー**：

```text
SearchManager._run_distributed_pipeline() — 各バッチ
  │
  ├─ Phase A:
  │   ├─ vllm_manager.start()     ← バッチごとに起動
  │   ├─ バッチ仮説生成 (AgentLLM → HTTP → vLLM)
  │   └─ vllm_manager.stop()      ← 生成完了で停止
  │
  ├─ Phase B: 実験実行（vLLMサーバー不要）
  │
  └─ Phase C: PPO学習（RemotePPOExecutor）
```

#### 23.10.3 vLLMサーバー起動スクリプト

`VLLMServerManager.start()` が生成するSLURMジョブスクリプト：

```bash
#!/bin/bash
#SBATCH --job-name=sera-vllm-server
#SBATCH --partition={inference.partition}
#SBATCH --nodes={inference.nodes}
#SBATCH --gpus-per-node={inference.gpu_count}
#SBATCH --mem={inference.memory_gb}G
#SBATCH --time={inference.time_limit}
#SBATCH --output={workspace}/logs/vllm_server_%j.log
{exclusive_flag}
{nodelist_flag}

# 環境モジュールロード
{module_load_commands}

# vLLMサーバー起動（OpenAI互換API）
python -m vllm.entrypoints.openai.api_server \
    --model {model_id} \
    --dtype {dtype} \
    --max-model-len {max_seq_len} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --enable-lora \
    --max-lora-rank {max_lora_rank} \
    --host 0.0.0.0 \
    --port 8000

# サーバー起動後、ノード名とポートをファイルに書き出す
echo "{hostname}:8000" > {workspace}/logs/vllm_server_address.txt
```

**URLの解決手順**：

1. ジョブ投入後、`squeue -j <job_id> -o "%N"` でノード名を取得
2. `http://{nodename}:8000/v1` をAPIベースURLとして構築
3. `GET /health` でサーバー応答を確認（最大30回リトライ、5秒間隔）
4. タイムアウト（150秒）内に応答がない場合は `scancel` + エラー

#### 23.10.4 AgentLLM のリモートモード

`distributed.enabled=True` の場合、`AgentLLM` はローカルvLLMエンジンの代わりに OpenAI互換クライアントを使用する：

```text
AgentLLM.generate() — 分散モード
  │
  ├─ provider == "local" AND distributed.enabled == True:
  │   └─ OpenAI 互換クライアント経由
  │       import openai
  │       client = openai.AsyncOpenAI(base_url=vllm_server_url)
  │       response = await client.chat.completions.create(
  │           model=model_id,
  │           messages=[{"role": "user", "content": prompt}],
  │           temperature=temperature,
  │           max_tokens=max_tokens,
  │       )
  │       ※ LoRA指定時は extra_body={"lora_request": {...}} を追加
  │
  └─ provider == "local" AND distributed.enabled == False:
      └─ 既存パス: VLLMInferenceEngine（インプロセス）
```

`set_remote_endpoint(url)` メソッドにより動的にエンドポイントを設定できる。`agent_llm.provider` は変更せず、内部でルーティングを分岐する。

### 23.11 リモートPPO実行

#### 23.11.1 課題

§23.3.4 のsleep/wakeプロトコルはヘッドノード上のインプロセスvLLMとPPOの協調を前提としている。分散モードではvLLMとPPOが別ノードで動作するため、sleep/wakeは不要だが、PPO学習自体をSLURMジョブとして投入する仕組みが必要。

#### 23.11.2 RemotePPOExecutor クラス

```python
class RemotePPOExecutor:
    """Executes PPO training as a SLURM job on a compute node."""

    def __init__(
        self,
        distributed_config: DistributedSlurmConfig,
        exec_spec: ExecutionSpecModel,
        model_spec: ModelSpecModel,
        workspace: Path,
    ):
        self._config = distributed_config
        self._exec_spec = exec_spec
        self._model_spec = model_spec
        self._workspace = workspace

    async def run_ppo_update(
        self,
        rollouts_path: Path,
        adapter_parent_id: str,
    ) -> PPOUpdateResult:
        """Submit PPO training as SLURM job and return results.

        Args:
            rollouts_path: Path to serialized PPORollout data (JSON).
            adapter_parent_id: Parent adapter node ID in the lineage tree.

        Returns:
            PPOUpdateResult with new adapter delta path and training metrics.
        """
        ...
```

#### 23.11.3 PPOジョブの投入・結果回収フロー

```text
RemotePPOExecutor.run_ppo_update(rollouts_path, adapter_parent_id)
  │
  ├─ 1. ロールアウトデータの準備
  │     rollouts_path に PPORollout をJSON/safetensors でシリアライズ済み
  │     共有ファイルシステム経由で計算ノードからアクセス可能
  │
  ├─ 2. PPOジョブスクリプト生成
  │     sera_ppo_worker.py を生成（引数: rollouts_path, adapter_parent_id, output_dir）
  │     training stage の StageNodeConfig で sbatch パラメータを設定
  │
  ├─ 3. sbatch でジョブ投入
  │     submitit.submit(sera_ppo_worker, ...)
  │     → 計算ノードで PPOTrainer._ppo_update_core() を実行
  │     → 結果を output_dir/ppo_result.json + adapter_delta.safetensors に書き出す
  │
  ├─ 4. ジョブ完了待ち（ポーリング）
  │     SlurmExecutor.wait_all() と同じポーリングメカニズム
  │
  └─ 5. 結果回収
        output_dir/ppo_result.json を読み込み → PPOUpdateResult として返却
        adapter_delta.safetensors → LineageManager に登録
```

**`PPOUpdateResult`** データクラス：

```python
@dataclass
class PPOUpdateResult:
    success: bool
    adapter_node_id: str             # 新規作成されたアダプタノードID
    adapter_delta_path: Path         # lineage/nodes/<id>/adapter_delta.safetensors
    metrics: dict[str, float]        # {"mean_reward": ..., "policy_loss": ..., "value_loss": ..., "kl_divergence": ...}
    wall_time_sec: float
```

#### 23.11.4 分散モードでの sleep/wake 不要性

| アーキテクチャ | vLLM | PPO | GPUメモリ協調 |
|---------------|------|-----|--------------|
| §23.1（ヘッドノード） | インプロセス（同一GPU） | インプロセス（同一GPU） | **sleep/wake必須** |
| §23.9（分散） | 計算ノードA（専用GPU） | 計算ノードB（専用GPU） | **不要**（別ノード） |

分散モードでは `PPOTrainer` の sleep/wake 呼び出しをスキップする。`research_cmd.py` で分散モード判定後、`PPOTrainer` の代わりに `RemotePPOExecutor` を使用する。

### 23.12 排他モード（Exclusive Node Allocation）

#### 23.12.1 目的

ベンチマーク実験やリソース集約型の処理では、計算ノードを排他的に確保して他ジョブの干渉を排除する必要がある。`StageNodeConfig.exclusive = True` により `--exclusive` フラグをsbatchに渡す。

#### 23.12.2 SlurmExecutor の --exclusive 対応

`StageNodeConfig.exclusive == True` の場合、submitit パラメータに `--exclusive` を追加する：

```python
def _build_stage_params(self, stage_config: StageNodeConfig) -> dict:
    """Build submitit parameters from StageNodeConfig."""
    params = {
        "slurm_partition": stage_config.partition,
        "slurm_gpus_per_node": stage_config.gpu_count,
        "slurm_mem": f"{stage_config.memory_gb}G",
        "slurm_cpus_per_task": stage_config.cpu_cores,
        "slurm_time": stage_config.time_limit,
    }
    if stage_config.exclusive:
        params["slurm_additional_parameters"] = {
            **params.get("slurm_additional_parameters", {}),
            "exclusive": "",  # --exclusive (no value)
        }
    if stage_config.nodelist:
        params.setdefault("slurm_additional_parameters", {})
        params["slurm_additional_parameters"]["nodelist"] = stage_config.nodelist
    # stage固有の sbatch_extra を追加
    for directive in stage_config.sbatch_extra:
        # 既存の _parse_sbatch_extra() ロジックを再利用
        ...
    return params
```

#### 23.12.3 ベンチマーク向けユースケース

| 設定パターン | 説明 | 排他モード |
|-------------|------|-----------|
| **ベンチマーク実験** | 全実験を同一ハードウェアで再現可能に実行 | `experiment.exclusive=True` |
| **大規模モデル推論** | vLLM推論でGPUメモリを最大限活用 | `inference.exclusive=True` |
| **PPO集約学習** | PPO更新の高速化のためGPU/CPUを独占 | `training.exclusive=True` |
| **完全排他ベンチマーク** | 全ステージを排他ノードで実行 | 全ステージ `exclusive=True` |

**排他モードと `vllm_mode` の連動**：

`inference.nodes=1` かつ `inference.exclusive=True` かつ `vllm_mode` 未指定の場合、`vllm_mode` は自動的に `"transient"` に設定される。理由：排他ノード1台をvLLMサーバーに専有させ続けるのはリソースの無駄であるため、推論が必要な期間のみ起動する。ユーザが明示的に `vllm_mode: persistent` を指定した場合はこの自動設定を上書きする。

### 23.13 分散パイプラインフロー

#### 23.13.1 SearchManager._run_distributed_pipeline() の全体フロー

```python
async def _run_distributed_pipeline(self) -> SearchNode | None:
    """Distributed SLURM pipeline: all GPU work on compute nodes."""
    dist_config = self.specs.resource.compute.distributed
    vllm_manager = VLLMServerManager(dist_config, self.specs.model, self._workspace)
    ppo_executor = RemotePPOExecutor(dist_config, self.specs.execution, self.specs.model, self._workspace)

    try:
        # persistent モード: 研究開始時にvLLMサーバーを起動
        if dist_config.vllm_mode == "persistent":
            api_url = await vllm_manager.start()
            self._agent_llm.set_remote_endpoint(api_url)

        while not self._should_terminate():
            # --- Phase A: バッチ生成 (計算ノード: vLLM推論) ---
            if dist_config.vllm_mode == "transient":
                api_url = await vllm_manager.start()
                self._agent_llm.set_remote_endpoint(api_url)

            submit_queue = await self._batch_generate(batch_size)

            if dist_config.vllm_mode == "transient":
                await vllm_manager.stop()

            if not submit_queue:
                break

            # --- Phase B: バッチ実験実行 (計算ノード: 実験) ---
            handles = [
                self._executor.submit_async(n.node_id, n.script_path, seed)
                for n, seed in submit_queue
            ]
            results = self._executor.wait_all(handles, timeout_sec=...)

            # --- Phase C: バッチ評価 + PPO学習 (計算ノード: PPO) ---
            await self._batch_evaluate(results)

            if self._should_run_ppo():
                ppo_result = await ppo_executor.run_ppo_update(
                    rollouts_path=self._serialize_rollouts(),
                    adapter_parent_id=self._current_adapter_id,
                )
                self._lineage_manager.register_delta(ppo_result)

            self._checkpoint_if_needed()

    finally:
        await vllm_manager.stop()  # persistent モード: 研究終了時にvLLMサーバーを停止
```

#### 23.13.2 3フェーズの時系列とノード使用状況

```text
時間 →

Phase A ████░░░░░░░░░░░░░████░░░░░░░░░░░░░████  (計算ノード: vLLM推論)
Phase B ░░░░████████████░░░░░████████████░░░░░░  (計算ノード: 実験実行)
Phase C ░░░░░░░░░░░░░████░░░░░░░░░░░░░████░░░░  (計算ノード: PPO学習)

ログイン ────────────────────────────────────────  (オーケストレーション: CPU only)
ノード   ポーリング・調整・チェックポイント保存

█ = GPU使用, ░ = 待機/未使用, ─ = CPU処理のみ
```

**§23.7 との差分**：§23.7 ではヘッドノードのGPUを Phase A と Phase C で共有していたが、分散モードでは各フェーズが **異なる計算ノード** で実行される。sleep/wake プロトコルは不要になり、代わりにSLURMジョブの投入・完了待ちで制御する。

#### 23.13.3 フェーズ間のデータフロー

```text
Phase A (vLLM推論ノード)
  │  生成物: experiment.py (runs/{node_id}/)
  │  経路: 共有ファイルシステム
  ▼
Phase B (実験実行ノード)
  │  入力: experiment.py
  │  出力: metrics.json, stdout.log, stderr.log
  │  経路: 共有ファイルシステム
  ▼
Phase C (PPO学習ノード)
  │  入力: rollouts (JSON), 親アダプタデルタ (safetensors)
  │  出力: 新アダプタデルタ (safetensors), ppo_result.json
  │  経路: 共有ファイルシステム
  ▼
ログインノード (オーケストレーター)
     入力: metrics.json, ppo_result.json
     処理: 統計計算, 探索木更新, チェックポイント保存
```

**前提条件**: すべてのノードが同一の共有ファイルシステム（NFS, Lustre, GPFS等）上の `sera_workspace/` にアクセス可能であること。これはHPCクラスタの標準的な構成である。

### 23.14 CLI/Wizard設定項目

#### 23.14.1 freeze-specs の追加オプション

`sera freeze-specs` コマンドに分散SLURM関連のオプションを追加する：

```python
@app.command()
def freeze_specs(
    # ... 既存オプション ...
    # 分散SLURMオプション
    distributed: Annotated[bool, typer.Option("--distributed")] = False,
    inference_gpu_count: Annotated[int, typer.Option("--inference-gpu-count")] = 1,
    inference_partition: Annotated[str, typer.Option("--inference-partition")] = "",
    inference_nodelist: Annotated[str, typer.Option("--inference-nodelist")] = "",
    training_gpu_count: Annotated[int, typer.Option("--training-gpu-count")] = 1,
    training_partition: Annotated[str, typer.Option("--training-partition")] = "",
    experiment_exclusive: Annotated[bool, typer.Option("--experiment-exclusive")] = False,
    vllm_mode: Annotated[str, typer.Option("--vllm-mode")] = "persistent",
):
```

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--distributed` | `bool` | `False` | 分散SLURMモードを有効化 |
| `--inference-gpu-count` | `int` | `1` | vLLM推論ノードのGPU数 |
| `--inference-partition` | `str` | `""` | vLLM推論ノードのパーティション（空=共通partition使用） |
| `--inference-nodelist` | `str` | `""` | vLLM推論ノードのノードリスト |
| `--training-gpu-count` | `int` | `1` | PPO学習ノードのGPU数 |
| `--training-partition` | `str` | `""` | PPO学習ノードのパーティション |
| `--experiment-exclusive` | `bool` | `False` | 実験ノードを排他割り当て |
| `--vllm-mode` | `str` | `"persistent"` | vLLMサーバーライフサイクル |

**既存オプションとの関係**: `--gpu-count`, `--gpu-type`, `--memory-gb` 等の既存オプションは実験ステージのデフォルト値として使用される。`--distributed` が `True` の場合、ステージ固有のオプションが優先される。

#### 23.14.2 Wizard Step 10c-slurm: SLURMノード設定

`sera setup` の Step 10c（ResourceSpec）に分散SLURM設定のサブステップを追加する。SLURM環境が検出された場合（`env.slurm_available == True`）にのみ表示される。

**対話フロー**：

```text
  [3/5] ResourceSpec — SLURM設定:

  SLURM環境を検出しました:
    パーティション: gpu, gpu-large, cpu
    デフォルトアカウント: my_project

  分散SLURMモードを使用しますか？
  （すべてのGPU処理を計算ノードで実行します）
    [1] はい（推奨: ログインノードにGPUがない環境）
    [2] いいえ（ヘッドノードでvLLM/PPOを実行）
  > 1

  ── vLLM推論ノード設定 ──
  パーティション [gpu-large]: >
  GPU数 [1]: > 4
  GPU種別 [A100]: >
  メモリ (GB) [256]: >
  時間制限 [12:00:00]: >
  ノードリスト (空=自動割り当て): >

  ── 実験実行ノード設定 ──
  パーティション [gpu]: >
  GPU数 [1]: >
  排他モード (ベンチマーク用) [N]: > y

  ── PPO学習ノード設定 ──
  パーティション [gpu-large]: >
  GPU数 [2]: >
  時間制限 [04:00:00]: >

  vLLMサーバーモード:
    [1] persistent — 研究中ずっと起動（推奨）
    [2] transient  — バッチごとに起動・停止
  > 1

  ┌──────────────────────────────────────────────────────┐
  │ distributed:                                          │
  │   enabled: true                                       │
  │   inference: {partition: gpu-large, gpu: 4×A100}      │
  │   experiment: {partition: gpu, exclusive: true}        │
  │   training: {partition: gpu-large, gpu: 2×A100}       │
  │   vllm_mode: persistent                               │
  └──────────────────────────────────────────────────────┘
  この設定でよろしいですか？ [Y/n/edit] >
```

**wizard_state への保存**：

```python
# wizard_state["phase1_params"] に追加されるキー
{
    "distributed_enabled": True,
    "distributed_inference_partition": "gpu-large",
    "distributed_inference_gpu_count": 4,
    "distributed_inference_gpu_type": "A100",
    "distributed_inference_memory_gb": 256,
    "distributed_inference_time_limit": "12:00:00",
    "distributed_inference_nodelist": "",
    "distributed_experiment_partition": "gpu",
    "distributed_experiment_exclusive": True,
    "distributed_training_partition": "gpu-large",
    "distributed_training_gpu_count": 2,
    "distributed_training_time_limit": "04:00:00",
    "distributed_vllm_mode": "persistent",
}
```

これらは `step11_freeze.py` で `cli_args` に展開され、`phase1_cmd.run_freeze_specs()` → `SpecBuilder` → `resource_spec.yaml` に反映される。

#### 23.14.3 ファイルリファレンス（§23.9–§23.14 追加分）

| ファイル | 主要クラス/関数 | 役割 |
|---------|---------------|------|
| `src/sera/specs/resource_spec.py` | `StageNodeConfig`, `DistributedSlurmConfig` | 分散SLURMスキーマ |
| `src/sera/execution/vllm_server_manager.py` | `VLLMServerManager` | vLLMリモートサーバー管理 |
| `src/sera/execution/remote_ppo_executor.py` | `RemotePPOExecutor`, `PPOUpdateResult` | リモートPPO実行 |
| `src/sera/search/search_manager.py` | `SearchManager._run_distributed_pipeline()` | 分散パイプライン制御 |
| `src/sera/agent/agent_llm.py` | `AgentLLM.set_remote_endpoint()` | リモートvLLMエンドポイント切替 |
| `src/sera/commands/phase1_cmd.py` | `freeze_specs()` 追加オプション | CLI分散SLURMオプション |
| `src/sera/commands/wizard/steps/step10_specs.py` | Step 10c-slurm | Wizard SLURM設定 |

---
