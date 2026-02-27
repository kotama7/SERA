# SERA クイックスタートガイド

## 前提条件

- **Python 3.11 以上**
- **CUDA 対応 GPU**（推奨: NVIDIA A100 / H100。LoRA 学習に必要。CPU のみでも API プロバイダ利用時は動作可能）
- **API キー**: 先行研究検索用に少なくとも 1 つの文献検索プロバイダのキーが必要

## インストール

### SLURM クラスタ環境（GPU ノードでのセットアップ — 推奨）

**重要:** ログインノード（GPU なし）で `pip install` すると PyTorch が CUDA を検出できず、local LLM が GPU を使用できなくなる。SLURM クラスタでは必ず GPU ノード上でセットアップを行うこと。

```bash
git clone <repository-url>
cd sera

# GPU ノード上で .venv を作成し CUDA 対応 PyTorch をインストール
srun --partition=<gpu-partition> --time=01:00:00 bash scripts/setup_env.sh

# 以後は .venv を activate して使用
source .venv/bin/activate
sera --help
```

詳細は [docs/setup_gpu.md](setup_gpu.md) を参照。

### ローカル環境（GPU マシン上での直接インストール）

GPU が直接利用できる環境では、通常の `pip install` で問題ない。

```bash
git clone <repository-url>
cd sera
pip install -e ".[dev]"
```

実行環境に応じてオプションの extras を追加できる。

```bash
# SLURM クラスタで実験を実行する場合
pip install -e ".[dev,slurm]"

# Docker コンテナで実験を実行する場合
pip install -e ".[dev,docker]"

# vLLM 推論エンジンを使用する場合
pip install -e ".[dev,vllm]"
```

各 extras の依存パッケージ:

| Extra | 追加パッケージ |
|-------|---------------|
| `dev` | pytest, pytest-asyncio, pytest-mock, respx, ruff |
| `slurm` | submitit |
| `docker` | docker |
| `vllm` | vllm |

インストールの確認:

```bash
sera --help
```

## 環境変数

以下の環境変数を設定する。必須のものはないが、設定することで機能が拡張される。

```bash
# 文献検索 API
export SEMANTIC_SCHOLAR_API_KEY="your-key"     # レート制限緩和（推奨）
export CROSSREF_EMAIL="you@example.com"        # CrossRef polite pool アクセス
export SERPAPI_API_KEY="your-key"              # Web 検索フォールバック（任意）

# LLM プロバイダ（local 以外を使用する場合）
export OPENAI_API_KEY="your-key"              # OpenAI プロバイダ使用時
export ANTHROPIC_API_KEY="your-key"           # Anthropic プロバイダ使用時

# モデルアクセス
export HF_TOKEN="your-token"                  # HuggingFace のゲート付きモデル用（任意）
```

API キー未設定時の動作: Phase 0 ではプロバイダチェーン（Semantic Scholar -> CrossRef -> arXiv -> Web）を順に試行し、成功した最初のプロバイダの結果を使用する。すべて失敗した場合はフォールバックヒューリスティックが適用される。

## Input-1 YAML の作成

SERA の入力は **Input-1 YAML** ファイルである。以下を `input1.yaml` として保存する。

```yaml
version: 1
data:
  description: "UCI Iris dataset"
  location: "./data/iris.csv"
  format: "csv"
  size_hint: "small(<1GB)"
domain:
  field: "ML"
  subfield: "classification"
task:
  brief: "Classify iris species"
  type: "prediction"
goal:
  objective: "maximize accuracy"
  direction: "maximize"
  baseline: "0.95"
constraints:
  - name: "inference_time_ms"
    type: "le"
    threshold: 100
notes: ""
```

### 各フィールドの説明

| フィールド | 説明 |
|-----------|------|
| `version` | スキーマバージョン（現在は `1`） |
| `data.description` | データセットの説明 |
| `data.location` | データファイルのパス |
| `data.format` | データ形式（csv, json 等） |
| `data.size_hint` | データサイズの目安（`small(<1GB)`, `medium(1-10GB)`, `large(>10GB)`） |
| `domain.field` | 研究分野 |
| `domain.subfield` | サブ分野 |
| `task.brief` | タスクの簡潔な説明 |
| `task.type` | タスク種別（prediction, generation 等） |
| `goal.objective` | 最適化目標の説明 |
| `goal.direction` | 最適化方向（`maximize` / `minimize`） |
| `goal.baseline` | ベースライン性能値 |
| `constraints` | ハード制約のリスト（`name`, `type`=`le`/`ge`/`bool`, `threshold`） |
| `notes` | 自由記述の補足情報 |

## セットアップウィザード（推奨）

対話型ウィザードで Input-1 の作成から Spec 凍結まで一括で実行できる。手動で YAML を書く必要がなく、環境の自動検出（GPU、SLURM）も行われる。

```bash
sera setup                         # 対話型ウィザード（日本語）
sera setup --lang en               # 英語版
sera setup --resume                # 中断したウィザードの再開
sera setup --from-input1 input1.yaml  # 既存 Input-1 から Phase 0 以降を実行
sera setup --skip-phase0           # Phase 0 をスキップ
```

### ウィザードの全体フロー

ウィザードは 3 フェーズ・11 ステップで構成される:

**Phase A: Input-1 作成（Step 1-7）**

| Step | 内容 | 入力例 |
|------|------|--------|
| 1 | データ情報 | データセット名、パス、形式、サイズ |
| 2 | 研究分野 | `ML` / `classification` |
| 3 | タスク定義 | `Classify iris species` |
| 4 | 目標設定 | `maximize accuracy`（方向は自動推定） |
| 5 | 制約条件 | `inference_time_ms <= 100` |
| 6 | 備考 | 自由記述 |
| 7 | プレビュー | Input-1 YAML を確認・編集 |

**Phase B: 先行研究（Step 8-9）**

| Step | 内容 |
|------|------|
| 8 | Phase 0 パラメータ設定・先行研究収集の実行 |
| 9 | 収集した論文のレビュー（クラスタ、ベースライン候補、教師論文の確認） |

**Phase C: Spec 確定（Step 10-11）**

Step 10 は 8 つのサブステップで全 Spec を生成する:

| サブステップ | 対象 | 内容 |
|------------|------|------|
| 10a | ProblemSpec | LLM が問題定義を生成。目標、制約、操作変数をフィールド単位で編集可能 |
| 10b | ModelSpec | GPU/SLURM 環境を自動検出。LLM プロバイダ選択（**デフォルト: local**）、ベースモデル、LoRA パラメータ |
| 10c | ResourceSpec | 実行バックエンド選択（local/slurm/docker）、GPU 設定、MCP サーバー |
| 10d | PlanSpec | 報酬方式（HiPER/MT-GRPO 等）、ECHO 有効化、Agent 関数の有効/無効 |
| 10e | ExecutionSpec | 探索パラメータ（max_nodes, repeats）、**PPO 学習（デフォルト: 有効）** |
| 10f | LanguageConfig | 実験言語（Python/R/Julia/C++/Rust/Go）、コンパイル設定、**複数ファイル（デフォルト: 有効）** |
| 10g | DependencyConfig | 依存管理ツール（pip/conda/cargo/cmake/go_mod）、LLM 生成ビルドファイル（オプション） |
| 10h | ContainerConfig | SLURM+コンテナ統合（Singularity/Apptainer/Docker）。executor=slurm 時のみ表示 |

Step 11 で全 Spec を最終確認し、ExecutionSpec を SHA-256 ハッシュでロックする。

### ウィザード実行例（Python + ローカル GPU の最小構成）

```text
$ sera setup

━━━ SERA セットアップウィザード ━━━

[Step 1/11] データ情報
  データの説明: > UCI Iris dataset
  ファイルパス: > ./data/iris.csv

    ... Step 2-9 省略 ...

[Step 10/11] Spec確定
  GPU detected: NVIDIA A100 80GB
  SLURM detected: available

  [1/8] ProblemSpec
    Generating ProblemSpec via LLM... ✓
    ProblemSpec action: > Confirm

  [2/8] ModelSpec
    LLM provider: > local
    Base model ID: > Qwen/Qwen2.5-Coder-7B-Instruct
    LoRA rank: > 16
    LoRA alpha: > 32

  [3/8] ResourceSpec
    Executor type: > slurm

  [4/8] PlanSpec
    Reward method: > hiper
    Enable ECHO? > Y

  [5/8] ExecutionSpec
    Enable PPO learning? > Y
    max_nodes: > 100
    repeats: > 3

  [6/8] LanguageConfig
    Experiment language: > python
    Allow multi-file projects? > Y

  [7/8] DependencyConfig
    Configure dependency management? > N

  [8/8] ContainerConfig (SLURM)
    Use container on SLURM? > N

[Step 11/11] 最終確認
  全 Spec を確定・ロックしますか？ > Y
  ✓ All specs frozen to sera_workspace/specs
```

### ウィザード実行例（C++ + SLURM + Singularity）

コンパイル型言語とコンテナを使う場合の Step 10 の抜粋:

```text
  [6/8] LanguageConfig
    Experiment language: > cpp
    Compile command: > g++
    Compile flags (space-separated): > -O2 -std=c++17 -Wall
    Link flags (space-separated): > -lm -lpthread
    Output binary name: > experiment
    Build timeout (sec): > 120
    Allow multi-file projects? > Y

  [7/8] DependencyConfig
    Configure dependency management? > Y
    Dependency manager: > cmake
    Build file name: > CMakeLists.txt
    Let LLM generate build file per experiment? > Y
    Install timeout (sec): > 300

  [8/8] ContainerConfig (SLURM)
    Use container on SLURM? > Y
    Container runtime: > singularity
    Container image: > /shared/images/sera_pytorch.sif
    Enable GPU passthrough? > Y
    Bind mounts: > /data:/data:ro,/scratch:/scratch
```

### デフォルト設定

ウィザードのデフォルト値は、SERA の全機能が有効になるよう設計されている:

| 設定 | デフォルト | 意味 |
|------|-----------|------|
| `agent_llm.provider` | `local` | ローカル GPU モデル（LoRA 学習に必須） |
| `learning.enabled` | `true` | PPO + LoRA 差分継承が有効 |
| `multi_file` | `true` | LLM が複数ソースファイルを生成可能 |
| ストリーミング実行 | `true` | 全 Executor でリアルタイム出力キャプチャ |
| `reward.method` | `hiper` | HiPER 3 層階層 PPO |
| `echo.enabled` | `true` | 失敗知識の再利用 |

途中で Ctrl+C を押すと状態が `.wizard_state.json` に保存され、`sera setup --resume` で中断箇所から再開可能。

## 最小ワークフロー（手動）

以下の CLI コマンドで全パイプラインを順に実行する。

```bash
# ステップ 1: ワークスペース初期化
sera init input1.yaml

# ステップ 2: Phase 0 - 先行研究収集
sera phase0-related-work

# ステップ 3: Phase 1 - Spec 生成・凍結
sera freeze-specs

# ステップ 4: Phase 2-6 - 研究ループ（探索・実験・評価・PPO・枝刈り）
sera research

# ステップ 5: Phase 7 - 論文生成
sera generate-paper

# ステップ 6: Phase 8 - 論文評価・改善ループ
sera evaluate-paper

# ステップ 7: 最良成果物のエクスポート
sera export-best
```

### 主要な CLI オプション

```bash
# freeze-specs の主要オプション
sera freeze-specs \
  --max-nodes 100 \          # 最大ノード数（デフォルト: 100）
  --repeats 3 \              # 実験繰り返し回数（デフォルト: 3）
  --rank 16 \                # LoRA ランク（デフォルト: 16）
  --alpha 32 \               # LoRA alpha（デフォルト: 32）
  --lr 1e-4 \                # 学習率（デフォルト: 1e-4）
  --lcb-coef 1.96 \          # LCB 係数（デフォルト: 1.96）
  --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \  # ベースモデル
  --executor local \          # 実行バックエンド（local/slurm/docker）
  --agent-llm "local:same_as_base"  # エージェント LLM 設定

# 研究ループの再開
sera research --resume

# ノードの状態確認
sera status
sera show-node <node_id>

# 特定ノードの実験再実行
sera replay <node_id> <seed>

# Spec 整合性チェック
sera validate-specs

# 探索木の可視化（インタラクティブ HTML 生成）
sera visualize
sera visualize --step 50           # 特定ステップ
sera visualize --output tree.html  # 出力先指定
```

## デフォルトのワークスペース構造

`sera init` は以下のディレクトリ構造を `./sera_workspace`（デフォルト）に作成する。

```
sera_workspace/
  specs/                     # 10 個の YAML ファイル + execution_spec.yaml.lock
    input1.yaml              #   Input-1 のコピー
    related_work_spec.yaml   #   先行研究 Spec
    paper_spec.yaml          #   論文 Spec
    paper_score_spec.yaml    #   論文評価 Spec
    teacher_paper_set.yaml   #   教師論文セット
    problem_spec.yaml        #   問題 Spec（LLM 生成）
    model_spec.yaml          #   モデル Spec
    resource_spec.yaml       #   リソース Spec
    plan_spec.yaml           #   計画 Spec（LLM 生成）
    execution_spec.yaml      #   実行 Spec（凍結レイヤ）
    execution_spec.yaml.lock #   SHA-256 ハッシュロック
  related_work/              # Phase 0 の API クエリログと結果
    results/
    teacher_papers/
  lineage/nodes/             # LoRA デルタ safetensors ファイル
  runs/                      # ノードごとの実験出力
    <node_id>/
      experiment.{ext}       #   生成されたエントリポイント（.py, .cpp, .rs 等）
      {補助ファイル}          #   multi_file=true 時の追加ソース（utils.py, solver.h 等）
      metrics.json           #   実験結果
      stdout.log
      stderr.log
      {build_file}           #   ビルド/依存ファイル（requirements.txt, CMakeLists.txt 等）
      build_stdout.log       #   コンパイルログ（compiled=true 時）
      install_stdout.log     #   依存インストールログ（dependency 設定時）
  logs/                      # JSONL ログファイル
    agent_llm_log.jsonl      #   全 LLM 呼び出し記録
    search_log.jsonl         #   探索ステップ記録
    eval_log.jsonl           #   評価記録
    ppo_log.jsonl            #   PPO 更新記録
  checkpoints/               # 探索状態スナップショット
  outputs/best/              # 最良成果物エクスポート先
  paper/                     # 生成論文
    paper.md                 #   論文本文（Markdown）
    paper.bib                #   参考文献
    figures/                 #   生成図表（PNG）
```

## トラブルシューティング

| 症状 | 原因 | 対処法 |
|------|------|--------|
| Phase 0 で論文が見つからない | API キー未設定 | Phase 0 はプロバイダチェーン（Semantic Scholar -> CrossRef -> arXiv -> Web）を順に試行する。`SEMANTIC_SCHOLAR_API_KEY` の設定を推奨 |
| `exit code 2` で異常終了 | ExecutionSpec の改竄検知 | `execution_spec.yaml` が凍結後に変更された。`sera freeze-specs` を再実行して Spec を再生成する |
| 実験がタイムアウトする | 実行時間超過 | `ResourceSpec` の `sandbox.experiment_timeout_sec`（デフォルト: 3600秒）を確認・増加する |
| OOM (Out of Memory) エラー | GPU メモリ不足 | `ModelSpec` の `base_model.load_method` を `"4bit"` に変更して量子化を有効にする。または小さいモデルを使用する |
| PPO 更新がスキップされる | `learning.enabled` が `False` またはローカル GPU モデル未使用 | PPO 学習には `agent_llm.provider="local"` かつ `learning.enabled=True` が必要 |
| チェックポイントから再開したい | SIGINT 等による中断 | `sera research --resume` で最新チェックポイントから再開可能。SIGINT 時は自動でチェックポイント保存後 `exit(20)` する |
| Spec 整合性エラー | YAML ファイルの不整合 | `sera validate-specs` でチェック。問題がある場合は `sera freeze-specs` を再実行する |

## 次のステップ

- [ワークフロー詳細](workflow.md) -- 全 8 フェーズの詳細な説明
- [アーキテクチャガイド](architecture.md) -- デュアルツリー設計と内部構造
- [設定リファレンス](configuration.md) -- 全設定パラメータの説明
