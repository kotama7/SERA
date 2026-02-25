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

対話型ウィザードで Input-1 の作成から Spec 凍結まで一括で実行できる。

```bash
sera setup                         # 対話型ウィザード（日本語）
sera setup --lang en               # 英語版
sera setup --resume                # 中断したウィザードの再開
sera setup --from-input1 input1.yaml  # 既存 Input-1 から Phase 0 以降を実行
sera setup --skip-phase0           # Phase 0 をスキップ
```

ウィザードは以下の 11 ステップを順に実行する:

1. データ情報入力
2. 研究分野入力
3. タスク定義
4. 目標設定（方向の自動推定付き）
5. 制約条件追加
6. 備考
7. プレビュー確認
8. Phase 0 パラメータ設定・実行
9. 収集論文レビュー
10. Spec パラメータ設定（GPU/SLURM 自動検出）
11. Spec 凍結

途中で Ctrl+C を押すと状態が保存され、`--resume` で再開可能。

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
      experiment.py          #   生成されたスクリプト
      metrics.json           #   実験結果
      stdout.log
      stderr.log
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
