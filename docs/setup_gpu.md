# GPU 環境セットアップガイド

SERA で local LLM（LoRA + PPO）を使用するには、CUDA 対応の PyTorch が GPU ノード上で動作している必要がある。**ログインノード（GPU なし）で `pip install` すると `torch.cuda.is_available() = False` になる**ため、必ず GPU のある計算ノード上でセットアップを行うこと。

## 問題の背景

PyTorch の `pip install` は実行環境の CUDA ドライバを検出してホイールを選択する。ログインノードに GPU がない場合:
- `torch.cuda.is_available()` が `False` を返す
- `AgentLLM` の local プロバイダが GPU を使用できない
- PPO 学習（Phase 5）が実行不可能

## 前提条件

- SLURM クラスタにアクセス可能であること
- GPU パーティション（例: `sx40`）が利用可能であること
- Python 3.11 以上がインストールされていること

## セットアップ手順

### 1. GPU ノード上で環境を構築

```bash
# srun で GPU ノード上でセットアップスクリプトを実行
srun --partition=sx40 --time=01:00:00 bash scripts/setup_env.sh
```

このスクリプトは以下を行う:
1. GPU の存在とCUDA バージョンを自動検出
2. `.venv` を作成（`SERA_VENV_DIR` で場所を変更可能）
3. 検出された CUDA バージョンに対応する PyTorch をインストール
4. `sera[dev,slurm]` をインストール
5. `torch.cuda.is_available() = True` であることを検証

### 2. 環境変数の設定

```bash
# API キー（Phase 0 の文献検索に必要）
export SEMANTIC_SCHOLAR_API_KEY="your-key"
export CROSSREF_EMAIL="you@example.com"

# オプション
export SERPAPI_API_KEY="your-key"        # Web 検索フォールバック
export OPENAI_API_KEY="your-key"         # OpenAI プロバイダ使用時
export ANTHROPIC_API_KEY="your-key"      # Anthropic プロバイダ使用時
export HF_TOKEN="your-token"             # HuggingFace モデルダウンロード
```

### 3. 研究の実行

```bash
# 方法 A: sbatch でジョブ投入（推奨）
sbatch scripts/run_research.sh

# 方法 B: srun でインタラクティブ実行
srun --partition=sx40 --time=20:00:00 bash -c '
    source .venv/bin/activate
    sera research --work-dir ./sera_workspace
'

# 方法 C: 中断した研究の再開
sbatch scripts/run_research.sh --resume
```

### 4. 論文生成・評価

```bash
# Phase 7 + Phase 8
sbatch scripts/run_paper.sh

# Phase 8（評価）のみ
sbatch scripts/run_paper.sh --eval-only
```

## スクリプト一覧

| スクリプト | 用途 |
|-----------|------|
| `scripts/setup_env.sh` | GPU ノード上で `.venv` を作成し CUDA 対応 PyTorch + SERA をインストール |
| `scripts/run_research.sh` | `sera research` を SLURM ジョブとして実行 |
| `scripts/run_paper.sh` | `sera generate-paper` + `sera evaluate-paper` を SLURM ジョブとして実行 |

## 環境変数によるカスタマイズ

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `SERA_VENV_DIR` | `./.venv` | .venv の作成場所 |
| `SERA_PYTHON` | `python3` | venv 作成に使用する Python |
| `SERA_CUDA_VERSION` | (自動検出) | PyTorch の CUDA ターゲット (例: `12.4`) |
| `SERA_WORK_DIR` | `./sera_workspace` | ワークスペースのパス |

## トラブルシューティング

### `torch.cuda.is_available()` が False

```bash
# 計算ノード上で確認
srun --partition=sx40 --time=00:05:00 bash -c '
    source .venv/bin/activate
    python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
'
```

原因と対処:
1. **ログインノードで実行している** → `srun` で GPU ノードに移動して実行
2. **CPU 版 PyTorch がインストールされている** → `.venv` を削除して `setup_env.sh` を再実行
3. **CUDA バージョン不一致** → `SERA_CUDA_VERSION` を明示的に指定して再セットアップ

### .venv の再作成

```bash
rm -rf .venv
srun --partition=sx40 --time=01:00:00 bash scripts/setup_env.sh
```

### ログインノードでのテスト実行（GPU 不要な操作）

ログインノードでも以下は実行可能:
- `sera init` — ワークスペース初期化
- `sera phase0-related-work` — 文献検索（API のみ、GPU 不要）
- `sera freeze-specs` — Spec 凍結
- `sera status` — 探索状態表示
- `sera validate-specs` — Spec 整合性チェック
- `pytest -m "not gpu" tests/` — GPU 不要テスト

GPU が必要な操作:
- `sera research` — Phase 2-6 ループ（local LLM 使用時）
- `sera generate-paper` — Phase 7（local LLM 使用時）
- `sera evaluate-paper` — Phase 8（local LLM 使用時）

**注意:** `agent_llm.provider` が `"openai"` や `"anthropic"` の場合は API 経由のため GPU 不要。
