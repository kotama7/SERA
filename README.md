# SERA — Self-Evolving Research Agent

任意の研究テーマに対して、先行研究調査から仮説探索・実験実行・統計評価・エージェント自己改善・論文生成までを自律的に実行する研究エージェントシステム。

## 主要機能

- **先行研究調査 (Phase 0):** Semantic Scholar / arXiv / CrossRef / Web 検索による文献収集・クラスタリング
- **Spec 確定 (Phase 1):** 研究仕様（PaperSpec, ProblemSpec, ExecutionSpec 等）を動的に生成・固定
- **Best-First 探索 (Phase 2):** 仮説・実験案を木構造で探索（Draft / Debug / Improve オペレータ）
- **実験実行 (Phase 3):** LLM による実験コード生成 → ローカル / Docker / Slurm でサンドボックス実行
- **統計評価 (Phase 4):** 反復実行 + SE + LCB による有望枝の選択
- **PPO 自己改善 (Phase 5):** LoRA アダプタのみを PPO で更新しエージェントを専門化
- **LoRA 系譜管理 (Phase 6):** 差分継承（delta inheritance）による多数分岐の効率的保持・剪定
- **論文生成・評価 (Phase 7–8):** AI-Scientist 型ワークフローで論文草稿を生成し改善ループを実行

## アーキテクチャ

二重木構造を採用:

- **外部探索木:** 仮説・実験案の解空間を Best-First で探索
- **内部 LoRA 系譜木:** エージェントの LoRA アダプタの分岐・専門化履歴を管理

詳細は [docs/architecture.md](docs/architecture.md) を参照。

## 必要環境

- Python 3.11+
- CUDA 対応 GPU（推奨: NVIDIA A100 / H100）
- 文献検索 API キー（Semantic Scholar 等）

## インストール

```bash
git clone https://github.com/your-org/sera.git
cd sera
pip install -e ".[dev]"
```

## 環境変数

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-s2-key"
export CROSSREF_MAILTO="you@example.com"
export SERPAPI_KEY="your-serpapi-key"        # optional
export OPENAI_API_KEY="your-openai-key"     # optional
export HF_TOKEN="your-huggingface-token"    # optional
```

## 使い方

```bash
# Input-1 YAML から研究を開始
sera init input1.yaml

# Phase 0: 先行研究調査
sera phase0

# Phase 1: Spec 確定
sera phase1

# 全フェーズ自動実行
sera research input1.yaml
```

詳細は [docs/quickstart.md](docs/quickstart.md) を参照。

## プロジェクト構成

```
src/sera/
├── agent/          # AgentLLM・プロンプトテンプレート
├── specs/          # 各種 Spec 定義 (Pydantic モデル)
├── phase0/         # 先行研究調査 (API クライアント・クラスタリング・ランキング)
├── phase1/         # Spec 構築・凍結
├── search/         # 探索木 (SearchNode, SearchManager, Priority)
├── execution/      # 実験実行 (Local / Docker / Slurm)
├── evaluation/     # 統計評価 (LCB, 実行可能性判定)
├── learning/       # PPO 学習 (LoRA-only)
├── lineage/        # LoRA 系譜管理・剪定・キャッシュ
├── paper/          # 論文生成・評価・図表・引用
├── commands/       # CLI コマンド群
├── utils/          # シード・ハッシュ・ログ・チェックポイント
└── cli.py          # Typer エントリポイント
tests/              # pytest テスト群
docs/               # ドキュメント
```

## テスト

```bash
pytest
```

## ライセンス

TBD
