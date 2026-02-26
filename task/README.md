
# SERA 要件定義書 — 目次

> **原本**: `TASK.md` v13.1 を機能単位で分割したもの。

---

## 基盤設計

| ファイル | 内容 | セクション |
|---------|------|-----------|
| [00_mission_principles.md](./00_mission_principles.md) | ミッション・エージェント定義・設計原則 | §0–1 |
| [01_overview_input.md](./01_overview_input.md) | 全体フェーズ概要・データフロー・入力仕様 | §2–3 |

## Phase別仕様

| ファイル | 内容 | セクション |
|---------|------|-----------|
| [02_phase0_related_work.md](./02_phase0_related_work.md) | Phase 0: 先行研究フェーズ（Web/API） | §4 |
| [03_phase1_spec.md](./03_phase1_spec.md) | Phase 1: Spec確定・ExecutionSpec固定・agent_commands定義 | §5 |
| [04_phase2_search.md](./04_phase2_search.md) | Phase 2: 探索木生成（Best-First） | §6 |
| [05_phase3_execution.md](./05_phase3_execution.md) | Phase 3: 実験実行（Executor）+ 多言語サポート | §7 |
| [06_phase4_evaluation.md](./06_phase4_evaluation.md) | Phase 4: 統計評価（Evaluator） | §8 |
| [07_phase5_learning.md](./07_phase5_learning.md) | Phase 5: 学習（PPO + LoRA差分継承） | §9 |
| [08_phase6_lineage.md](./08_phase6_lineage.md) | Phase 6: 系譜管理・剪定 | §10 |
| [09_phase7_paper.md](./09_phase7_paper.md) | Phase 7: 論文生成（AI-Scientist-v2参考） | §11 |
| [10_phase8_paper_eval.md](./10_phase8_paper_eval.md) | Phase 8: 論文評価・改善ループ | §12 |

## システム設定・運用

| ファイル | 内容 | セクション |
|---------|------|-----------|
| [11_defaults_directory_docs.md](./11_defaults_directory_docs.md) | 既定値一覧・ディレクトリ構成・docs要件 | §13–15 |
| [12_cli_logging.md](./12_cli_logging.md) | CLI仕様・ログ仕様 | §16–17 |
| [13_requirements_mvp.md](./13_requirements_mvp.md) | 非機能要件・受け入れ基準・MVP優先順位 | §18–20 |
| [14_module_structure.md](./14_module_structure.md) | モジュール構成（ソースコード構造） | §21 |

## 付録・実装ガイド

| ファイル | 内容 | セクション |
|---------|------|-----------|
| [15_appendix.md](./15_appendix.md) | 付録A（先行研究）・B（禁止事項）・C（AgentLLMインターフェース） | 付録A–C |
| [16_implementation_guide.md](./16_implementation_guide.md) | 実装手順書（Step 0–12 + テスト戦略） | §22 |

## 拡張仕様

| ファイル | 内容 | セクション |
|---------|------|-----------|
| [17_slurm.md](./17_slurm.md) | SLURM実行パイプライン | §23 |
| [18_agent_model.md](./18_agent_model.md) | Agentモデル拡張（マルチモデル対応） | §24 |
| [19_tool_using_agent.md](./19_tool_using_agent.md) | Tool-Using Agent拡張（HiPER + ECHO統合） | §25 |
| [20_setup_wizard.md](./20_setup_wizard.md) | 対話型セットアップウィザード | §26 |
| [21_agent_functions.md](./21_agent_functions.md) | Agent Function System（タスク定義 + 統一エントリポイント） | §27 |
| [22_tool_execution.md](./22_tool_execution.md) | Tool Execution Engine（ツール実行基盤 + AgentLoop + MCP） | §28 |
| [23_visualization.md](./23_visualization.md) | 探索木可視化ツール（HTML出力） | §29 |

---

## バージョン履歴

> v13.1 で旧§23（多言語実験サポート）を §7.3.1 に統合しファイルを削除。以降のセクション番号を詰めて §23–§29 に再採番。以下の履歴は再採番後のセクション番号で記載。

| バージョン | 変更内容 |
|-----------|---------|
| v13.1 | 旧 `17_multilang.md`（多言語実験サポート）を §7.3.1 に統合・削除。ファイル番号 17–23 / セクション番号 §23–§29 を再採番し、ファイル間の相互参照を全て更新 |
| v13.0 | §5.7 PlanSpec に `agent_commands` 追加（§5.8）: ツール18種・関数19種の有効化リスト、Phase別ツールマップ、関数→ツールバインディング、ループ設定をPhase 1で凍結。実装状況を実態に更新。全ファイルヘッダをv13.0に統一 |
| v12.9 | §27/§28/§25 再構成: §27を統一エントリポイント化（allowed_tools/loop_config追加）、§28をツール実行基盤に特化、学習統合を§25.5.3に集約 |
| v12.8 | §28.14 ツール使用経験からの学習統合追加（PPORolloutV3, ToolCallRecord, tool_aware報酬, HiPER拡張） |
| v12.7 | §28 Tool Execution Engine追加 |
| v12.6 | §27 Agent Function Registry追加 |
| v12.5 | §26 対話型セットアップウィザード追加 |
| v12.4 | 全面Agent対応更新: §0–§22を tool-calling + MT-GRPO + ECHO軽量版前提に改訂 |
| v12.3 | §24 Agentモデル拡張、§25 Tool-Using Agent拡張（HiPER+ECHO）追加 |
| v12.2 | §23 SLURM実行パイプライン追加 |
| v12.1 | 多言語実験サポート追加（後に §7.3.1 へ統合） |
| v12.0 | 初版（§0–22 + 付録A–C） |
