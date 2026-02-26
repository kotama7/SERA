# TASK.md
# Self-Evolving Research Agent（SERA）
# 完全最終要件定義書 v13.1

> **本ファイルは分割されました。** 各セクションの詳細は [`task/`](./task/) ディレクトリ以下を参照してください。

---

## 目次

→ **[task/README.md](./task/README.md)** を参照

### ファイル一覧

| ファイル | 内容 |
|---------|------|
| [task/00_mission_principles.md](./task/00_mission_principles.md) | §0–1: ミッション・設計原則 |
| [task/01_overview_input.md](./task/01_overview_input.md) | §2–3: 全体フェーズ・入力仕様 |
| [task/02_phase0_related_work.md](./task/02_phase0_related_work.md) | §4: Phase 0 先行研究 |
| [task/03_phase1_spec.md](./task/03_phase1_spec.md) | §5: Phase 1 Spec確定 |
| [task/04_phase2_search.md](./task/04_phase2_search.md) | §6: Phase 2 探索木生成 |
| [task/05_phase3_execution.md](./task/05_phase3_execution.md) | §7: Phase 3 実験実行 |
| [task/06_phase4_evaluation.md](./task/06_phase4_evaluation.md) | §8: Phase 4 統計評価 |
| [task/07_phase5_learning.md](./task/07_phase5_learning.md) | §9: Phase 5 学習（PPO + LoRA） |
| [task/08_phase6_lineage.md](./task/08_phase6_lineage.md) | §10: Phase 6 系譜管理・剪定 |
| [task/09_phase7_paper.md](./task/09_phase7_paper.md) | §11: Phase 7 論文生成 |
| [task/10_phase8_paper_eval.md](./task/10_phase8_paper_eval.md) | §12: Phase 8 論文評価・改善 |
| [task/11_defaults_directory_docs.md](./task/11_defaults_directory_docs.md) | §13–15: 既定値・ディレクトリ・docs |
| [task/12_cli_logging.md](./task/12_cli_logging.md) | §16–17: CLI・ログ仕様 |
| [task/13_requirements_mvp.md](./task/13_requirements_mvp.md) | §18–20: 非機能要件・受け入れ基準・MVP |
| [task/14_module_structure.md](./task/14_module_structure.md) | §21: モジュール構成 |
| [task/15_appendix.md](./task/15_appendix.md) | 付録A–C: 先行研究・禁止事項・AgentLLM |
| [task/16_implementation_guide.md](./task/16_implementation_guide.md) | §22: 実装手順書 |
| [task/17_slurm.md](./task/17_slurm.md) | §23: SLURM実行パイプライン |
| [task/18_agent_model.md](./task/18_agent_model.md) | §24: Agentモデル拡張 |
| [task/19_tool_using_agent.md](./task/19_tool_using_agent.md) | §25: Tool-Using Agent拡張（HiPER + ECHO） |
| [task/20_setup_wizard.md](./task/20_setup_wizard.md) | §26: 対話型セットアップウィザード |
| [task/21_agent_functions.md](./task/21_agent_functions.md) | §27: Agent Function System（タスク定義 + 統一エントリポイント） |
| [task/22_tool_execution.md](./task/22_tool_execution.md) | §28: Tool Execution Engine（ツール実行基盤 + AgentLoop） |
| [task/23_visualization.md](./task/23_visualization.md) | §29: 探索木可視化ツール |
