
# SERA 要件定義書 — 目次

> **原本**: `TASK.md` v12.4 を機能単位で分割したもの。

---

## 基盤設計

| ファイル | 内容 | 原本セクション |
|---------|------|--------------|
| [00_mission_principles.md](./00_mission_principles.md) | ミッション・エージェント定義・設計原則 | §0–1 |
| [01_overview_input.md](./01_overview_input.md) | 全体フェーズ概要・データフロー・入力仕様 | §2–3 |

## Phase別仕様

| ファイル | 内容 | 原本セクション |
|---------|------|--------------|
| [02_phase0_related_work.md](./02_phase0_related_work.md) | Phase 0: 先行研究フェーズ（Web/API） | §4 |
| [03_phase1_spec.md](./03_phase1_spec.md) | Phase 1: Spec確定・ExecutionSpec固定 | §5 |
| [04_phase2_search.md](./04_phase2_search.md) | Phase 2: 探索木生成（Best-First） | §6 |
| [05_phase3_execution.md](./05_phase3_execution.md) | Phase 3: 実験実行（Executor） | §7 |
| [06_phase4_evaluation.md](./06_phase4_evaluation.md) | Phase 4: 統計評価（Evaluator） | §8 |
| [07_phase5_learning.md](./07_phase5_learning.md) | Phase 5: 学習（PPO + LoRA差分継承） | §9 |
| [08_phase6_lineage.md](./08_phase6_lineage.md) | Phase 6: 系譜管理・剪定 | §10 |
| [09_phase7_paper.md](./09_phase7_paper.md) | Phase 7: 論文生成（AI-Scientist-v2参考） | §11 |
| [10_phase8_paper_eval.md](./10_phase8_paper_eval.md) | Phase 8: 論文評価・改善ループ | §12 |

## システム設定・運用

| ファイル | 内容 | 原本セクション |
|---------|------|--------------|
| [11_defaults_directory_docs.md](./11_defaults_directory_docs.md) | 既定値一覧・ディレクトリ構成・docs要件 | §13–15 |
| [12_cli_logging.md](./12_cli_logging.md) | CLI仕様・ログ仕様 | §16–17 |
| [13_requirements_mvp.md](./13_requirements_mvp.md) | 非機能要件・受け入れ基準・MVP優先順位 | §18–20 |
| [14_module_structure.md](./14_module_structure.md) | モジュール構成（ソースコード構造） | §21 |

## 付録・実装ガイド

| ファイル | 内容 | 原本セクション |
|---------|------|--------------|
| [15_appendix.md](./15_appendix.md) | 付録A（先行研究）・B（禁止事項）・C（AgentLLMインターフェース） | 付録A–C |
| [16_implementation_guide.md](./16_implementation_guide.md) | 実装手順書（Step 0–12 + テスト戦略） | §22 |

## 拡張仕様

| ファイル | 内容 | 原本セクション |
|---------|------|--------------|
| [17_multilang.md](./17_multilang.md) | 多言語実験サポート | §23 |
| [18_slurm.md](./18_slurm.md) | SLURM実行パイプライン | §24 |
| [19_agent_model.md](./19_agent_model.md) | Agentモデル拡張（マルチモデル対応） | §25 |
| [20_tool_using_agent.md](./20_tool_using_agent.md) | Tool-Using Agent拡張（HiPER + ECHO統合） | §26 |

---

## バージョン履歴

| バージョン | 変更内容 |
|-----------|---------|
| v12.4 | 全面Agent対応更新: §0-§22を tool-calling + MT-GRPO + ECHO軽量版前提に改訂 |
| v12.3 | §25 Agentモデル拡張、§26 Tool-Using Agent拡張（HiPER+ECHO）追加 |
| v12.2 | §24 SLURM実行パイプライン追加 |
| v12.1 | §23 多言語実験サポート追加 |
| v12.0 | 初版（§0–22 + 付録A–C） |
