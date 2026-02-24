# SERA 要件定義書 — ミッション・設計原則

> 本ファイルは TASK.md v12.4 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 0. ミッション

任意の研究テーマに対して、

1) Web/API による先行研究調査（Scholar優先）
2) 研究仕様（Spec群）を **動的に確定**
3) Best-First 木構造探索で仮説/実験案を探索
4) 実験を実行しログを収集
5) 統計評価（LCB等）で有望枝を選択
6) PPO（LoRAのみ更新）でエージェントを専門化
7) LoRA差分継承（delta inheritance）で多数分岐を保持
8) AI-Scientist型ワークフローを参考に論文生成
9) 論文評価基準（PaperScoreSpec）に基づき改善ループ

を自律実行する研究エージェントを実装せよ。

### 0.1 エージェントの定義

SERAの「エージェント」とは、**コード生成・ツール呼び出しが可能なLLM（ベースモデル）にLoRAアダプタを装着した自律エージェント**である。
このエージェントは以下の行為を遂行する：

| 行為 | Phase | 具体的出力 |
|------|-------|-----------|
| ツール呼び出し | Phase 0: 文献検索API、Phase 2: 仮説検証ツール、Phase 3: デバッグツール | `ToolCall`（構造化ツール呼び出し。Phase Cで有効化） |
| 仮説生成 | Phase 2 | 自然言語の仮説文 + 実験条件の差分JSON |
| 実験コード生成 | Phase 3 | `experiment.*`（ProblemSpec.languageに基づく多言語対応スクリプト。Python/R/Julia/Go/C++/bash等） |
| 結果分析 | Phase 4 | metrics.jsonの解釈、次ステップ提案 |
| 論文執筆 | Phase 7 | Markdown/LaTeX形式の論文草稿（エージェント的反復改善） |
| 自己改善 | Phase 5 | PPO（MT-GRPOターン報酬）によるLoRA更新を通じた方策改善 |

エージェントのLLM呼び出しは全て `AgentLLM` クラスを経由する（§21参照）。
ベースモデルは変更せず、**LoRAアダプタのみが探索木のノードごとに分岐・専門化**される。
ツール呼び出し機能は Phase C（§26）で有効化され、`GenerationOutput` 型で統一的に返される（付録C参照）。

---

## 1. 設計原則（Non-negotiable）

1. **テーマ非依存**：学習は研究対象そのものではなく、実行エージェント内部（方策/生成能力）に適用される。
2. **二重木構造**：外部探索木（仮説/実験案）と内部LoRA系譜木（専門化）を同期して管理する。
3. **LoRA差分継承が標準**：ノードは親との差分Δのみ保存し、必要時に累積復元する。
4. **統計的妥当性必須**：反復評価＋SE＋LCB（またはUCB/CI）を基本にする。
5. **再現性保証**：seed、モデルrevision、APIクエリと取得日時、実行コマンド、環境情報を必ず保存。
6. **実行前固定**：Phase 2/3/4/5 に関わる既定値（探索/評価/学習/剪定/停止/予算）は Phase 1 で **ExecutionSpecとして固定**し、以後変更不可。
7. **既定値＋引数上書き**：既定値を持つが、CLI引数またはSpecで上書き可能。上書きは Phase 1 までに確定し、以降は固定。
8. **docs/必須**：Quickstart/Workflow/Architecture/各モジュール詳細/先行研究整理を `docs/` に含める。
9. **変数可変性の三層分離**：システム内の変数は以下の3層に厳密に分離される。層の境界を越えた変更は禁止。
10. **ターンレベル信用割当**：各Phaseの出力品質を個別に報酬化し（MT-GRPO、§26.4.2）、フェーズ間の信用割当を明示的に行う。PPO更新時に `R = Σ_t(w_t * r_turn_t) - penalties` で集約される。
11. **失敗知識の再利用**：失敗ノードの知識をECHO軽量版（§26.4.3）で抽出し、兄弟ノードのコンテキストに注入する。失敗は単に破棄するのではなく、後続探索の情報源として活用する。

| 層 | 所属Spec | Phase 2以降の可変性 | 例 |
|---|---------|-------------------|---|
| **固定層（Frozen）** | ExecutionSpec | **完全不変** | lr, clip_range, repeats, lcb_coef, lambda_cost, beta, max_nodes, rank, alpha |
| **操作層（Manipulated）** | ProblemSpec.manipulated_variables | **分岐生成で変更可能**（ホワイトリスト） | 実験のlearning_rate, batch_size, method, data_augmentation等 |
| **導出層（Derived）** | 実行時自動計算 | **自動決定**（手動変更不可） | priority, mu, se, lcb, feasible, reward |

> **重要**: 固定層と操作層の区別を厳守すること。ExecutionSpec.learning.lr（PPOの学習率）は固定層であり、ProblemSpec.manipulated_variables に定義された learning_rate（実験対象モデルの学習率）は操作層である。名前が類似していても層が異なる。

---
