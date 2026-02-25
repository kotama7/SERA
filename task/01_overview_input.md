# SERA 要件定義書 — 全体フェーズ・入力仕様

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 2. 全体フェーズ

```text
Phase 0: 先行研究フェーズ（Web/API）
Phase 1: Spec確定（PaperSpec/PaperScoreSpec/TeacherPaperSet + ExecutionSpec固定）
Phase 2: 探索木生成（Best-First）
Phase 3: 実験実行（Executor）
Phase 4: 統計評価（Evaluator: repeats, SE, LCB, sequential eval）
Phase 5: 学習（PPO: LoRA-only）
Phase 6: 系譜管理・剪定（delta lineage + squash + pruning）
Phase 7: 論文生成（AI-Scientist参考）
Phase 8: 論文評価・改善（PaperScoreSpec, loop）
```

### 2.1 Phase間データフロー（具体）

```text
Input-1 (YAML)
  │
  ▼
Phase 0 ── LLM がクエリ生成 ──► Scholar API/Semantic Scholar/CrossRef/arXiv
  │                                     │
  │◄── papers[], clusters[], baselines ─┘
  │
  ▼
Phase 1 ── LLM が Spec群を草案 → ユーザ承認 or 自動確定
  │         出力: specs/*.yaml（9ファイル）
  │         ExecutionSpec は SHA-256 ハッシュを計算し specs/execution_spec.yaml.lock に保存
  │
  ▼
Phase 2 ── SearchManager が open_list から最高優先度ノードを pop
  │         LLM（AgentLLM）が子ノード候補を生成（仮説+実験条件 JSON）
  │         子ノードを open_list に追加
  │         ※ 失敗兄弟ノードの failure_context がある場合、improve プロンプトに注入（§6.8.2）
  │
  │── [r_turn: hypothesis_novelty] ──►
  ▼
Phase 3 ── Executor がノードの実験コードを LLM 生成 → サンドボックス内実行
  │         出力: runs/<node_id>/metrics.json + stdout.log + stderr.log
  │         失敗時: FailureKnowledgeExtractor → 兄弟ノードcontext注入（ECHO軽量版）
  │
  │── [r_turn: code_executability] ──►
  ▼
Phase 4 ── Evaluator が metrics.json を集約、μ/SE/LCB を計算
  │         逐次評価: 初回1回 → Top-k なら repeats まで追加
  │
  │── [r_turn: metric_improvement] ──►
  ▼
Phase 5 ── PPORolloutV2: ターンレベル報酬の集約 → LoRA-only 更新 → Δ保存
  │         条件: 評価済みノードが ppo_trigger_interval 個溜まるごと
  │         報酬: R = Σ_t(w_t * r_turn_t) - penalties（MT-GRPO、§26.4.2）
  │
  ▼
Phase 6 ── LineageManager が squash/prune を実行
  │         ループ: Phase 2 へ戻る（終了条件まで）
  │
  ▼
Phase 7 ── PaperComposer が EvidenceStore から証拠収集
  │         Step 1: ログ要約 → Step 2: 図生成・集約
  │         Step 3: 自動引用検索（Semantic Scholar、最大20ラウンド）
  │         Step 4: VLM 図記述生成（VLM有効時）
  │         Step 5: 論文1パス生成 + ライティング内反省ループ（最大3回、VLMレビュー含む）
  │         Step 6: 統合・最終出力
  │
  ▼
Phase 8 ── PaperEvaluator が PaperScoreSpec で採点
  │         アンサンブルレビュー（num_reviews_ensemble 件の独立レビュー）
  │         各レビュアーに反省ループ（num_reviewer_reflections 回）
  │         メタレビュー（Area Chair 集約）→ 改善指示 → Phase 7 再実行
           最大 paper_revision_limit 回ループ
```

---

## 3. 入力仕様（2種類）

### 3.1 Input-1（簡素版）
ユーザが与える最小入力。Input-1受領後、Phase 0 を経て Phase 1 で Input-2 を確定する。

```yaml
version: 1
data:
  description: ""     # データの性質（ログ、CSV、コード、PDF群…）
  location: ""        # パス/URI/リポジトリ
  format: ""          # "csv" | "json" | "parquet" | "code" | "pdf" | "mixed"
  size_hint: ""       # "small(<1GB)" | "medium(1-100GB)" | "large(>100GB)"
domain:
  field: ""           # "HPC", "materials", "software", "NLP", "CV", etc.
  subfield: ""        # より具体的な分野（"compiler optimization", "protein folding"等）
task:
  brief: ""           # 研究タスクの短い説明（1-3文）
  type: ""            # "optimization" | "prediction" | "generation" | "analysis" | "comparison"
goal:
  objective: ""       # 目標（"minimize runtime", "maximize accuracy", "minimize variance"...）
  direction: ""       # "minimize" | "maximize"
  baseline: ""        # 既知のベースライン値（あれば）
constraints:          # 制約条件（0個以上）
  - name: ""
    type: ""          # "ge" (>=) | "le" (<=) | "eq" (==) | "bool" (true/false)
    threshold: null
notes: ""             # 任意
```

### 3.2 Input-2（完全版）
Input-1 + Phase 0結果 + 実行前固定値を含む機械実行可能な仕様。以下を含む：

- `RelatedWorkSpec`（§4.4.1）
- `PaperSpec`（§4.4.2）
- `PaperScoreSpec`（§4.4.3）
- `TeacherPaperSet`（§4.4.4）
- `ProblemSpec`（§5.5）
- `ModelSpec`（§5.3）
- `ResourceSpec`（§5.6）
- `PlanSpec`（§5.7）
- `ExecutionSpec`（§5.4）

---
