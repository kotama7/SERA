# SERA 要件定義書 — 全体フェーズ・入力仕様

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

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
  │         報酬: R = Σ_t(w_t * r_turn_t) - penalties（MT-GRPO、§25.4.2）
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

## 3.3 実装レビュー：Input-1 スキーマと下流との整合性

> **レビュー日**: 2026-02-25
> **対象ソース**: `specs/input1.py`, `phase0/related_work_engine.py`, `commands/wizard/steps/step4_goal.py`, `prompts/*.yaml`

---

### 3.3.1 【高】`goal.metric` フィールドの三者不整合

ウィザード・Pydantic モデル・下流コードの三者で `goal.metric` の扱いが食い違っている。

| コンポーネント | `goal.metric` の扱い | ソース位置 |
|---|---|---|
| **仕様書 (§3.1)** | **定義なし** | `01_overview_input.md:99-102` |
| **Input1Model (GoalConfig)** | **フィールドなし** | `specs/input1.py:51-56` |
| **ウィザード Step 4** | **収集している** (`goal["metric"]`) | `step4_goal.py:68` |
| **Phase 0 (related_work_engine)** | **アクセス試行** (`getattr(input1.goal, "metric", "")`) | `related_work_engine.py:536, 547-550` |
| **ProblemSpec (ObjectiveConfig)** | `metric_name` として定義 | `problem_spec.py:16` |
| **Phase 1 テンプレート** | `{goal_objective}` のみ使用、metric 個別参照なし | `phase1_templates.yaml:11` |

**データフロー上の問題**:

```
ウィザード Step 4 → input1_data["goal"]["metric"] = "accuracy"
                        ↓
init_cmd → Input1Model(**input1_data) → GoalConfig にmetricフィールドなし
                        ↓
            Pydantic が extra fields を無視（デフォルト動作）→ metric が消失
                        ↓
Phase 0 → getattr(input1.goal, "metric", "") → 常に "" を返す
                        ↓
Phase 0 → hasattr(input1.goal, "metric") → 常に False → dead code
```

ウィザードがユーザから `metric` を聞いているにも関わらず、`Input1Model` に到達した時点で値が消失し、Phase 0 で利用できない。

**修正方針**: `GoalConfig` に `metric` フィールドを追加する。このフィールドは Phase 1 で `ProblemSpec.objective.metric_name` にマッピングされる。

```python
# specs/input1.py — GoalConfig に追加
class GoalConfig(BaseModel):
    objective: str = Field(...)
    direction: Literal["minimize", "maximize"] = Field(...)
    metric: str = Field("score", description="Name of the primary metric (e.g., accuracy, runtime_sec)")
    baseline: str = Field("")
```

仕様書 (§3.1) の YAML にも追加:

```yaml
goal:
  objective: ""
  direction: ""
  metric: "score"     # 追加: 評価指標名（例: accuracy, runtime_sec）
  baseline: ""
```

### 3.3.2 【中】Phase 0 の `input1.goal.metric` / `input1.task.dataset` 参照がデッドコード

`related_work_engine.py` に、`Input1Model` に存在しないフィールドを参照するコードが3箇所ある。いずれも `getattr`/`hasattr` のデフォルト値により実行時エラーにはならないが、意図した機能が動作しない。

| 行 | コード | 期待フィールド | 状態 |
|---|---|---|---|
| 536 | `getattr(input1.goal, "metric", "")` | `goal.metric` | 常に `""` → baseline_candidates の metric name が空 |
| 547-550 | `hasattr(input1.goal, "metric")` + `input1.goal.metric` | `goal.metric` | 常に `False` → common_metrics にユーザ指標が含まれない |
| 561 | `getattr(input1.task, "dataset", None)` | `task.dataset` | 常に `None` → ユーザ指定データセットが dataset_mentions に含まれない |

**影響**:
- baseline_candidates の `reported_metric.name` が常に空文字
- ユーザが指定した metric が common_metrics に反映されず、Phase 1 テンプレートの `{common_metrics_json}` が空になる
- ユーザが指定した dataset が dataset 出現頻度に反映されない

**修正方針 (§3.3.1 と連携)**:
- `goal.metric` → §3.3.1 で `GoalConfig` に追加すれば自動的に解決
- `task.dataset` → `TaskConfig` に `dataset: str = ""` を追加するか、`data.description` から推定するロジックに変更

### 3.3.3 【中】ウィザード Step 4 で収集するが仕様に未定義のフィールド

ウィザード Step 4 (`step4_goal.py:68`) は `goal.metric` を収集しており、i18n にもメッセージが定義されている:

```python
# step4_goal.py:68
goal["metric"] = Prompt.ask(get_message("goal_metric", lang), default=goal.get("metric", "score"))
```

```python
# i18n.py
"goal_metric": "評価指標の名前を入力してください（例: accuracy, runtime_sec）",  # ja
"goal_metric": "Enter metric name (e.g., accuracy, runtime_sec)",                  # en
```

しかし仕様書 §26.5.4 (Step 4: 目標情報) の収集フィールド一覧には `goal.metric` が含まれていない:

| フィールド | 仕様 §26.5.4 |
|-----------|-------------|
| `goal.objective` | 定義あり |
| `goal.direction` | 定義あり |
| `goal.baseline` | 定義あり |
| **`goal.metric`** | **定義なし** |

**修正方針**: §26.5.4 の収集フィールド表に `goal.metric` を追加する（ウィザード実装が先行して正しい動作を行っている）。

### 3.3.4 【低】Input-1 → ProblemSpec への metric マッピングが未定義

ウィザードが `goal.metric` を収集し、`GoalConfig` にフィールドを追加した場合でも、Phase 1 の `SpecBuilder.build_problem_spec()` が `input1.goal.metric` → `ProblemSpec.objective.metric_name` へのマッピングを行うコードが存在しない。

現状の `spec_builder.py` は Input-1 全体を JSON シリアライズして LLM に渡し、LLM が `ProblemSpec` を生成する。LLM が `goal.metric` を `objective.metric_name` に正しくマッピングする保証はない。

**修正方針**: `spec_builder.py` の fallback ProblemSpec 生成時に `metric_name` を明示的にセットする:

```python
# spec_builder.py — fallback ProblemSpec 生成
defaults = ProblemSpecModel(
    title=...,
    objective=ObjectiveConfig(
        description=getattr(getattr(input1, "goal", None), "objective", ""),
        metric_name=getattr(getattr(input1, "goal", None), "metric", "score"),
        direction=getattr(getattr(input1, "goal", None), "direction", "maximize"),
    ),
)
```

### 3.3.5 【低】Input-1 プロンプトテンプレートの整合性

Phase 0/1/2/3 のプロンプトテンプレート (YAML) で使われるプレースホルダーと Input1Model フィールドの対応:

| テンプレート変数 | Input1 フィールド | 状態 |
|---|---|---|
| `{task_brief}` | `input1.task.brief` | 整合 |
| `{task_description}` | `input1.task.brief` (Phase 0 テンプレートでの別名) | 整合（呼び出し時に mapping） |
| `{field}` | `input1.domain.field` | 整合 |
| `{subfield}` | `input1.domain.subfield` | 整合 |
| `{data_description}` | `input1.data.description` | 整合 |
| `{data_format}` | `input1.data.format` | 整合 |
| `{data_size}` | `input1.data.size_hint` | 整合 |
| `{goal_objective}` | `input1.goal.objective` | 整合 |
| `{goal_direction}` | `input1.goal.direction` | 整合 |
| `{baseline}` | `input1.goal.baseline` | 整合 |
| `{constraints_json}` | `input1.constraints` (シリアライズ) | 整合 |
| `{notes}` | `input1.notes` | 整合 |
| `{metric_name}` | `problem_spec.objective.metric_name` (**Input-1 経由ではない**) | Phase 3 テンプレートで使用。Input-1 ではなく ProblemSpec から取得 |

テンプレート変数は Input1Model と整合している。`{metric_name}` は ProblemSpec 由来であり Input-1 直接参照ではないため問題なし。

---

### 3.3.6 修正優先度まとめ

| # | 問題 | 優先度 | 影響 |
|---|------|-------|------|
| 3.3.1 | `goal.metric` が仕様・モデル未定義だがウィザードが収集・Phase 0 が参照 | **高** | ウィザードで入力した metric 名が消失し、Phase 0/1 で利用不能 |
| 3.3.2 | Phase 0 の `goal.metric` / `task.dataset` 参照がデッドコード | **中** | baseline_candidates と common_metrics の質が低下 |
| 3.3.3 | ウィザード仕様 §26.5.4 に `goal.metric` 未記載 | **中** | 仕様と実装の乖離 |
| 3.3.4 | Input-1 → ProblemSpec の metric マッピングが未定義 | **低** | LLM 生成依存、fallback 時に metric_name がデフォルト "score" |
| 3.3.5 | プロンプトテンプレートの整合性 | — | **問題なし** |
