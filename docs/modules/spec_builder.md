# SpecBuilder / SpecFreezer / AllSpecs

Phase 1 のスペック生成・凍結を担当するモジュール群のドキュメント。

## 対象モジュール

| モジュール | ファイルパス |
|-----------|------------|
| `SpecBuilder` | `src/sera/phase1/spec_builder.py` |
| `SpecFreezer` | `src/sera/phase1/spec_freezer.py` |
| `AllSpecs` | `src/sera/specs/__init__.py` |

## 依存関係

- `sera.agent.agent_llm` (AgentLLM) -- LLM 呼び出し
- `sera.agent.prompt_templates` (`SPEC_GENERATION_PROMPT`) -- プロンプトテンプレート
- `sera.specs.*` -- 全スペックモデル (Pydantic v2)
- `sera.utils.hashing` (`compute_spec_hash`, `compute_adapter_spec_hash`) -- SHA-256 ハッシュ計算

---

## SpecBuilder

LLM を用いて Input-1 と Phase 0 出力から `ProblemSpec` / `PlanSpec` を生成するクラス。

### コンストラクタ

```python
def __init__(self, agent_llm: Any)
```

- `agent_llm`: LLM クライアント。`generate(prompt, purpose, temperature)` 非同期メソッドを持つオブジェクト。

### build_problem_spec(input1, related_work) -> dict

ProblemSpec を LLM で生成する非同期メソッド。

**処理フロー:**

1. `_build_context()` で Input-1 と RelatedWorkSpec を JSON 文字列に変換してコンテキストを構築
2. `SPEC_GENERATION_PROMPT` テンプレートに `spec_type="ProblemSpec"` とスキーマ説明（`_problem_spec_schema()`）を埋め込む
3. 最大 3 回リトライ（温度: 0.7, 0.8, 0.9 とインクリメント）
4. `_parse_json()` で応答から JSON を抽出
5. パース結果から `parsed.get("problem_spec", parsed)` でスペック部分を取得し、`ProblemSpecModel(**...)` でバリデーション
6. バリデーション成功時: `spec.model_dump()` を返す
7. バリデーション失敗時: 失敗メッセージをプロンプトに追記して次の試行に渡す

**フォールバック:**

全試行失敗時は `ProblemSpecModel(title=...)` でデフォルト値を生成。`title` は `input1` の `task.brief` フィールドから取得する（dict / オブジェクト双方に対応）。

### build_plan_spec(input1, problem_spec) -> dict

PlanSpec を LLM で生成する非同期メソッド。基本パターンは `build_problem_spec` と同一。

- コンテキスト: Input-1 と ProblemSpec の JSON 文字列
- 3 回リトライ（温度: 0.7, 0.8, 0.9）
- パース結果から `parsed.get("plan_spec", parsed)` でスペック部分を取得
- フォールバック: 引数なしの `PlanSpecModel().model_dump()`

### JSON 抽出ロジック (_parse_json)

```python
def _parse_json(self, response: str) -> dict | None
```

1. 正規表現 `` ```json ... ``` `` ブロックを検索（`re.DOTALL`）
2. マッチした場合は `json.loads()` でパース
3. マッチしない場合はレスポンス全体を `json.loads()` で直接パース
4. いずれも失敗した場合は `None` を返す

### スキーマ説明ヘルパー

- `_problem_spec_schema()`: ProblemSpec に必要なフィールド（title, objective, constraints, secondary_metrics, manipulated_variables, observed_variables, evaluation_design, experiment_template）を記述した文字列を返す
- `_plan_spec_schema()`: PlanSpec に必要なフィールド（search_strategy, branching, reward, logging, artifacts）を記述した文字列を返す

---

## SpecFreezer

全スペックを YAML ファイルとしてディスクに保存し、ExecutionSpec の SHA-256 ハッシュで改竄検知を行うクラス。

### freeze(specs, specs_dir: Path)

**処理フロー:**

1. `specs_dir` を作成（`mkdir(parents=True, exist_ok=True)`）
2. 10 個のスペックを YAML ファイルとして保存。各スペックは `model_dump()` があればそれを、なければそのまま dict として `yaml.dump()` する
3. ModelSpec のメタデータを自動補完:
   - `base_model.revision` が未設定の場合: `transformers.AutoConfig.from_pretrained(bm.id)` で `_commit_hash` を取得（例外時は `"unknown"`）
   - `compatibility.adapter_spec_hash`: `compute_spec_hash(adapter_dict)` で計算・設定
   - 更新後に `model_spec.yaml` を再保存
4. ExecutionSpec のハッシュ計算: `compute_spec_hash(exec_data)` で SHA-256 を算出
5. ハッシュを `execution_spec.yaml.lock` に書き込む

**ファイルマッピング（spec_mapping 辞書）:**

| ファイル名 | 属性名 |
|-----------|--------|
| `input1.yaml` | `input1` |
| `related_work_spec.yaml` | `related_work` |
| `paper_spec.yaml` | `paper` |
| `paper_score_spec.yaml` | `paper_score` |
| `teacher_paper_set.yaml` | `teacher_paper_set` |
| `problem_spec.yaml` | `problem` |
| `model_spec.yaml` | `model` |
| `resource_spec.yaml` | `resource` |
| `plan_spec.yaml` | `plan` |
| `execution_spec.yaml` | `execution` |

### verify(specs_dir: Path) -> bool

ExecutionSpec のハッシュ整合性を検証する。

1. `execution_spec.yaml` を `yaml.safe_load()` で読み込む
2. `execution_spec.yaml.lock` から保存済みハッシュを読み込む
3. `compute_spec_hash(data)` で再計算し、保存値と比較
4. 一致すれば `True`、不一致なら `False`（改竄検知ログを出力）
5. いずれかのファイルが存在しない場合は `False` を返す

---

## AllSpecs

10 個の全スペックを保持する **プレーン dataclass**（`@dataclasses.dataclass`、Pydantic モデルではない）。

### フィールド

```python
@dataclasses.dataclass
class AllSpecs:
    input1: Input1Model
    related_work: RelatedWorkSpecModel
    paper: PaperSpecModel
    paper_score: PaperScoreSpecModel
    teacher_paper_set: TeacherPaperSetModel
    problem: ProblemSpecModel
    model: ModelSpecModel
    resource: ResourceSpecModel
    plan: PlanSpecModel
    execution: ExecutionSpecModel
```

### I/O メソッド

- `load_from_dir(cls, specs_dir) -> AllSpecs`: `_SPEC_FILES` 辞書を用いて各 YAML ファイルから対応するモデルクラスの `from_yaml()` メソッドでロード
- `save_to_dir(self, specs_dir)`: `_SPEC_FILES` 辞書を用いて各スペックの `to_yaml()` メソッドで保存

### _SPEC_FILES 辞書

フィールド名と YAML ファイル名の正規対応表:

```python
_SPEC_FILES: dict[str, str] = {
    "input1": "input1.yaml",
    "related_work": "related_work_spec.yaml",
    "paper": "paper_spec.yaml",
    "paper_score": "paper_score_spec.yaml",
    "teacher_paper_set": "teacher_paper_set.yaml",
    "problem": "problem_spec.yaml",
    "model": "model_spec.yaml",
    "resource": "resource_spec.yaml",
    "plan": "plan_spec.yaml",
    "execution": "execution_spec.yaml",
}
```

### _SPEC_CLASSES 辞書

フィールド名とモデルクラスのマッピング。`load_from_dir` 内でモデルインスタンスの生成に使用される。

```python
_SPEC_CLASSES: dict[str, type] = {
    "input1": Input1Model,
    "related_work": RelatedWorkSpecModel,
    "paper": PaperSpecModel,
    "paper_score": PaperScoreSpecModel,
    "teacher_paper_set": TeacherPaperSetModel,
    "problem": ProblemSpecModel,
    "model": ModelSpecModel,
    "resource": ResourceSpecModel,
    "plan": PlanSpecModel,
    "execution": ExecutionSpecModel,
}
```
