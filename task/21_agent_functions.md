# SERA 要件定義書 — Agent Function System

> 本ファイルは TASK.md v13.1 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 27. Agent Function System

### 27.1 概要

> **実装状況**: ✅ 全て実装済み。AgentFunctionRegistry、全19関数定義・ハンドラ、`call_function()` メソッド（AgentLoop分岐含む）、呼び出し側マイグレーション（tree_ops, experiment_generator, spec_builder）が完了。

SERAのAgentLLMは19箇所のLLM呼び出しで `generate(prompt, purpose) -> str` を使い、各呼び出し側が独自にJSON/コード抽出・パースを行っている。

本セクションでは、全LLM呼び出しを統一的な Agent Function 定義で管理するシステムを構築する。各関数は**出力スキーマ**（何を生成するか）と**ツールアクセス**（どのツールを使えるか）の両方を定義し、`call_function()` が単発生成と AgentLoop（§28）を自動判定する統一エントリポイントとなる。

> **§5.8 との関係**: 各関数がどのツールを使えるか（`function_tool_bindings`）、ループパラメータ（`function_loop_overrides`）、有効化状態（`tools.enabled`）は **PlanSpec §5.8 `agent_commands`** で Phase 1 に確定・凍結される。本セクションのコード内定義はデフォルト値であり、実行時には §5.8 の値で上書きされる。

```text
■ §27 の位置付け

  呼び出し側（tree_ops, experiment_generator 等）
       │
       ▼
  call_function("search_draft", prompt)   ← 呼び出し側は分岐を意識しない
       │
       ├─ allowed_tools == None → 単発 generate()
       │    → handler(parse) → return
       │
       └─ allowed_tools あり かつ ToolConfig.enabled
            → §28 AgentLoop（ReAct ループ）
            → ツール使用 → 観測 → ... → 最終出力
            → handler(parse) → return

  ※ handler / return_schema は両パスで共通（最終出力のパース・検証）
```

### 27.2 設計方針

1. **スキーマ駆動**: 各LLM呼び出しの入出力をJSON Schemaで定義
2. **レジストリパターン**: `reward.py` の `_REWARD_METHODS` と同様の登録パターン
3. **段階的移行**: 既存の `generate()` パスを維持しつつ `call_function()` を追加
4. **プロバイダ透過**: OpenAI/Anthropic のネイティブtool-calling とローカルのプロンプトベースを自動切り替え
5. **統一エントリポイント**: `call_function()` が関数定義の `allowed_tools` を参照し、単発生成と AgentLoop を自動分岐。呼び出し側のコード変更は不要

### 27.3 コアデータ構造

```python
class OutputMode(Enum):
    JSON = "json"           # 構造化 JSON（return_schema で検証）
    CODE = "code"           # コードブロック抽出
    FREE_TEXT = "free_text" # 自由テキスト

@dataclass(frozen=True)
class AgentFunction:
    name: str                           # snake_case 一意識別子
    description: str                    # LLM に表示される説明
    parameters: dict[str, Any]          # JSON Schema (OpenAI function calling 形式)
    return_schema: dict[str, Any] | None  # 戻り値の JSON Schema
    output_mode: OutputMode             # 出力モード
    phase: str                          # "search", "execution", "spec", "paper", "evaluation", "phase0"
    default_temperature: float          # デフォルト温度
    max_retries: int                    # 最大リトライ回数
    handler: Callable | None            # レスポンスパーサ

    # §28 AgentLoop 統合（ToolConfig.enabled=True 時のみ有効）
    allowed_tools: list[str] | None = None     # None → 単発生成, リスト → AgentLoop
    loop_config: dict[str, Any] | None = None  # AgentLoop 設定 (max_steps, budget 等)
```

**`allowed_tools` と `loop_config` の意味**:

- `allowed_tools = None`: 従来通り単発 `generate()` で処理。`loop_config` は無視される
- `allowed_tools = ["tool_a", "tool_b"]` かつ `ToolConfig.enabled=True` かつ `ToolExecutor` が利用可能: §28 の `AgentLoop` 経由で実行。LLM は指定されたツールのみ使用可能
- `allowed_tools` ありだが `ToolConfig.enabled=False` の場合: 単発にフォールバック。既存動作に影響なし

> **§5.8 連携**: `allowed_tools` のデフォルト値は各関数定義に記載されるが、実行時には PlanSpec §5.8 `agent_commands.functions.function_tool_bindings` の値が優先される。`loop_config` も同様に §5.8 `function_loop_overrides` で上書きされる。これにより Phase 1 でツールアクセスを一元管理できる。

`loop_config` は `AgentLoopConfig`（§28.4.2）のフィールドを辞書で指定:

```python
loop_config = {
    "max_steps": 5,
    "tool_call_budget": 10,
    "observation_max_tokens": 2000,
    "timeout_sec": 120.0,
}
```

### 27.4 レジストリAPI

```python
class AgentFunctionRegistry:
    def register(func: AgentFunction) -> None       # 登録
    def get(name: str) -> AgentFunction              # 取得
    def list_all() -> list[AgentFunction]             # 全件取得
    def list_by_phase(phase: str) -> list[AgentFunction]  # Phase別取得
    def list_by_mode(mode: str) -> list[AgentFunction]    # 呼び出しパターン別取得
    def to_openai_tools(names: list[str] | None) -> list[dict]    # OpenAI形式変換
    def to_anthropic_tools(names: list[str] | None) -> list[dict] # Anthropic形式変換
    def to_prompt_schema(names: list[str] | None) -> str           # プロンプト埋め込み用テキスト

REGISTRY = AgentFunctionRegistry()  # シングルトン
```

### 27.5 登録関数一覧（全19関数）

> **§5.8 参照**: 各関数の有効化リストは PlanSpec §5.8 `agent_commands.functions.available_functions` で定義される。ツールバインディングは §5.8 `function_tool_bindings`、ループ設定は §5.8 `function_loop_overrides` で Phase 1 に凍結される。以下はコード内のデフォルト定義。

#### 27.5.1 AGENT_LOOP 関数（10関数）

`ToolConfig.enabled=True`（デフォルト）時に AgentLoop で実行される関数。ツールを使った情報収集・検証が品質向上に寄与する。`ToolConfig.enabled=False` 時は単発にフォールバック。

| 関数名 | Phase | 出力 | allowed_tools | 既存マッピング |
|--------|-------|------|---------------|--------------|
| `search_draft` | search | JSON | `get_node_info`, `list_nodes`, `read_metrics` | `TreeOps._generate_proposals()` (draft) |
| `search_debug` | search | JSON | `read_experiment_log`, `read_file`, `execute_code_snippet` | `TreeOps.debug()` |
| `search_improve` | search | JSON | `get_best_node`, `read_metrics`, `get_search_stats` | `TreeOps._generate_proposals()` (improve) |
| `experiment_code_gen` | execution | CODE | `read_file`, `execute_code_snippet` | `ExperimentGenerator._generate_code()` |
| `query_generation` | phase0 | FREE_TEXT | `semantic_scholar_search`, `arxiv_search` | `RelatedWorkEngine._build_queries()` |
| `citation_identify` | paper | FREE_TEXT | `semantic_scholar_search`, `web_search` | `CitationSearcher.search_loop()` step1 |
| `citation_select` | paper | FREE_TEXT | `semantic_scholar_search` | `CitationSearcher.search_loop()` step3 |
| `aggregate_plot_generation` | paper | JSON | `execute_code_snippet` | `FigureGenerator.aggregate_plots()` |
| `aggregate_plot_fix` | paper | CODE | `execute_code_snippet` | `FigureGenerator.aggregate_plots()` (fix) |
| `paper_clustering` | phase0 | JSON | `semantic_scholar_search` | `clustering.cluster_papers()` |

**loop_config 一覧**:

| 関数名 | max_steps | tool_call_budget | timeout_sec | 根拠 |
|--------|-----------|-----------------|-------------|------|
| `search_draft` | 5 | 10 | 120 | 過去ノード参照→仮説立案 |
| `search_debug` | 5 | 10 | 120 | エラーログ読み→修正テスト |
| `search_improve` | 5 | 10 | 120 | ベストノード参照→改善案 |
| `experiment_code_gen` | 8 | 15 | 180 | コード生成→テスト実行→修正の反復 |
| `query_generation` | 5 | 10 | 120 | クエリ生成→検索テスト→調整 |
| `citation_identify` | 8 | 15 | 180 | 論文検索を複数ラウンド |
| `citation_select` | 5 | 10 | 120 | 候補比較→選択 |
| `aggregate_plot_generation` | 5 | 10 | 120 | コード生成→テスト実行 |
| `aggregate_plot_fix` | 5 | 10 | 120 | 修正→テスト |
| `paper_clustering` | 3 | 5 | 60 | 検索で分類を補完 |

#### 27.5.2 SINGLE_SHOT 関数（9関数）

常に単発 `generate()` で処理される。入力として十分なコンテキストが渡されるため、ツールによる追加情報収集は不要。

| 関数名 | Phase | 出力モード | 既存マッピング |
|--------|-------|----------|--------------|
| `spec_generation_problem` | spec | JSON | `SpecBuilder.build_problem_spec()` |
| `spec_generation_plan` | spec | JSON | `SpecBuilder.build_plan_spec()` |
| `paper_outline` | paper | FREE_TEXT | `PaperComposer._step5` (outline) |
| `paper_draft` | paper | FREE_TEXT | `PaperComposer._step5` (draft) |
| `paper_reflection` | paper | FREE_TEXT | `PaperComposer._step5` (reflection) |
| `citation_bibtex` | paper | FREE_TEXT | `CitationSearcher.search_loop()` step4 |
| `paper_review` | evaluation | FREE_TEXT | `PaperEvaluator._generate_review()` |
| `paper_review_reflection` | evaluation | FREE_TEXT | `PaperEvaluator` (reflection loop) |
| `meta_review` | evaluation | FREE_TEXT | `PaperEvaluator._generate_meta_review()` |

### 27.6 `AgentLLM.call_function()` メソッド

```python
async def call_function(
    self,
    function_name: str,
    prompt: str,
    purpose: str | None = None,
    adapter_node_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
```

処理フロー:

```text
1. REGISTRY.get(function_name) で AgentFunction を取得
2. purpose 未指定なら function_name を使用
3. temperature 未指定なら func.default_temperature を使用

4. 呼び出しパターン判定:
   ┌─────────────────────────────────────────────────────┐
   │ func.allowed_tools is not None                      │
   │ AND self._tool_executor is not None                 │
   │ AND self._tool_config.enabled == True               │
   ├─────────────────────────────────────────────────────┤
   │ → AgentLoop 経由                                    │
   │   a. AgentLoopConfig を func.loop_config から構築   │
   │   b. agent_loop.run(                                │
   │        task_prompt=prompt,                           │
   │        purpose=function_name,                        │
   │        allowed_tools=func.allowed_tools,             │
   │        config=loop_config,                           │
   │        adapter_node_id=adapter_node_id,              │
   │      )                                              │
   │   c. raw = agent_loop_result.final_output           │
   │   d. self._last_loop_result = agent_loop_result     │
   │      (§25.5.3 学習統合用に保持)                     │
   └─────────────────────────────────────────────────────┘
   ┌─────────────────────────────────────────────────────┐
   │ それ以外 → 単発生成（既存パス）                     │
   ├─────────────────────────────────────────────────────┤
   │ ├─ output_mode==JSON かつ API プロバイダ            │
   │ │  → generate_with_tools() でネイティブtool-calling │
   │ └─ それ以外                                        │
   │    → JSON なら return_schema をプロンプト末尾に追記 │
   │    → generate() で生成                              │
   └─────────────────────────────────────────────────────┘

5. func.handler(raw) でパース
6. return_schema があれば validate_against_schema() で検証
7. リトライ: パース/検証失敗時、max_retries 回まで温度を +0.1 して再試行
```

**重要**: `ToolConfig.enabled=False` の場合、全19関数が単発生成にフォールバックする（デフォルトは `True` — 全関数が AgentLoop 経由で実行される）。呼び出し側（`tree_ops.py` 等）は `call_function()` を呼ぶだけで、AgentLoop の有無を意識しない。

### 27.7 マイグレーション状況

> **実装状況**: ✅ 全て完了。

初回マイグレーション（✅ 完了）:

1. `src/sera/search/tree_ops.py` — `search_draft`, `search_debug`, `search_improve`
2. `src/sera/execution/experiment_generator.py` — `experiment_code_gen`
3. `src/sera/phase1/spec_builder.py` — `spec_generation_problem`, `spec_generation_plan`

残り13関数のマイグレーション（✅ 完了）:

4. `src/sera/paper/paper_composer.py` — `paper_outline`, `paper_draft`, `paper_reflection`
5. `src/sera/paper/citation_searcher.py` — `citation_identify`, `citation_select`, `citation_bibtex`
6. `src/sera/paper/figure_generator.py` — `aggregate_plot_generation`, `aggregate_plot_fix`
7. `src/sera/paper/paper_evaluator.py` — `paper_review`, `paper_review_reflection`, `meta_review`
8. `src/sera/phase0/related_work_engine.py` — `query_generation`, `paper_clustering`

### 27.8 §25, §28 との関係

```text
全体アーキテクチャ:

  §5.8 PlanSpec agent_commands（Phase 1 凍結）
  ┌──────────────────────────────────────────┐
  │ available_tools / available_functions     │
  │ phase_tool_map / function_tool_bindings   │
  │ loop_defaults / function_loop_overrides   │
  └───────────┬────────────┬─────────────────┘
              │            │
              ▼            ▼
  §27 AgentFunction              §28 Tool Execution Engine
  ┌─────────────────────┐       ┌──────────────────────────┐
  │ name                │       │ ToolExecutor             │
  │ return_schema       │       │   ├─ 18個のツールハンドラ │
  │ handler             │       │   └─ ToolPolicy          │
  │ allowed_tools ──────┼──────►│ AgentLoop                │
  │ loop_config ────────┼──────►│   └─ ReAct 反復ループ    │
  └────────┬────────────┘       └──────────┬───────────────┘
           │                               │
  call_function()                  AgentLoopResult
           │                       (turns, tool_calls)
           ▼                               │
  最終出力パース                           ▼
  (handler + validate)          §25 学習統合
                                ┌──────────────────────────┐
                                │ PPORolloutV3             │
                                │ ToolCallRecord           │
                                │ tool_aware 報酬          │
                                │ HiPER ツール品質         │
                                └──────────────────────────┘
```

| セクション | 役割 | 本セクションとの関係 |
|-----------|------|-------------------|
| §5.8 (agent_commands) | ツール・関数の有効化リスト + Phase別マッピング（Phase 1 凍結） | `allowed_tools`/`loop_config` のランタイム値を §5.8 から取得 |
| §25 (Tool-Using Agent) | 信用割当 + ツール使用経験からの学習 | AgentLoop の軌跡が §25 の学習パイプラインに流入 |
| §27 (本セクション) | 全 LLM 呼び出しのタスク定義 + 統一エントリポイント | — |
| §28 (Tool Execution) | ツール実装 + AgentLoop メカニズム + 安全制御 | §27 の `allowed_tools` / `loop_config` に基づき §28 の AgentLoop が起動 |

### 27.9 新規ファイル一覧

| ファイル | 役割 | 状態 |
|---------|------|------|
| `src/sera/agent/agent_functions.py` | コアレジストリ + パースユーティリティ | ✅ 実装済み |
| `src/sera/agent/functions/__init__.py` | 全関数モジュールのインポート | ✅ 実装済み |
| `src/sera/agent/functions/search_functions.py` | search_draft, search_debug, search_improve | ✅ 実装済み |
| `src/sera/agent/functions/execution_functions.py` | experiment_code_gen | ✅ 実装済み |
| `src/sera/agent/functions/spec_functions.py` | spec_generation_problem, spec_generation_plan | ✅ 実装済み |
| `src/sera/agent/functions/paper_functions.py` | paper系8関数 | ✅ 実装済み |
| `src/sera/agent/functions/evaluation_functions.py` | paper_review, paper_review_reflection, meta_review | ✅ 実装済み |
| `src/sera/agent/functions/phase0_functions.py` | query_generation, paper_clustering | ✅ 実装済み |
| `tests/test_agent/test_agent_functions.py` | レジストリ・パース・ハンドラのテスト | ✅ 実装済み |

### 27.10 修正ファイル一覧

| ファイル | 変更内容 | 状態 |
|---------|---------|------|
| `src/sera/agent/agent_llm.py` | `call_function()` メソッド追加（AgentLoop 分岐ロジック含む） | ✅ 実装済み |
| `src/sera/search/tree_ops.py` | `call_function()` 経由に移行 | ✅ 実装済み |
| `src/sera/execution/experiment_generator.py` | `call_function()` 経由に移行 | ✅ 実装済み |
| `src/sera/phase1/spec_builder.py` | `call_function()` 経由に移行 | ✅ 実装済み |

---

（§27 Agent Function System — TASK.md v13.1）
