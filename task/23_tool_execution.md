# SERA 要件定義書 — Tool Execution Engine

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 29. Tool Execution Engine（ツール実行基盤）

### 29.1 概要

§28（Agent Function System）で定義された `AgentFunction` のうち、`allowed_tools` を持つ10関数は AgentLoop 経由で実行される。本セクションでは、その実行基盤となるツール実装・AgentLoop メカニズム・安全制御を定義する。

> **Note**: ツールの利用可否は `plan_spec.agent_commands`（§5.8）で定義され、Phase 1 で凍結される。§5.8 の `tools.enabled` が各ツールの有効/無効を制御し、`phase_tool_map` が Phase ごとの利用可能ツールを決定する。

```text
■ §28 と §29 の役割分担

  §28 AgentFunction（タスク定義）
    ├─ 何を生成するか（return_schema, handler）
    ├─ どのツールを使えるか（allowed_tools）
    └─ ループ設定（loop_config: max_steps, budget）

  §29 Tool Execution Engine（実行基盤）
    ├─ ツールの実関数（ToolExecutor + 18 ハンドラ）
    ├─ ReAct ループ制御（AgentLoop）
    └─ 安全制御（ToolPolicy）
```

§28 の `call_function()` が `allowed_tools` の有無と `ToolConfig.enabled` に基づいて本セクションの AgentLoop を起動する。詳細は §28.6 を参照。

本セクションが提供する層:

| 層 | 提供物 |
|----|-------|
| ツール実装（executor） | `ToolExecutor` — 18 ツールの実関数ディスパッチ |
| エージェントループ | `AgentLoop` — Think → Act → Observe 反復 |
| 観測フィードバック | ツール結果をコンテキストに追加し LLM に返す |
| 停止判定 | 完了シグナル or 最大ステップ打ち切り |
| コスト・安全制御 | バジェット制御、サンドボックス、レート制限 |

### 29.2 ツールカテゴリ

SERA の全 Phase で必要なツールを4カテゴリに分類する。各 Phase で利用可能なツールは §5.8 の `phase_tool_map` で定義される。

#### 29.2.1 Web/API 検索ツール

Phase 0（先行研究収集）および Phase 7（引用検索）で使用。現在は `RelatedWorkEngine` と `CitationSearcher` がハードコードで API クライアントを呼んでいるが、LLM が動的にクエリを構成・結果を判断する能力がない。

| ツール名 | 説明 | 既存基盤 | Phase |
|---------|------|---------|-------|
| `semantic_scholar_search` | Semantic Scholar API で論文検索 | `api_clients/semantic_scholar.py` | 0, 7 |
| `semantic_scholar_references` | 論文の参照文献を取得 | `api_clients/semantic_scholar.py` | 0 |
| `semantic_scholar_citations` | 論文の被引用文献を取得 | `api_clients/semantic_scholar.py` | 0 |
| `crossref_search` | CrossRef API で論文検索 | `api_clients/crossref.py` | 0 |
| `arxiv_search` | arXiv API で論文検索 | `api_clients/arxiv_client.py` | 0 |
| `web_search` | 汎用Web検索（SerpAPI等） | `resource_spec.api_keys.serpapi` | 0, 7 |

**現状からの変化**: `RelatedWorkEngine._search_with_fallback()` は固定順序でクライアントをフォールバックするが、ツール化後は LLM が検索結果を見て「Semantic Scholar で不十分だから arXiv で補完しよう」と判断できる。

#### 29.2.2 コード実行ツール

Phase 3（実験実行）および Phase 7（図表生成）で使用。現在は `ExperimentGenerator` → `Executor.run()` の一方通行だが、ツール化後は LLM が実行結果を観測して即座にデバッグ・修正できる。

| ツール名 | 説明 | 既存基盤 | Phase |
|---------|------|---------|-------|
| `execute_experiment` | 実験スクリプトをサンドボックス内で実行 | `LocalExecutor` / `SlurmExecutor` / `DockerExecutor` | 3 |
| `execute_code_snippet` | 短いコード片を安全に実行（matplotlib等） | `figure_generator.py` の `exec()` | 7 |
| `run_shell_command` | サンドボックス内でシェルコマンドを実行 | なし（新規） | 3, 7 |

**サンドボックス制約**（`SandboxConfig` 準拠）:
- `experiment_timeout_sec`: タイムアウト（デフォルト 3600秒）
- `experiment_memory_limit_gb`: メモリ上限（デフォルト 16GB）
- `isolate_experiments`: 実験間の隔離（デフォルト True）
- `run_shell_command` は明示的なホワイトリストコマンドのみ許可

#### 29.2.3 ファイル I/O ツール

全 Phase で使用。LLM が実験結果・ログ・中間成果物を直接読み書きする。

| ツール名 | 説明 | 既存基盤 | Phase |
|---------|------|---------|-------|
| `read_file` | ワークスペース内のファイルを読む | なし | 全 |
| `write_file` | ワークスペース内にファイルを書く | なし | 3, 7 |
| `read_metrics` | `runs/<node_id>/metrics.json` を読む | `RunResult.metrics_path` | 4 |
| `read_experiment_log` | `runs/<node_id>/stdout.log` / `stderr.log` を読む | `RunResult.stdout_path` | 3 |
| `list_directory` | ディレクトリの内容をリストする | なし | 全 |

**セキュリティ制約**:
- パスは `sera_workspace/` 以下に限定（パストラバーサル防止）
- `.lock` ファイルへの書き込みは禁止（ExecutionSpec固定違反防止）
- `specs/` ディレクトリへの書き込みは Phase 1 完了後は禁止

#### 29.2.4 内部状態参照ツール

LLM がSearchTree の状態を参照して意思決定に活用する。

| ツール名 | 説明 | 既存基盤 | Phase |
|---------|------|---------|-------|
| `get_node_info` | 指定ノードの詳細（hypothesis, config, mu, se, lcb, status）を返す | `SearchNode.to_dict()` | 2 |
| `list_nodes` | ノード一覧をフィルタ付きで返す（status, top-k等） | `SearchManager.all_nodes` | 2 |
| `get_best_node` | 現在の最良ノードを返す | `SearchManager.best_node` | 2 |
| `get_search_stats` | 探索統計（total_nodes, evaluated, failed, best_lcb等） | なし（集計ロジック新規） | 2 |

### 29.3 ToolExecutor（ツール実行ディスパッチャ）

#### 29.3.1 データ構造

```python
# src/sera/agent/tool_executor.py

class ToolResult:
    """ツール実行の結果を表す構造体。"""
    tool_name: str            # 実行したツール名
    call_id: str              # 対応するToolCall.call_id
    success: bool             # 実行成功/失敗
    output: Any               # ツールの出力（dict, str, list 等）
    error: str | None         # エラーメッセージ（失敗時）
    wall_time_sec: float      # 実行にかかった時間
    truncated: bool           # 出力が切り詰められたか

class ToolExecutor:
    """ToolCallを受け取り、対応するツールを実行してToolResultを返す。"""

    def __init__(
        self,
        resource_spec: ResourceSpec,
        workspace_dir: Path,
        executor: Executor | None = None,         # 実験実行用
        scholar_clients: list[BaseScholarClient] | None = None,  # 検索API用
        search_manager: SearchManager | None = None,  # 内部状態参照用
    ): ...

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """ToolCallをディスパッチし、対応する実関数を実行する。"""
        ...

    def available_tools(self, phase: str | None = None) -> list[str]:
        """現在実行可能なツール名のリストを返す。"""
        ...
```

#### 29.3.2 ディスパッチロジック

```python
async def execute(self, tool_call: ToolCall) -> ToolResult:
    handler = self._handlers.get(tool_call.tool_name)
    if handler is None:
        return ToolResult(
            tool_name=tool_call.tool_name,
            call_id=tool_call.call_id,
            success=False,
            output=None,
            error=f"Unknown tool: {tool_call.tool_name}",
            wall_time_sec=0.0,
            truncated=False,
        )

    # ポリシーチェック
    if not self._check_policy(tool_call):
        return ToolResult(..., success=False, error="Policy violation")

    # バジェットチェック
    if not self._check_budget(tool_call):
        return ToolResult(..., success=False, error="Budget exhausted")

    start = time.monotonic()
    try:
        output = await handler(tool_call.arguments)
        wall_time = time.monotonic() - start
        # 出力の切り詰め（observation_max_tokens）
        output, truncated = self._truncate_output(output)
        self._log_execution(tool_call, output, wall_time, success=True)
        return ToolResult(
            tool_name=tool_call.tool_name,
            call_id=tool_call.call_id,
            success=True,
            output=output,
            error=None,
            wall_time_sec=wall_time,
            truncated=truncated,
        )
    except Exception as exc:
        wall_time = time.monotonic() - start
        self._log_execution(tool_call, None, wall_time, success=False, error=str(exc))
        return ToolResult(..., success=False, error=str(exc), wall_time_sec=wall_time)
```

#### 29.3.3 ツールハンドラの実装パターン

各ツールは `async def handler(arguments: dict) -> Any` の形式で実装する。既存の SERA コンポーネントをラップする:

```python
# Web検索ツールの例
async def _handle_semantic_scholar_search(self, args: dict) -> dict:
    query = args["query"]
    limit = args.get("limit", 10)
    year_from = args.get("year_from", None)
    results = await self._ss_client.search(query, limit=limit, year_from=year_from)
    return {
        "papers": [
            {"paper_id": r.paper_id, "title": r.title, "year": r.year,
             "authors": r.authors[:3], "citation_count": r.citation_count,
             "abstract": r.abstract[:300]}
            for r in results
        ],
        "total_results": len(results),
    }

# ファイル読み取りツールの例
async def _handle_read_file(self, args: dict) -> dict:
    path = self._resolve_safe_path(args["path"])  # パストラバーサル防止
    content = path.read_text(encoding="utf-8")
    content, truncated = self._truncate_text(content, max_chars=10000)
    return {"content": content, "truncated": truncated, "size_bytes": path.stat().st_size}

# 実験実行ツールの例
async def _handle_execute_experiment(self, args: dict) -> dict:
    node_id = args["node_id"]
    script_path = self._workspace / "runs" / node_id / "experiment.py"
    seed = args.get("seed", 42)
    result = await self._executor.run(node_id, script_path, seed, timeout=self._timeout)
    return {
        "success": result.success,
        "exit_code": result.exit_code,
        "wall_time_sec": result.wall_time_sec,
        "metrics": self._read_metrics_safe(result.metrics_path),
        "stderr_tail": self._read_tail(result.stderr_path, lines=20),
    }
```

### 29.4 AgentLoop（エージェント反復ループ）

#### 29.4.1 概要

LLM がツールを選択的に使いながらタスクを完了する ReAct 型のループ。

```text
AgentLoop のフロー:

  初期プロンプト（タスク説明 + 利用可能ツール一覧）
       │
       ▼
  ┌─── LLM 推論 ◄──────────────────────┐
  │    GenerationOutput(text, tool_calls) │
  │         │                             │
  │    tool_calls あり?                   │
  │    ├─ YES → ToolExecutor.execute()    │
  │    │         ToolResult               │
  │    │         │                        │
  │    │    observation をコンテキストに追加 ─┘
  │    │
  │    └─ NO (text のみ) → 完了判定
  │         ├─ 完了 → 最終結果を返す
  │         └─ 未完了 → 続行プロンプトで再推論 ─┐
  │                                            │
  └────────────────────────────────────────────┘

  打ち切り条件:
    - max_steps 到達
    - tool_call_budget 枯渇
    - LLM が明示的に完了を宣言
    - タイムアウト
```

#### 29.4.2 データ構造

```python
# src/sera/agent/agent_loop.py

@dataclass
class AgentTurn:
    """エージェントループの1ターンを記録する。"""
    step: int
    prompt: str                  # LLM への入力
    generation: GenerationOutput # LLM の出力
    tool_results: list[ToolResult]  # ツール実行結果（0個以上）
    wall_time_sec: float

@dataclass
class AgentLoopResult:
    """エージェントループ全体の結果。"""
    final_output: Any           # LLM が出した最終的な結果
    turns: list[AgentTurn]      # 全ターンの記録
    total_steps: int
    total_tool_calls: int
    total_wall_time_sec: float
    exit_reason: str            # "completed", "max_steps", "budget_exhausted", "timeout"

class AgentLoopConfig:
    """エージェントループの設定。"""
    max_steps: int = 10              # 最大ステップ数
    tool_call_budget: int = 20       # 1ループあたりのツール呼び出し上限
    observation_max_tokens: int = 2000  # 各ツール結果の最大トークン数
    timeout_sec: float = 300.0       # ループ全体のタイムアウト
    allowed_tools: list[str] | None = None  # 利用可能ツールの制限（Noneは全許可）

class AgentLoop:
    """ReAct型のエージェント反復ループ。"""

    def __init__(
        self,
        agent_llm: AgentLLM,
        tool_executor: ToolExecutor,
        config: AgentLoopConfig | None = None,
    ): ...

    async def run(
        self,
        task_prompt: str,
        purpose: str,
        available_tools: list[str] | None = None,
        adapter_node_id: str | None = None,
    ) -> AgentLoopResult:
        """エージェントループを実行し、結果を返す。"""
        ...
```

#### 29.4.3 ループ実装の詳細

```python
async def run(self, task_prompt, purpose, available_tools, adapter_node_id):
    turns = []
    context = task_prompt
    tool_call_count = 0
    start_time = time.monotonic()

    # ツールスキーマを取得
    tool_names = available_tools or self.tool_executor.available_tools()
    tool_schemas = REGISTRY.to_openai_tools(names=tool_names)

    for step in range(self.config.max_steps):
        # タイムアウトチェック
        if time.monotonic() - start_time > self.config.timeout_sec:
            return AgentLoopResult(..., exit_reason="timeout")

        # LLM 推論
        gen_out = await self.agent_llm.generate_with_tools(
            prompt=context,
            available_tools=tool_schemas,
            purpose=f"{purpose}_step{step}",
            adapter_node_id=adapter_node_id,
        )

        tool_results = []

        # ツール呼び出しがある場合、実行する
        if gen_out.tool_calls:
            for tc in gen_out.tool_calls:
                if tool_call_count >= self.config.tool_call_budget:
                    tool_results.append(ToolResult(
                        ..., success=False, error="Tool call budget exhausted"
                    ))
                    break
                result = await self.tool_executor.execute(tc)
                tool_results.append(result)
                tool_call_count += 1

            # Observation をコンテキストに追加
            observation = self._format_observations(tool_results)
            context = f"{context}\n\nAssistant: {gen_out.text or ''}\n\n{observation}\n\nContinue:"
        else:
            # ツール呼び出しなし → 完了と判断
            turns.append(AgentTurn(step=step, ...))
            return AgentLoopResult(
                final_output=gen_out.text,
                turns=turns,
                exit_reason="completed",
                ...
            )

        turns.append(AgentTurn(step=step, ...))

    return AgentLoopResult(..., exit_reason="max_steps")
```

#### 29.4.4 Observation フォーマット

ツール結果を LLM が理解できるテキストに変換する:

```text
Tool Results:
[1] semantic_scholar_search (call_id: abc123)
  Status: success
  Output:
    papers:
      - "Attention Is All You Need" (2017) by Vaswani et al. [cited: 120000]
      - "BERT: Pre-training of Deep Bidirectional Transformers" (2019) by Devlin et al. [cited: 85000]
    total_results: 10

[2] read_file (call_id: def456)
  Status: success
  Output:
    content: "import numpy as np\n..."
    truncated: true
    size_bytes: 15234
```

### 29.5 Phase 別統合

各 Phase での AgentLoop の使い方は §5.8 の `phase_tool_map` で定義される。`phase_tool_map` は Phase ごとに利用可能なツールのリストを持ち、AgentLoop 起動時に参照される。SearchManager のオーケストレーション（状態マシン）は変更しない。

呼び出し側のコードは `call_function()` を呼ぶだけで、AgentLoop の有無を意識しない:

```python
# tree_ops.py — AgentLoop 有効/無効を問わず同じコード
proposals = await self.agent_llm.call_function(
    "search_draft", prompt=prompt, temperature=temp
)
# call_function() 内部で:
#   ToolConfig.enabled=True  → AgentLoop 経由（allowed_tools で情報収集後に出力）
#   ToolConfig.enabled=False → 単発 generate()（既存動作）
```

**Phase 別の影響**:

| Phase | 対象関数 | ツール化による変化 |
|-------|---------|-----------------|
| 0 | `query_generation`, `paper_clustering` | LLM がクエリの有効性を検索 API で検証しながら生成 |
| 2 | `search_draft`, `search_improve` | LLM が過去ノードの結果を参照して仮説立案 |
| 3 | `search_debug`, `experiment_code_gen` | LLM がエラーログを読み、テストコードを実行して修正を検証 |
| 7 | `citation_*`, `aggregate_plot_*` | LLM が実際に論文検索 API を呼び、プロットコードをテスト実行 |

**注意**: `execute_experiment`（本番実験実行）は SearchManager が引き続きオーケストレーションする。LLM にはテストコード片の実行（`execute_code_snippet`）のみ許可する。

### 29.6 安全制御とポリシー

#### 29.6.1 ToolPolicy

```python
# src/sera/agent/tool_policy.py

@dataclass
class ToolPolicy:
    """ツール実行のポリシー定義。"""

    # Phase 別のツール許可リスト
    phase_allowed_tools: dict[str, list[str]] = field(default_factory=dict)

    # グローバル設定
    max_file_read_bytes: int = 1_000_000        # 1MB
    max_file_write_bytes: int = 500_000          # 500KB
    max_output_tokens: int = 2000                # ツール出力の切り詰め上限
    allowed_write_dirs: list[str] = field(       # 書き込み可能ディレクトリ
        default_factory=lambda: ["runs/", "paper/", "outputs/"]
    )
    blocked_write_patterns: list[str] = field(   # 書き込み禁止パターン
        default_factory=lambda: ["specs/*.yaml", "*.lock", "*.jsonl"]
    )
    allowed_shell_commands: list[str] = field(   # 許可シェルコマンド
        default_factory=lambda: ["pip", "python", "ls", "cat", "wc"]
    )

    # レート制限
    api_rate_limit_per_minute: int = 30          # 外部API呼び出し上限/分
    api_rate_limit_burst: int = 5                # バースト許可数

    # NetworkConfig との統合
    require_network_allowed: bool = True         # NetworkConfig.allow_internet をチェック
    require_api_allowed: bool = True             # NetworkConfig.allow_api_calls をチェック
```

#### 29.6.2 パストラバーサル防止

```python
def _resolve_safe_path(self, relative_path: str) -> Path:
    """ワークスペース内のパスに解決する。外部へのアクセスを防止。"""
    resolved = (self._workspace / relative_path).resolve()
    if not str(resolved).startswith(str(self._workspace.resolve())):
        raise PermissionError(f"Path traversal attempt: {relative_path}")
    return resolved
```

#### 29.6.3 ExecutionSpec 固定との整合

- ツールによる `specs/` ディレクトリへの書き込みは Phase 1 完了後は禁止
- `write_file` ツールは `blocked_write_patterns` をチェック
- ツールの引数に ExecutionSpec のパラメータ（`lr`, `clip_range`, `repeats` 等）を含めることはできない（付録 B-1 固定原則）

#### 29.6.4 vLLM sleep/wake との互換

ツール実行中は LLM 推論が不要なため、vLLM の sleep/wake 状態に影響しない。ただし AgentLoop 内の各ステップでは:

1. LLM 推論 → vLLM wake 状態が必要
2. ツール実行 → vLLM の状態は不問（CPUで実行）
3. 次の LLM 推論 → vLLM wake 状態が必要

PPO 更新時は従来通り vLLM を `sleep(level=2)` する。AgentLoop はPPO更新中に一時停止する。

### 29.7 ログ仕様

#### 29.7.1 ツール実行ログ

`sera_workspace/logs/tool_execution_log.jsonl` に全ツール実行を記録:

```json
{
    "event": "tool_execution",
    "timestamp": "2026-02-24T12:00:00Z",
    "call_id": "uuid",
    "tool_name": "semantic_scholar_search",
    "arguments": {"query": "attention mechanism", "limit": 10},
    "success": true,
    "output_size_bytes": 1234,
    "truncated": false,
    "wall_time_sec": 0.45,
    "error": null,
    "purpose": "phase0_related_work_step3",
    "node_id": "abc123"
}
```

#### 29.7.2 エージェントループログ

`sera_workspace/logs/agent_loop_log.jsonl` にループ全体を記録:

```json
{
    "event": "agent_loop_complete",
    "timestamp": "2026-02-24T12:05:00Z",
    "purpose": "phase0_related_work",
    "total_steps": 8,
    "total_tool_calls": 15,
    "total_wall_time_sec": 45.2,
    "exit_reason": "completed",
    "tools_used": {"semantic_scholar_search": 6, "arxiv_search": 3, "crossref_search": 2},
    "node_id": null
}
```

### 29.8 PlanSpec への設定追加

> **権威的な設定は PlanSpec §5.8 `agent_commands` に定義される。** `ToolConfig` クラスは実装レベルのモデルとして存続するが、Phase 1 凍結時に §5.8 の値から populate される。

**§5.8 PlanSpec → ToolConfig マッピング**:

| §5.8 PlanSpec field | ToolConfig field |
|---|---|
| `agent_commands.tools.enabled` | `enabled` |
| `agent_commands.loop_defaults.max_steps` | `max_steps_per_loop` |
| `agent_commands.loop_defaults.tool_call_budget` | `tool_call_budget_per_loop` |
| `agent_commands.loop_defaults.observation_max_tokens` | `observation_max_tokens` |
| `agent_commands.loop_defaults.timeout_sec` | `loop_timeout_sec` |
| `agent_commands.tools.api_rate_limit_per_minute` | `api_rate_limit_per_minute` |

```python
# src/sera/specs/plan_spec.py に追加

class ToolConfig(BaseModel):
    """ツール実行エンジンの設定（Phase C）。

    値は Phase 1 凍結時に PlanSpec §5.8 agent_commands から populate される。
    """
    enabled: bool = False                    # ツール実行の有効化
    max_steps_per_loop: int = 10             # エージェントループ最大ステップ
    tool_call_budget_per_loop: int = 20      # ループあたりツール呼び出し上限
    observation_max_tokens: int = 2000       # ツール出力の切り詰め上限
    loop_timeout_sec: float = 300.0          # ループタイムアウト
    api_rate_limit_per_minute: int = 30      # API レート制限
```

`ToolConfig.enabled = False` がデフォルトであり、有効化しない限り既存の動作に影響しない。

### 29.9 MCP（Model Context Protocol）対応

将来的には MCP サーバー経由で外部ツールを動的に追加できるようにする。

#### 29.9.1 MCP 統合の設計方針

```text
MCP統合の位置付け:

  ToolExecutor
    ├─ 内蔵ツール（§29.2 の全ツール）
    │   ├─ Web/API 検索ツール
    │   ├─ コード実行ツール
    │   ├─ ファイル I/O ツール
    │   └─ 内部状態参照ツール
    │
    └─ MCP ツール（外部 MCP サーバー経由）
        ├─ MCP サーバーからツール一覧を取得
        ├─ LLM に提示するスキーマに統合
        └─ ToolCall を MCP プロトコルで転送
```

#### 29.9.2 MCP クライアント

```python
# src/sera/agent/mcp_client.py（将来実装）

class MCPToolProvider:
    """MCP サーバーに接続し、ツール一覧と実行をブリッジする。"""

    def __init__(self, server_url: str, api_key: str | None = None): ...

    async def list_tools(self) -> list[dict]:
        """MCP サーバーからツール定義を取得し、AgentFunction 形式に変換。"""
        ...

    async def execute(self, tool_name: str, arguments: dict) -> Any:
        """MCP プロトコルでツールを実行。"""
        ...
```

#### 29.9.3 ResourceSpec への MCP 設定

```python
# resource_spec.py に追加（将来）

class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)

class MCPServerConfig(BaseModel):
    name: str
    url: str
    api_key_env: str = ""       # 環境変数名
    allowed_tools: list[str] = Field(default_factory=list)  # 空 = 全許可
    timeout_sec: float = 30.0
```

### 29.10 段階的実装ロードマップ

```text
Step 1: ToolExecutor + ToolResult（基盤） ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  新規:
    - src/sera/agent/tool_executor.py（ToolResult, ToolExecutor, ディスパッチ）
    - src/sera/agent/tool_policy.py（ToolPolicy, パス制約, レート制限）
    - src/sera/agent/tools/（各ツールハンドラの実装）
      - search_tools.py（semantic_scholar_search, crossref_search, arxiv_search, web_search）
      - execution_tools.py（execute_experiment, execute_code_snippet, run_shell_command）
      - file_tools.py（read_file, write_file, read_metrics, read_experiment_log, list_directory）
      - state_tools.py（get_node_info, list_nodes, get_best_node, get_search_stats）
  テスト:
    - tests/test_agent/test_tool_executor.py（ディスパッチ、ポリシーチェック、パス制約）
    - tests/test_agent/test_tool_policy.py（許可/拒否判定）
    - tests/test_agent/test_tools/（各ツールハンドラの単体テスト）

Step 2: AgentLoop（エージェント反復ループ） ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  新規:
    - src/sera/agent/agent_loop.py（AgentLoop, AgentTurn, AgentLoopResult, AgentLoopConfig）
  修正:
    - src/sera/agent/agent_llm.py（load_tools() の活性化）
    - src/sera/specs/plan_spec.py（ToolConfig 追加）
  テスト:
    - tests/test_agent/test_agent_loop.py（ループ制御、停止条件、バジェット管理）

Step 3: Phase 0 統合（先行研究収集のツール化） ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  修正:
    - src/sera/phase0/related_work_engine.py（AgentLoop 経由のオプション追加）
    - src/sera/commands/phase0_cmd.py（ToolConfig に応じた分岐）
  テスト:
    - tests/test_phase0/test_related_work_engine_with_tools.py

Step 4: Phase 2-3 統合（仮説生成・デバッグのツール化） ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  修正:
    - src/sera/search/tree_ops.py（AgentLoop 経由のオプション追加）
    - src/sera/search/search_manager.py（ツール有効時のフロー分岐）
  テスト:
    - tests/test_search/test_tree_ops_with_tools.py

Step 5: Phase 7 統合（論文生成のツール化） ✅ 実装済み
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  修正:
    - src/sera/paper/citation_searcher.py（AgentLoop 経由のオプション追加）
    - src/sera/paper/figure_generator.py（execute_code_snippet ツール経由に移行）
    - src/sera/paper/paper_composer.py（AgentLoop 統合）
  テスト:
    - tests/test_paper/test_citation_searcher_with_tools.py
    - tests/test_paper/test_figure_generator_with_tools.py

Step 6: MCP 対応 🔲 未着手
━━━━━━━━━━━━━━━━━━━━━━━━
  新規:
    - src/sera/agent/mcp_client.py（MCPToolProvider）
    - src/sera/specs/resource_spec.py（MCPConfig 追加）
  テスト:
    - tests/test_agent/test_mcp_client.py

Step 7: ツール使用経験の学習統合 🔲 未着手 → §26.5.3 参照
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  前提: Step 1–5 完了
  詳細は task/20_tool_using_agent.md §26.5.3 を参照
```

### 29.11 §26, §28 との関係

| セクション | 役割 | 本セクションとの関係 |
|-----------|------|-------------------|
| §26 (Tool-Using Agent) | 信用割当 + ツール使用経験からの学習（§26.5.3） | AgentLoop 軌跡が §26 の学習パイプラインに流入 |
| §28 (Agent Function System) | タスク定義（スキーマ + allowed_tools + loop_config） | §28.6 `call_function()` が §29 の AgentLoop を起動 |
| §29 (本セクション) | ツール実行基盤（ToolExecutor + AgentLoop + ToolPolicy） | — |

```text
§28 call_function()                  §29 AgentLoop
  │                                    │
  │  func.allowed_tools ──────────►    │  allowed_tools で利用ツールを制限
  │  func.loop_config ────────────►    │  AgentLoopConfig を構築
  │                                    │
  │                               AgentLoopResult
  │                                    │
  │  handler(final_output) ◄───────    │  最終出力を §28 のパーサで処理
  │                                    │
  │                                    ▼
  │                               §26.5.3 学習統合
  │                                    │  PPORolloutV3 にツール軌跡を記録
  │                                    │  tool_aware 報酬 / HiPER 拡張
```

### 29.14 ツール使用経験からの学習統合

> **本セクションの詳細は §26.5.3（task/20_tool_using_agent.md）に移動。** ここでは §29 側のインターフェースのみ定義する。

AgentLoop の実行軌跡を §26 の PPO/LoRA 学習パイプラインにフィードバックするため、以下のインターフェースを提供する:

#### 29.14.1 AgentLoopResult の学習用フィールド

§29.4.2 で定義した `AgentLoopResult` は以下の情報を含み、§26.5.3 の学習統合で使用される:

- `turns: list[AgentTurn]` — 各ステップの生成結果 + ツール実行結果
- `total_steps` / `total_tool_calls` — ループ統計
- `exit_reason` — 終了理由

#### 29.14.2 GenerationOutput のログ確率拡張

```python
@dataclass
class GenerationOutput:
    text: str | None
    tool_calls: list[ToolCall] | None
    purpose: str
    text_log_prob: float | None = None           # テキスト部分のログ確率（新規）
    tool_call_log_probs: list[float] | None = None  # 各ツール呼び出しのログ確率（新規）
```

| プロバイダ | テキスト log_prob | ツール選択 log_prob |
|-----------|------------------|-------------------|
| ローカル (vLLM) | ✅ トークン列から直接計算 | ✅ tool_call トークン列から分離計算 |
| OpenAI API | △ `logprobs` オプション | △ API 仕様に依存 |
| Anthropic API | ❌ 未提供（0.0） | ❌ 未提供（0.0） |

#### 29.14.3 SearchNode へのツール使用メタデータ

```python
# src/sera/search/search_node.py
tool_usage: dict = field(default_factory=dict)
# 例: {"total_tool_calls": 5, "tool_success_rate": 0.8, "tools_used": {...}, ...}
```

チェックポイントに永続化。`from_dict()` は unknown keys を無視するため後方互換。

データ構造の詳細（PPORolloutV3, ToolCallRecord, tool_aware 報酬, HiPER 拡張等）は §26.5.3 を参照。

### 29.15 新規ファイル一覧

| ファイル | Step | 役割 | 状態 |
|---------|------|------|------|
| `src/sera/agent/tool_executor.py` | 1 | ToolResult, ToolExecutor, ディスパッチ | ✅ 実装済み |
| `src/sera/agent/tool_policy.py` | 1 | ToolPolicy, パス制約, レート制限 | ✅ 実装済み |
| `src/sera/agent/tools/__init__.py` | 1 | ツールハンドラパッケージ | ✅ 実装済み |
| `src/sera/agent/tools/search_tools.py` | 1 | Web/API 検索ツール（6個） | ✅ 実装済み |
| `src/sera/agent/tools/execution_tools.py` | 1 | コード実行ツール（3個） | ✅ 実装済み |
| `src/sera/agent/tools/file_tools.py` | 1 | ファイル I/O ツール（5個） | ✅ 実装済み |
| `src/sera/agent/tools/state_tools.py` | 1 | 内部状態参照ツール（4個） | ✅ 実装済み |
| `src/sera/agent/agent_loop.py` | 2 | AgentLoop, AgentTurn, AgentLoopResult | ✅ 実装済み |
| `src/sera/agent/mcp_client.py` | 6 | MCPToolProvider（将来） | 🔲 未着手 |

### 29.16 修正ファイル一覧

| ファイル | Step | 変更内容 | 状態 |
|---------|------|---------|------|
| `src/sera/specs/plan_spec.py` | 2 | ToolConfig 追加 | ✅ 実装済み |
| `src/sera/agent/agent_llm.py` | 2 | load_tools() 活性化、GenerationOutput ログ確率フィールド追加 | ✅ 実装済み |
| `src/sera/phase0/related_work_engine.py` | 3 | AgentLoop 経由オプション | ✅ 実装済み |
| `src/sera/commands/phase0_cmd.py` | 3 | ToolConfig 分岐 | ✅ 実装済み |
| `src/sera/search/tree_ops.py` | 4 | AgentLoop 経由オプション | ✅ 実装済み |
| `src/sera/search/search_manager.py` | 4 | ツール有効時フロー分岐 | ✅ 実装済み |
| `src/sera/search/search_node.py` | 4 | tool_usage フィールド追加 | ✅ 実装済み |
| `src/sera/paper/citation_searcher.py` | 5 | AgentLoop 経由オプション | ✅ 実装済み |
| `src/sera/paper/figure_generator.py` | 5 | execute_code_snippet 経由 | ✅ 実装済み |
| `src/sera/paper/paper_composer.py` | 5 | AgentLoop 統合 | ✅ 実装済み |
| `src/sera/specs/resource_spec.py` | 6 | MCPConfig 追加（将来） | 🔲 未着手 |

> Step 7（学習統合）の修正ファイルは §26.5.3（task/20_tool_using_agent.md）を参照。

---

（§29 Tool Execution Engine — TASK.md v13.0）
