# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SERA (Self-Evolving Research Agent) is a Python autonomous research system that conducts end-to-end scientific research through tree-based solution exploration and parameter-efficient model adaptation (LoRA + PPO). It operates through an 8-phase pipeline from related work collection to paper generation.

## Build & Development Commands

```bash
# Install in development mode (GPU available locally)
pip install -e ".[dev]"

# Optional extras: slurm, docker, vllm (local LLM inference), latex (paper generation)
pip install -e ".[dev,slurm]"

# SLURM cluster: install on GPU node to get CUDA-enabled PyTorch
srun --partition=<gpu-partition> --time=01:00:00 bash scripts/setup_env.sh
source .venv/bin/activate

# Run all tests
pytest tests/

# Run a specific test file or directory
pytest tests/test_search/ -v
pytest tests/test_cli/test_init.py -v

# Run tests by marker
pytest -m "not gpu and not slow and not network" tests/   # skip heavy tests
pytest -m gpu tests/                                       # GPU-only tests

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Run research on SLURM GPU node
sbatch scripts/run_research.sh
```

**GPU setup caveat**: Running `pip install` on a login node (no GPU) causes `torch.cuda.is_available() = False`. Always install on a GPU node via `scripts/setup_env.sh`. See `docs/setup_gpu.md` for details.

## Code Style

- Python 3.11+, line length 120 (Ruff)
- Async tests use `asyncio_mode = "auto"` (no manual `@pytest.mark.asyncio` needed)
- Pydantic v2 models for all specs/configs
- structlog for logging (JSONL format)

## Architecture

### Dual-Tree Design

The system manages two synchronized tree structures:

1. **External Search Tree** — Best-first search over hypotheses/experiments. Nodes represent candidate solutions; selection uses LCB (Lower Confidence Bound) scoring. Operators: Draft, Debug, Improve.
2. **Internal LoRA Lineage Tree** — Tracks evolution of LoRA adapters. Uses **delta inheritance**: child nodes store only the diff from parent. Full weights are reconstructed by summing deltas along root-to-node path. Deep nodes get "squashed" snapshots to bound reconstruction cost.

These are **separate trees with different node IDs**. `SearchNode.node_id` (uuid) is distinct from `SearchNode.adapter_node_id` (reference into lineage tree). Not every search node triggers a PPO update, so not every search node has an `adapter_node_id`.

### Eight-Phase Pipeline

```
Phase 0: Related Work Collection   → Web/API search (Semantic Scholar, CrossRef, arXiv)
Phase 1: Spec Freezing             → Generate & freeze all specs (immutable after this)
Phase 2: Best-First Tree Search    → Hypothesis exploration via SearchManager
Phase 3: Experiment Execution      → LLM-generated code run in sandbox (local/SLURM/Docker)
Phase 4: Statistical Evaluation    → Repeats + SE + LCB computation
Phase 5: PPO Learning              → LoRA-only updates via PPO
Phase 6: Lineage Management        → Delta squash + pruning
Phase 7: Paper Generation          → LaTeX composition with citation search + VLM review
Phase 8: Paper Evaluation          → Ensemble review + meta-review + revision loop
```

PPO/lineage (Phases 5-6) are **optional**. In `research_cmd.py`, `ppo_trainer`, `lineage_manager`, and `pruner` are only created if `learning.enabled=True` AND `agent_llm.provider="local"`. The search loop (Phases 2-4) runs correctly without them.

### Variable Mutability (Three-Layer Separation)

This is a core design constraint enforced throughout the codebase:

| Layer | Location | Mutable after Phase 1? | Examples |
|-------|----------|----------------------|----------|
| **Frozen** | ExecutionSpec | No | lr, clip_range, repeats, lcb_coef, max_nodes, rank, alpha |
| **Manipulated** | ProblemSpec.manipulated_variables | Yes (per-node branching) | experiment learning_rate, batch_size, method |
| **Derived** | Computed at runtime | Auto-computed only | priority, mu, se, lcb, feasible, reward |

Frozen-layer and manipulated-layer variables can have similar names (e.g., `ExecutionSpec.learning.lr` vs `ProblemSpec.manipulated_variables[].learning_rate`) — they are in different layers.

### Spec System

**`AllSpecs`** (`specs/__init__.py`) is a plain dataclass (not Pydantic) aggregating 10 Pydantic spec models. It provides `load_from_dir()` / `save_to_dir()` and is passed everywhere as `specs`. Each spec model has `from_yaml()` / `to_yaml()` methods. File mapping is in `_SPEC_FILES` dict (e.g., `"execution"` → `"execution_spec.yaml"`).

**Spec locking**: Only `ExecutionSpecModel` gets locked (SHA-256 hash in `.lock` file via `compute_hash()` using `json.dumps(model_dump(), sort_keys=True)`). `research_cmd.py` verifies the lock before loading.

**Backward-compatible schema evolution**: `PruningConfig`, `TerminationConfig`, and `PaperExecConfig` use `@model_validator(mode="before")` to handle field renames (e.g., `keep_top_k` → `keep_topk`). Use this pattern when renaming spec fields.

**API keys**: `ResourceSpecModel.api_keys` stores **environment variable names**, not actual keys. `AgentLLM` reads the env var name then calls `os.environ.get(key_name)`.

### Tool-Using Agent Architecture

The agent layer is a multi-layer system for structured LLM calls with optional tool use:

```
AgentFunction (schema/registry)
  └── AgentFunctionRegistry (REGISTRY singleton)
        └── AgentLLM.call_function() (invoke by name)
              └── AgentLoop (ReAct loop: Think → Act → Observe)
                    └── ToolExecutor (dispatches 18 tools)
                          └── ToolPolicy (safety/rate-limiting)
```

**AgentFunction / REGISTRY** (`agent/agent_functions.py`): Every structured LLM call is defined as a frozen `AgentFunction` dataclass with `name`, `parameters` (JSON Schema), `return_schema`, `output_mode` (`JSON`/`CODE`/`FREE_TEXT`), `phase`, `default_temperature`, `max_retries`, and a `handler` callable. Functions are registered via `@register_function` decorator into the module-level `REGISTRY` singleton. Importing `sera.agent.functions` triggers all sub-module registrations (side-effect imports). REGISTRY can convert to OpenAI/Anthropic tool formats or inject schemas into prompts.

**19 registered functions** across 6 sub-modules in `agent/functions/`: `phase0_functions` (2), `search_functions` (3), `execution_functions` (1), `spec_functions` (2), `paper_functions` (8), `evaluation_functions` (3).

**`AgentLLM.call_function(function_name, prompt, ...)`** is the primary entry point for structured LLM calls. It looks up the function in REGISTRY, uses native tool-calling for API providers (OpenAI/Anthropic) when `return_schema` is defined, injects schema into prompt text for local providers, applies the function's `handler`, validates output, and retries up to `max_retries` (temperature += 0.1 per retry). This supersedes raw `generate()` for all schema-validated calls.

**`AgentLoop`** (`agent/agent_loop.py`): ReAct (Reason + Act) loop. Calls `agent_llm.generate_with_tools()`, dispatches tool calls via `ToolExecutor`, formats observations, appends to context, repeats. Exit reasons: `"completed"` (no tool calls), `"max_steps"`, `"budget_exhausted"`, `"timeout"`. Config via `AgentLoopConfig`: `max_steps=10`, `tool_call_budget=20`, `timeout_sec=300.0`. Functions not listed in `PlanSpec.agent_commands.functions.function_tool_bindings` are `SINGLE_SHOT` (no loop, single `call_function()`).

**18 tools** in `agent/tools/` across 4 categories: SEARCH (6: semantic_scholar_search/references/citations, crossref_search, arxiv_search, web_search), EXECUTION (3: execute_experiment, execute_code_snippet, run_shell_command), FILE (5: read/write_file, read_metrics, read_experiment_log, list_directory), STATE (4: get_node_info, list_nodes, get_best_node, get_search_stats).

**`ToolPolicy`** (`agent/tool_policy.py`): Per-phase tool allow-lists (`phase_allowed_tools`), write directory whitelist (`runs/`, `paper/`, `outputs/`), blocked write patterns (`specs/*.yaml`, `*.lock`, `*.jsonl`), shell command whitelist (`pip`, `python`, `ls`, `cat`, `wc`), API rate limiting (30 calls/min sliding window), path traversal protection.

**`PlanSpec.agent_commands`**: Configures the agent layer per-research-run. Contains `tools` (enabled tools, phase_tool_map, rate limits), `functions` (available_functions, function_tool_bindings), `loop_defaults`, and `function_loop_overrides` (per-function max_steps/budget/timeout). Frozen at Phase 1.

**Prompt formatters**: `AgentLLM` applies model-family-aware formatting for local providers: `chatml` (Qwen2), `llama3`, `deepseek`, `default`. API providers passthrough.

### Key Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `cli.py` | Typer CLI entry point | `app` |
| `agent/` | LLM interface + tool-using agent layer | `AgentLLM`, `AgentLoop`, `ToolExecutor`, `ToolPolicy`, `AgentFunctionRegistry` |
| `search/` | Best-first tree search | `SearchManager`, `SearchNode`, `TreeOps` |
| `execution/` | Experiment runners (`SlurmExecutor` auto-maps `ComputeConfig` → submitit params; `sbatch_extra` overrides) | `Executor` (ABC), `LocalExecutor`, `SlurmExecutor`, `DockerExecutor` |
| `evaluation/` | Statistical evaluation | `Evaluator`, `StatisticalEvaluator`, `FeasibilityChecker` |
| `learning/` | PPO training (LoRA-only) | `PPOTrainer`, `PPORollout`, `RewardComputer` |
| `lineage/` | LoRA delta management | `LineageManager`, `LRUCache`, `Pruner` |
| `paper/` | Paper generation | `PaperComposer`, `FigureGenerator`, `CitationSearcher`, `VLMReviewer` |
| `phase0/` | Related work collection | `RelatedWorkEngine`, API clients |
| `phase1/` | Spec building & freezing | `SpecBuilder`, `SpecFreezer` |
| `specs/` | Pydantic spec models | `AllSpecs`, `ExecutionSpecModel`, `ProblemSpecModel`, etc. |
| `commands/` | CLI command handlers | `init_cmd`, `phase0_cmd`, `research_cmd`, `paper_cmd`, `setup_cmd` |

### Key Interfaces

**`RunResult`** dataclass is the contract between Executor and Evaluator:
- `success`, `exit_code`, `stdout_path`, `stderr_path`, `metrics_path`, `wall_time_sec`, `seed`
- `metrics_path` points to `runs/<node_id>/metrics.json` — the only output channel from experiment scripts back to the evaluator
- Exit codes: `0`=ok, `-9`=timeout, `-7`=OOM (SERA-specific sentinel), `137`=Linux OOM killer

**LLM → JSON contract**: Every prompt ends with "Output ONLY the JSON, no other text." Parsing uses 3-stage fallback: (1) extract from ```json fence, (2) raw JSON parse, (3) regex for `[...]`/`{...}`. Canonical implementations are in `agent_functions.py` (`parse_json_response`, `extract_code_block`, `validate_against_schema`). `tree_ops.py` has its own copy used at runtime for search. After 3 retries (temperature += 0.1 each), a hardcoded fallback node is returned.

**Dual prompt systems**: `prompt_templates.py` defines 21 templates in `TEMPLATE_REGISTRY` (richly structured, for the full pipeline). `tree_ops.py` defines its own `DRAFT_PROMPT`/`DEBUG_PROMPT`/`IMPROVE_PROMPT` that are actually used at runtime. When modifying search prompts, edit the ones in `tree_ops.py`.

### Storage

Persistence is flat files (no SQLite):

```
sera_workspace/
  specs/                          # 10 YAML files + .lock
  logs/
    agent_llm_log.jsonl           # Every LLM call
    agent_loop_log.jsonl          # Every AgentLoop completion
    tool_execution_log.jsonl      # Every tool call execution
    search_log.jsonl              # Every node processed
    ppo_log.jsonl                 # Every PPO update
  checkpoints/
    search_state_step_N.json      # Full SearchManager state
  lineage/nodes/<adapter_id>/
    meta.json                     # parent_id, depth, tensor info, is_snapshot
    adapter_delta.safetensors     # Per-layer weight deltas
    adapter_snapshot.safetensors  # Full weights at squash depth (optional)
  runs/<node_id>/
    experiment.py                 # Generated experiment code
    stdout.log / stderr.log
    metrics.json                  # Must contain primary metric key
  outputs/best/                   # Exported best artifacts
  paper/figures/                  # Generated figures
```

### Important Constants and Conventions

**Node statuses** (plain strings, not enum): `"pending"`, `"running"`, `"evaluated"`, `"failed"`, `"oom"`, `"pruned"`, `"expanded"`, `"timeout"`

**Priority formula**: `lcb - lambda_cost * total_cost + beta_exploration * (1/sqrt(eval_runs + 1))`. Infeasible nodes get `-inf`; unevaluated nodes get `+inf` (always expanded first).

**Reward formula**: `primary_value - constraint_penalty * n_violated - lambda_cost * normalized_cost - kl_coef * kl_divergence`. Failed/timeout/oom nodes get hardcoded `-100.0`.

**Draft root categorization**: Root nodes are split into 3 equal groups: `"baseline"`, `"open_problem"`, `"novel"`.

**Seed derivation**: `SHA-256(base_seed:node_id:repeat_idx) % 2^31` — deterministic and reproducible.

**Signal handling**: `SearchManager` traps SIGINT → saves checkpoint → exits with code 20 (distinguishes graceful stop from crash).

### LLM Dependency Injection Pattern

When a module only needs simple text generation (not schema validation or tool use), it accepts `agent_llm` as a `Callable[[str], Awaitable[str]]` rather than the full `AgentLLM` instance (e.g., `phase0/clustering.py`). Use `call_function()` for structured/validated calls, simple callables for free-text.

### Concurrency

- Phase 0: `asyncio` with per-provider rate limiters
- Phase 3: Parallel experiment runs across seeds (`ProcessPoolExecutor`)
- Phase 5: Single-process PPO (optional `accelerate` for distributed)
- vLLM engine is put to sleep (`sleep(level=2)`) during PPO updates to free GPU memory — vLLM and PyTorch training cannot coexist on the same GPU

### Multi-Language Support

`ProblemSpecModel.language` (`LanguageConfig`) carries `interpreter_command`, `file_extension`, `seed_arg_format`, and `code_block_tag`. `LocalExecutor` accepts these overrides, enabling R, Julia, or other language experiments.

## CLI Commands

```
sera init                   # Initialize workspace from Input-1 YAML
sera setup                  # Interactive wizard (--lang en/ja, --resume, --from-input1, --skip-phase0)
sera phase0-related-work    # Collect related work
sera freeze-specs           # Generate & freeze specs (--gpu-count, --memory-gb, --cpu-cores, --gpu-type, --gpu-required)
sera research               # Run Phases 2-6 loop
sera generate-paper         # Phase 7
sera evaluate-paper         # Phase 8
sera export-best            # Export best artifacts
sera visualize              # Generate interactive HTML visualization of search tree (--step N)
sera status                 # Show search tree status
sera show-node <id>         # Show node details
sera replay <id> <seed>     # Replay experiment
sera validate-specs         # Validate spec integrity
```

**Setup wizard** (`setup_cmd.py`): 11-step interactive flow via `rich` console. Phase A (steps 1-7) collects Input-1 data interactively, Phase B (8-9) runs Phase 0 with review, Phase C (10-11) detects GPU/SLURM environment and freezes specs. Bilingual (ja/en via `MESSAGES` dict). Resumable via `.wizard_state.json`.

## Testing Patterns

Key shared fixtures in `tests/conftest.py`:
- `tmp_workspace` — Creates temporary directory structure mimicking a SERA workspace
- `sample_input1` — Test Input-1 YAML data (Iris classification)
- `mock_llm_response` — Simple LLM response mock generator

**`set_mock()` hook**: `AgentLLM` and `PPOTrainer` expose `set_mock(fn)` to inject a callable for testing, bypassing all real I/O. Tests never monkey-patch internals — always use this hook:
```python
agent_llm.set_mock(lambda prompt, purpose: f"mock:{purpose}")
ppo_trainer.set_mock(lambda rollouts: {"mean_reward": 0.5})
```

**Agent-layer tests** (`tests/test_agent/`): Use `AsyncMock` for `AgentLLM` (mock `generate_with_tools` as `AsyncMock` with `.return_value` or `.side_effect` lists). The `ToolExecutor` and `AgentLoop` tests mock `SearchManager` with `SimpleNamespace` (consistent with spec mock pattern).

**Duck-typed spec mocks**: Tests use `types.SimpleNamespace` instead of real Pydantic models, since components read specs via `getattr(obj, "field", default)` with two-level fallback patterns. This makes tests lightweight:
```python
spec = SimpleNamespace(search=SimpleNamespace(lambda_cost=0.1, beta_exploration=0.05))
```

**Markers**: `@pytest.mark.gpu`, `@pytest.mark.slow`, `@pytest.mark.network`. Default safe run: `pytest -m "not gpu and not slow and not network" tests/`
