# SERA 要件定義書 — 実装手順書

> 本ファイルは TASK.md v13.0 を分割したものである。目次は [README.md](./README.md) を参照。

---

## 22. 実装手順書（Implementation Guide）

> **このセクションは実装者（Claude等のAIエージェント）向けの具体的な作業指示書である。**
> §20のMVP優先順位に従い、各ステップで「何を作り」「何をテストし」「何が完了条件か」を明示する。

### 22.1 実装の大原則

```text
1. ボトムアップ: ユーティリティ → Specモデル → 各Phase モジュール → CLI → 統合テスト
2. 各ステップ完了後に必ずテストを書いて通すこと（Red-Green サイクル）
3. モックファースト: 外部API・LLM・GPU を必要とする部分はモックで先にテストを通す
4. 1ファイル1責務: 1つのモジュールが複数の責務を持たないこと
5. 型ヒント必須: 全関数に引数・戻り値の型ヒントをつけること
6. passは禁止: 本書の擬似コードにある pass は実装時に必ず実コードに置き換えること
```

### 22.2 Step 0: プロジェクトブートストラップ

**作業内容**: プロジェクトの骨格を作成する。

```bash
# 実行するコマンド
mkdir -p src/sera/{specs,phase0/api_clients,phase1,search,execution,evaluation,learning,lineage,paper,agent,utils}
mkdir -p tests/{test_specs,test_phase0,test_search,test_evaluation,test_lineage,test_paper,test_cli}
touch src/sera/__init__.py
touch src/sera/{cli,specs/__init__,phase0/__init__,phase0/api_clients/__init__,phase1/__init__,search/__init__,execution/__init__,evaluation/__init__,learning/__init__,lineage/__init__,paper/__init__,agent/__init__,utils/__init__}.py
touch tests/__init__.py tests/conftest.py
```

**pyproject.toml を作成**（§18.2 の内容をそのまま使用）:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sera"
version = "0.1.0"
description = "Self-Evolving Research Agent"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "trl>=0.8.0",
    "safetensors>=0.4.0",
    "typer>=0.12.0",
    "pyyaml>=6.0",
    "httpx>=0.27.0",
    "tenacity>=8.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "graphviz>=0.20.0",
    "numpy>=1.26.0",
    "pydantic>=2.6.0",
    "rich>=13.7.0",
    "structlog>=24.1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "pytest-mock>=3.12", "respx>=0.21", "ruff>=0.3"]
slurm = ["submitit>=1.5.0"]
docker = ["docker>=7.0.0"]

[project.scripts]
sera = "sera.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/sera"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 120
```

**tests/conftest.py の雛形**:
```python
import pytest
from pathlib import Path
import tempfile
import json
import yaml

@pytest.fixture
def tmp_workspace(tmp_path):
    """一時的な sera_workspace ディレクトリを作成"""
    dirs = ["specs", "related_work/results", "related_work/teacher_papers",
            "lineage/nodes", "runs", "logs", "checkpoints", "outputs/best",
            "paper/figures", "docs/modules"]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture
def sample_input1():
    """テスト用 Input-1"""
    return {
        "version": 1,
        "data": {"description": "UCI Iris dataset", "location": "./data/iris.csv", "format": "csv", "size_hint": "small(<1GB)"},
        "domain": {"field": "ML", "subfield": "classification"},
        "task": {"brief": "Classify iris species", "type": "prediction"},
        "goal": {"objective": "maximize accuracy", "direction": "maximize", "baseline": "0.95"},
        "constraints": [{"name": "inference_time_ms", "type": "le", "threshold": 100}],
        "notes": "",
    }

@pytest.fixture
def mock_llm_response():
    """LLM応答のモック生成器"""
    def _mock(content: str):
        return content
    return _mock
```

**完了条件**:
- `pip install -e ".[dev]"` が成功する
- `python -c "import sera"` がエラーなく通る
- `pytest` が 0 テスト 0 エラーで通る

#### 22.2.1 GPU ノードでの環境セットアップ（SLURM クラスタ）

> **重要**: ログインノード（GPU なし）で `pip install` を実行すると、PyTorch が CUDA ランタイムを検出できず `torch.cuda.is_available() = False` になる。local LLM を GPU で実行するには、**必ず GPU のある計算ノード上**でインストールを行うこと。

**セットアップスクリプト**: `scripts/setup_env.sh`

このスクリプトは以下を自動で行う:
1. GPU の存在と CUDA バージョンを検出
2. `.venv` を作成
3. 検出された CUDA バージョンに対応する PyTorch ホイールを `--index-url` 指定でインストール
4. `sera[dev,slurm]` をインストール
5. `torch.cuda.is_available() = True` であることを検証

```bash
# GPU ノード上でセットアップ実行
srun --partition=<gpu-partition> --time=01:00:00 bash scripts/setup_env.sh

# 以後は .venv を activate して使用
source .venv/bin/activate
```

**研究実行ジョブスクリプト**: `scripts/run_research.sh`

```bash
# SLURM ジョブとして研究を実行
sbatch scripts/run_research.sh

# 中断した研究の再開
sbatch scripts/run_research.sh --resume
```

**GPU が不要な操作**（ログインノードで実行可能）:
- `sera init` / `sera phase0-related-work` / `sera freeze-specs` / `sera status` / `sera validate-specs`
- `pytest -m "not gpu" tests/`

**GPU が必要な操作**（計算ノードで実行すること）:
- `sera research`（local LLM 使用時）
- `sera generate-paper` / `sera evaluate-paper`（local LLM 使用時）

> **注意**: `ModelSpec.agent_llm.provider` が `"openai"` や `"anthropic"` の場合は API 経由のため GPU 不要。

---

### 22.3 Step 1: ユーティリティモジュール（`src/sera/utils/`）

**作業順序と各ファイルの実装内容**:

#### 22.3.1 `utils/seed.py`
```python
"""実装すべき関数"""
def set_global_seed(seed: int) -> None:
    """np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all, random.seed を一括設定"""

def get_seed_for_node(base_seed: int, node_id: str, repeat_idx: int) -> int:
    """ノードIDと反復インデックスから決定論的にseedを導出。hash(node_id + repeat_idx) % 2**31"""
```

#### 22.3.2 `utils/hashing.py`
```python
"""実装すべき関数"""
def compute_spec_hash(spec_dict: dict) -> str:
    """dictをcanonical JSON化しSHA-256を計算。'sha256:xxxx' 形式で返す"""

def compute_adapter_spec_hash(adapter_spec: dict) -> str:
    """type+target_modules+target_layers+rank+alpha からハッシュ。形状契約の同一性判定に使用"""

def verify_spec_hash(spec_path: Path, lock_path: Path) -> bool:
    """spec_path のハッシュと lock_path の記録を比較"""
```

#### 22.3.3 `utils/logging.py`
```python
"""実装すべき関数・クラス"""
class JsonlLogger:
    """指定パスに JSONL 形式でログを追記するロガー"""
    def __init__(self, path: Path): ...
    def log(self, event: dict) -> None: ...  # timestamp 自動付与、json.dumps + '\n' で追記

def setup_structlog() -> None:
    """structlog の設定。Rich + JSONL 出力"""
```

#### 22.3.4 `utils/checkpoint.py`
```python
"""実装すべき関数"""
def save_checkpoint(state: dict, checkpoint_dir: Path, step: int) -> Path: ...
def load_latest_checkpoint(checkpoint_dir: Path) -> dict | None: ...
```

**テスト**: `tests/test_utils/` に各関数の単体テスト。hashing はラウンドトリップ、seed は決定論性を検証。

---

### 22.4 Step 2: Pydantic Spec モデル（`src/sera/specs/`）

**全 Spec を Pydantic v2 BaseModel として定義する。§3〜§5 の YAML スキーマをそのまま Python クラスに変換する。**

**実装順序**（依存関係順）:

```text
1. input1.py          — Input1Model（§3.1 のフィールド）
2. related_work_spec.py — Paper, Cluster, BaselineCandidate, RelatedWorkSpecModel（§4.4.1）
3. paper_spec.py       — SectionRequirement, FigureRequirement, PaperSpecModel（§4.4.2）
4. paper_score_spec.py — Criterion, PaperScoreSpecModel（§4.4.3）
5. teacher_paper_set.py — TeacherPaper, TeacherPaperSetModel（§4.4.4）
6. problem_spec.py     — Constraint, Variable, ProblemSpecModel（§5.5）
7. model_spec.py       — BaseModelConfig, AdapterSpec, ModelSpecModel（§5.3）
8. resource_spec.py    — ComputeConfig, NetworkConfig, ResourceSpecModel（§5.6）
9. plan_spec.py        — BranchingOp, RewardConfig, PlanSpecModel（§5.7）
10. execution_spec.py  — SearchConfig, EvaluationConfig, LearningConfig, ... ExecutionSpecModel（§5.4）
```

**実装パターン（全Specで共通）**:

```python
# 例: execution_spec.py
from pydantic import BaseModel, model_validator
from sera.utils.hashing import compute_spec_hash

class SearchConfig(BaseModel):
    strategy: str = "best_first"
    priority_rule: str = "epsilon_constraint_lcb"
    lambda_cost: float = 0.1
    beta_exploration: float = 0.05
    max_nodes: int = 100
    max_depth: int = 10
    branch_factor: int = 3
    initial_root_children: int = 5

class EvaluationConfig(BaseModel):
    repeats: int = 3
    lcb_coef: float = 1.96
    sequential_eval: bool = True
    sequential_eval_initial: int = 1
    sequential_eval_topk: int = 5
    bootstrap: bool = False
    bootstrap_samples: int = 1000

# ... 他の Config も同様 ...

class ExecutionSpecModel(BaseModel):
    search: SearchConfig = SearchConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    learning: LearningConfig = LearningConfig()
    lora_runtime: LoraRuntimeConfig = LoraRuntimeConfig()
    pruning: PruningConfig = PruningConfig()
    termination: TerminationConfig = TerminationConfig()
    paper: PaperConfig = PaperConfig()

    def compute_hash(self) -> str:
        return compute_spec_hash(self.model_dump())

    # YAML <-> Model 変換
    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutionSpecModel":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("execution_spec", data))

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump({"execution_spec": self.model_dump()}, f, default_flow_style=False)
```

**共通ユーティリティ** — `specs/__init__.py` に以下を実装:
```python
class AllSpecs:
    """全Specを束ねるコンテナ。Phase 1 完了後に構築される"""
    input1: Input1Model
    related_work: RelatedWorkSpecModel
    paper_spec: PaperSpecModel
    paper_score_spec: PaperScoreSpecModel
    teacher_papers: TeacherPaperSetModel
    problem_spec: ProblemSpecModel
    model_spec: ModelSpecModel
    resource_spec: ResourceSpecModel
    plan_spec: PlanSpecModel
    execution_spec: ExecutionSpecModel

    @classmethod
    def load_from_dir(cls, specs_dir: Path) -> "AllSpecs": ...
    def save_to_dir(cls, specs_dir: Path) -> None: ...
```

**テスト**: `tests/test_specs/` — 各 Spec について:
- 既定値でインスタンス化できること
- YAML ラウンドトリップ（to_yaml → from_yaml で同一）
- 不正値で ValidationError が発生すること
- ExecutionSpec のハッシュが決定論的であること

**完了条件**: 全10 Spec モデルが定義され、テストが通る。

---

### 22.5 Step 3: Phase 0 — 先行研究エンジン（`src/sera/phase0/`）

**実装順序**:

```text
1. api_clients/base.py         — BaseScholarClient（ABC）
2. api_clients/semantic_scholar.py — SemanticScholarClient
3. api_clients/crossref.py     — CrossRefClient
4. api_clients/arxiv.py        — ArxivClient
5. api_clients/web_search.py   — WebSearchClient（SerpAPI）
6. ranking.py                  — citation_norm, relevance_score, rank_papers
7. clustering.py               — cluster_papers（LLMベース）
8. related_work_engine.py      — RelatedWorkEngine（統合エントリポイント）
```

**APIクライアント共通パターン**:
```python
# api_clients/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@dataclass
class PaperResult:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    abstract: str
    citation_count: int
    url: str
    doi: str
    arxiv_id: str
    source_api: str

class BaseScholarClient(ABC):
    @abstractmethod
    async def search(self, query: str, limit: int, year_from: int | None) -> list[PaperResult]: ...

    @abstractmethod
    async def get_references(self, paper_id: str, limit: int) -> list[PaperResult]: ...

    @abstractmethod
    async def get_citations(self, paper_id: str, limit: int) -> list[PaperResult]: ...
```

```python
# api_clients/semantic_scholar.py — 具体実装の構造
class SemanticScholarClient(BaseScholarClient):
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,abstract,year,citationCount,authors,venue,externalIds,url"

    def __init__(self, api_key: str | None = None):
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(base_url=self.BASE_URL, headers=headers, timeout=30.0)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=60),
           retry=retry_if_exception_type(httpx.HTTPStatusError))
    async def search(self, query: str, limit: int = 20, year_from: int | None = None) -> list[PaperResult]:
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        if year_from:
            params["year"] = f"{year_from}-"
        resp = await self._client.get("/paper/search", params=params)
        resp.raise_for_status()
        # パース処理...
```

**related_work_engine.py**（統合）:
```python
class RelatedWorkEngine:
    """
    Phase 0 の統合エントリポイント。
    1. Input-1 → LLM でクエリ生成
    2. API優先順位に従い論文取得
    3. ランキング・クラスタリング
    4. RelatedWorkSpec, PaperSpec, PaperScoreSpec, TeacherPaperSet を生成
    5. 全クエリを queries.jsonl に記録
    """
    def __init__(self, clients: list[BaseScholarClient], agent_llm: AgentLLM, logger: JsonlLogger): ...

    async def run(self, input1: Input1Model, config: Phase0Config) -> Phase0Output:
        # 1. クエリ生成
        queries = await self._build_queries(input1)
        # 2. 各クエリでAPI検索（フォールバック付き）
        papers = await self._search_with_fallback(queries, config)
        # 3. ランキング
        ranked = rank_papers(papers, config.ranking_weight)
        # 4. クラスタリング
        clusters = await cluster_papers(ranked[:config.top_k_papers], self.agent_llm)
        # 5. Spec 生成
        return Phase0Output(
            related_work_spec=self._build_related_work_spec(ranked, clusters),
            paper_spec=await self._build_paper_spec(ranked, input1),
            paper_score_spec=await self._build_paper_score_spec(ranked, input1),
            teacher_paper_set=self._build_teacher_set(ranked, config),
        )

    async def _search_with_fallback(self, queries, config):
        """§4.2.1 の優先順位に従い、APIフォールバック"""
        all_papers = []
        for query in queries:
            for client in self.clients:  # 優先順位順
                try:
                    results = await client.search(query.text, limit=config.top_k_papers)
                    # queries.jsonl にログ
                    self.logger.log({...})
                    all_papers.extend(results)
                    if len(all_papers) >= config.top_k_papers:
                        break
                except Exception as e:
                    self.logger.log({"error": str(e), ...})
                    continue  # 次のAPIへフォールバック
        return all_papers
```

**テスト方法**:
- `respx` で HTTP レスポンスをモック（各APIクライアント）
- `mock_llm_response` でクエリ生成・クラスタリングの LLM 応答をモック
- `related_work_engine` は統合テスト（全モック）

**完了条件**:
- 各 API クライアントがモックで正常応答を返す
- フォールバック（API1失敗→API2成功）が動作する
- queries.jsonl にクエリログが記録される
- Phase0Output から全4つの Spec が生成される

---

### 22.6 Step 4: Phase 1 — Spec確定＋凍結（`src/sera/phase1/`）

**実装ファイル**:

```text
1. spec_builder.py  — LLM で ProblemSpec, PlanSpec を草案生成
2. spec_freezer.py  — ExecutionSpec ハッシュ計算・lock ファイル書き出し・検証
```

```python
# spec_builder.py
class SpecBuilder:
    def __init__(self, agent_llm: AgentLLM): ...

    async def build_problem_spec(self, input1: Input1Model, related_work: RelatedWorkSpecModel) -> ProblemSpecModel:
        """
        LLM に input1 + related_work を与え、ProblemSpec の JSON を生成させる。
        生成結果を ProblemSpecModel(**json_output) でバリデーション。
        バリデーション失敗時は最大3回リトライ（エラーメッセージをLLMに返して修正させる）。
        """

    async def build_plan_spec(self, input1: Input1Model, problem_spec: ProblemSpecModel) -> PlanSpecModel:
        """同様にLLMでPlanSpecを生成"""

# spec_freezer.py
class SpecFreezer:
    def freeze(self, specs: AllSpecs, specs_dir: Path) -> None:
        """
        1. 全 Spec を specs_dir に YAML で保存
        2. ExecutionSpec のハッシュを計算
        3. execution_spec.yaml.lock に書き出し
        """

    def verify(self, specs_dir: Path) -> bool:
        """execution_spec.yaml と .lock のハッシュを比較。不一致は False"""
```

**テスト**: LLMモックでSpec生成→バリデーション通過、ハッシュのラウンドトリップ検証

---

### 22.7 Step 5: AgentLLM（`src/sera/agent/`）

**注意: これは全Phase で横断的に使われるため、早めに実装する。ただし完全実装はPhase 5（PPO）時点。Step 5 では推論機能のみ。**

```text
1. agent_llm.py         — §付録C のインターフェース実装
2. prompt_templates.py  — 各Phase用プロンプトテンプレート（文字列テンプレート集）
```

**agent_llm.py の実装戦略**:

```python
class AgentLLM:
    def __init__(self, model_spec: ModelSpecModel, resource_spec: ResourceSpecModel, log_path: Path):
        self.model_spec = model_spec
        self.logger = JsonlLogger(log_path)
        self._provider = model_spec.agent_llm.provider

        if self._provider == "local":
            # transformers + peft でモデルロード
            # ※ GPU がない環境ではスキップ可能に（テスト用）
            self._init_local_model()
        elif self._provider == "openai":
            import openai
            self._client = openai.AsyncOpenAI(api_key=os.environ[resource_spec.api_keys.openai])
        elif self._provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=os.environ[resource_spec.api_keys.anthropic])

    async def generate(self, prompt: str, purpose: str, ...) -> "GenerationOutput":
        # 1. provider に応じて推論
        # 2. agent_llm_log.jsonl にログ（tool_calls, turn_rewards フィールド含む）
        # 3. GenerationOutput(text=result, tool_calls=None, purpose=purpose) を返す
        #    ※ Phase A/B では tool_calls は常に None
        #    ※ Phase C で generate_with_tools() が有効化された場合のみ tool_calls に値が入る

    async def generate_with_tools(self, prompt: str, available_tools: list[dict],
                                  purpose: str, ...) -> "GenerationOutput":
        # Phase C で有効化。ToolRegistry からツール定義を取得し、LLMに渡す。
        # 戻り値: GenerationOutput(text=..., tool_calls=[ToolCall(...)], purpose=purpose)
        # Phase A/B では NotImplementedError を送出（段階的有効化）

    def load_tools(self, tool_registry: "ToolRegistry") -> None:
        # ToolRegistry をロードし、generate_with_tools で利用可能にする

    # load_adapter, get_log_probs, get_turn_log_probs は Step 9（PPO実装時）に追加
```

**prompt_templates.py**:
```python
"""
各Phaseで使うプロンプトテンプレートを定数として定義。
str.format() または jinja2 で変数埋め込み。

定義すべきテンプレート:
- QUERY_GENERATION_PROMPT      — Phase 0: Input-1 → 検索クエリ生成
- PAPER_CLUSTERING_PROMPT      — Phase 0: 論文クラスタリング
- RELEVANCE_SCORING_PROMPT     — Phase 0: 関連度スコアリング
- SPEC_GENERATION_PROMPT       — Phase 1: ProblemSpec/PlanSpec 生成
- DRAFT_PROMPT                 — Phase 2: draft オペレータ（§6.5.1 のプロンプト）
- DEBUG_PROMPT                 — Phase 2: debug オペレータ（§6.5.2 のプロンプト）
- IMPROVE_PROMPT               — Phase 2: improve オペレータ（§6.5.3 のプロンプト）
- EXPERIMENT_CODE_PROMPT       — Phase 3: 実験コード生成
- PAPER_OUTLINE_PROMPT         — Phase 7: アウトライン生成
- PAPER_FULL_GENERATION_PROMPT — Phase 7: 論文全体1パス生成（ステップ5b）
- PAPER_WRITEUP_REFLECTION_PROMPT — Phase 7: ライティング内反省（ステップ5c）
- CITATION_SEARCH_PROMPT       — Phase 7: 不足引用特定+検索クエリ生成（ステップ3）
- CITATION_SELECT_PROMPT       — Phase 7: 検索結果から関連論文選択（ステップ3）
- PLOT_AGGREGATION_PROMPT      — Phase 7: 図集約スクリプト生成（ステップ2）
- VLM_FIGURE_DESCRIPTION_PROMPT — Phase 7: VLM図記述生成（ステップ4）
- VLM_FIGURE_CAPTION_REVIEW_PROMPT — Phase 7: VLM図・キャプション整合性レビュー（ステップ5c-iv）
- VLM_DUPLICATE_DETECTION_PROMPT — Phase 7: VLM重複図検出（ステップ5c-v）
- PAPER_EVALUATION_PROMPT      — Phase 8: 論文評価（§12.1 のプロンプト）
- REVIEWER_REFLECTION_PROMPT   — Phase 8: レビュアー反省ループ（§12.1 ステップ1d）
- META_REVIEW_PROMPT           — Phase 8: Area Chair メタレビュー（§12.1 ステップ2）
- PAPER_REVISION_PROMPT        — Phase 8: 論文修正指示

各テンプレートは {変数名} プレースホルダーを含み、呼び出し側で .format(**kwargs) する。
"""
```

**テスト**: OpenAI/Anthropic クライアントはモック。local は conftest で `skip_if_no_gpu` マーカー。

---

### 22.8 Step 6: Phase 2 — 探索木（`src/sera/search/`）

```text
1. search_node.py       — SearchNode データクラス（§6.2）
2. priority.py          — compute_priority, compute_exploration_bonus（§6.3）
3. validation.py        — validate_experiment_config（§6.6.1）
4. tree_ops.py          — TreeOps クラス: draft / debug / improve 3オペレータ（§6.5）
5. search_manager.py    — SearchManager: select_next_node + メインループ（§6.4, §6.11）
```

**tree_ops.py の重要ポイント（AIDE参考3オペレータ）**:

```python
class TreeOps:
    """AIDE参考の3オペレータを提供する（§6.5）"""

    def __init__(self, specs: AllSpecs, agent_llm: AgentLLM, rng):
        self.specs = specs
        self.agent_llm = agent_llm
        self.rng = rng

    def draft(self, n: int) -> list[SearchNode]:
        """§6.5.1: 新規アプローチの起草。親なし。
        ルート時: baseline/open_problem/novel を n//3 ずつ配分。
        再draft時: 既存ノード一覧を提示し異なるアプローチを要求。"""

    def debug(self, failed_node: SearchNode) -> SearchNode:
        """§6.5.2: 失敗ノードの修復。experiment_config は変更せずコードのみ修正。
        debug_depth をインクリメント。max_debug_depth 超過なら呼ばれない（§6.4で制御）。"""

    def improve(self, parent: SearchNode, all_nodes: dict,
                n_children: int) -> list[SearchNode]:
        """§6.5.3: 原子的改善。1子=1変数変更。
        validate_experiment_config でバリデーション。"""
```

**search_manager.py の重要ポイント**:

```python
import heapq

class SearchManager:
    def __init__(self, specs: AllSpecs, agent_llm: AgentLLM, executor: Executor,
                 evaluator: Evaluator, ppo_trainer: PPOTrainer | None,
                 lineage_manager: LineageManager, tree_ops: TreeOps,
                 logger: JsonlLogger):
        self.open_list: list[SearchNode] = []
        self.closed_set: set[str] = set()
        self.all_nodes: dict[str, SearchNode] = {}
        self.best_node: SearchNode | None = None
        self.tree_ops = tree_ops
        self.step = 0
        ...

    def run(self) -> SearchNode:
        """§6.11 の research_loop を実装。チェックポイント保存・復帰対応。
        1. tree_ops.draft() でルートノード生成
        2. select_next_node() でオペレータ自動選択（§6.4）
        3. evaluate / debug / draft / improve を分岐実行
        4. PPO更新・剪定を適宜実行"""

    def select_next_node(self) -> tuple[SearchNode | None, str]:
        """§6.4: 状態に基づくオペレータ自動選択。
        pending → 'evaluate', failed → 'debug', 多様性不足 → 'draft', else → 'improve'"""

    def _should_terminate(self) -> bool:
        """§5.4 termination 条件を全てチェック"""
```

**テスト**:
- `search_node.py`: データクラスの生成・シリアライズ
- `priority.py`: 既知の入力に対する期待値の一致
- `validation.py`: ホワイトリスト検証（許可キー/禁止キー/範囲外の各パターン）
- `tree_ops.py`: モックLLMで draft/debug/improve 各オペレータの単体テスト
- `search_manager.py`: max_nodes=5 の小規模探索が完走すること（全モック、3オペレータ遷移確認）

---

### 22.9 Step 7: Phase 3 — 実験実行（`src/sera/execution/`）

```text
1. executor.py              — Executor ABC + RunResult（§7.3）
2. local_executor.py        — LocalExecutor（subprocess.Popen）
3. experiment_generator.py  — ExperimentGenerator（LLMによるコード生成）
4. slurm_executor.py        — SlurmExecutor（MVP後、スタブのみ先に作成）
5. docker_executor.py       — DockerExecutor（MVP後、スタブのみ先に作成）
```

**local_executor.py の実装ポイント**:

```python
import subprocess
import time
import signal

class LocalExecutor(Executor):
    def __init__(self, work_dir: Path, resource_spec: ResourceSpecModel):
        self.work_dir = work_dir
        self.timeout = resource_spec.sandbox.experiment_timeout_sec
        self.memory_limit = resource_spec.sandbox.experiment_memory_limit_gb

    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int | None = None) -> RunResult:
        run_dir = self.work_dir / "runs" / node_id
        run_dir.mkdir(parents=True, exist_ok=True)
        timeout = timeout_sec or self.timeout

        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"

        start = time.monotonic()
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            try:
                proc = subprocess.Popen(
                    ["python", str(script_path), "--seed", str(seed), "--output-dir", str(run_dir)],
                    stdout=out, stderr=err,
                    env={**os.environ, "SERA_NODE_ID": node_id, "SERA_SEED": str(seed)},
                )
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return RunResult(node_id=node_id, success=False, exit_code=-9, ...)
            except MemoryError:
                return RunResult(node_id=node_id, success=False, exit_code=-7, ...)

        wall_time = time.monotonic() - start
        metrics_path = run_dir / "metrics.json" if (run_dir / "metrics.json").exists() else None

        return RunResult(
            node_id=node_id, success=(exit_code == 0), exit_code=exit_code,
            stdout_path=stdout_path, stderr_path=stderr_path,
            metrics_path=metrics_path, artifacts_dir=run_dir / "artifacts",
            wall_time_sec=wall_time, seed=seed,
        )
```

**テスト**: 簡単な Python スクリプト（`echo '{"primary":{"name":"acc","value":0.9}}' > metrics.json`）を実行し、RunResult を検証。タイムアウトテストも。

**slurm_executor.py の実装ポイント**:

SlurmExecutor は `submitit` ライブラリを使用して SLURM クラスタにジョブを投入する。
ResourceSpecModel の SlurmConfig（partition, account, time_limit, modules, sbatch_extra）を読み込む。

```python
import time
from pathlib import Path
from sera.execution.executor import Executor, RunResult
from sera.specs.resource_spec import SlurmConfig

class SlurmExecutor(Executor):
    """submitit 経由で SLURM ジョブを投入・完了待ち・結果収集"""

    def __init__(self, work_dir: Path, slurm_config: SlurmConfig, python_executable: str = "python"):
        self.work_dir = Path(work_dir)
        self.slurm_config = slurm_config
        self.python_executable = python_executable

    def run(self, node_id: str, script_path: Path, seed: int, timeout_sec: int | None = None) -> RunResult:
        """
        1. runs/<node_id>/ ディレクトリを作成
        2. submitit.AutoExecutor を設定（partition, account, time_limit, modules, sbatch_extra）
        3. ラッパー関数（subprocess で実験スクリプトを実行）をジョブとして投入
        4. job.results() でブロッキング完了待ち（timeout_sec 超過時はジョブキャンセル）
        5. stdout/stderr をジョブログから run_dir にコピー
        6. metrics.json の有無をチェック
        7. OOM判定: sacct の MaxRSS / State=OUT_OF_MEMORY、または stderr 内のパターン検出
        8. RunResult を返却
        """
        pass
```

実装要件:
- `submitit` が未インストールの場合は `ImportError` を明確なメッセージ付きで送出
- `slurm_config.modules` の各モジュールは `module load` コマンドでロード（sbatch スクリプトの前処理として）
- `slurm_config.sbatch_extra` は submitit の `additional_parameters` として渡す
- タイムアウト: `timeout_sec` を SLURM の wall-time limit と独立に管理。Python 側のポーリングループで `timeout_sec` 超過時に `scancel` でジョブをキャンセルし `exit_code=-9` を返す
- OOM検知: SLURM ジョブステート `OUT_OF_MEMORY` またはexit code 137、stderr 内の OOM パターンで判定し `exit_code=-7` を返す
- ポーリング間隔: 10秒（デフォルト）
- ジョブ名: `sera-{node_id[:8]}` で識別可能にする

**テスト**: submitit をモックし、正常完了・タイムアウト・OOM・submitit未インストールのケースを検証。

---

### 22.10 Step 8: Phase 4 — 統計評価（`src/sera/evaluation/`）

```text
1. evaluator.py              — Evaluator ABC
2. statistical_evaluator.py  — StatisticalEvaluator（§8.1〜8.2 の実装）
3. feasibility.py            — check_feasibility（§8.3）
```

**statistical_evaluator.py**:
```python
class StatisticalEvaluator:
    """§8.1〜8.2 の実装。update_stats, evaluate_node_sequential, evaluate_node_full"""

    def __init__(self, executor: Executor, exec_spec: ExecutionSpecModel, problem_spec: ProblemSpecModel): ...

    def evaluate_initial(self, node: SearchNode) -> None:
        """sequential_eval_initial 回実行して暫定統計を計算"""

    def evaluate_full(self, node: SearchNode) -> None:
        """repeats まで追加実行して最終統計を計算"""

    def is_topk(self, node: SearchNode, all_nodes: list[SearchNode]) -> bool:
        """LCBでソートしてTop-kに入るか判定"""

    @staticmethod
    def update_stats(node: SearchNode, lcb_coef: float) -> None:
        """§8.2 の update_stats をそのまま実装"""
```

**テスト**: 既知の値リスト（[0.7, 0.8, 0.9]）に対する μ, SE, LCB の計算結果を検証。

---

### 22.11 Step 9: Phase 5-6 — PPO + LoRA系譜（`src/sera/learning/` + `src/sera/lineage/`）

**これが最も複雑なステップ。GPU が必要な部分はモックテスト優先。**

```text
learning/
  1. rollout.py       — PPORollout, PPORolloutV2 データクラス（§9.2）
  2. reward.py        — compute_reward, compute_reward_v2（§9.2）
  3. ppo_trainer.py   — PPOTrainer（§9.3、GRPOモード/PPOモード切替対応）

lineage/
  4. cache.py         — LRUCache（collections.OrderedDict ベース）
  5. lineage_manager.py — materialize, save_delta, squash（§10.2〜10.3）
  6. pruner.py        — Pareto剪定、LCB閾値剪定、予算剪定（§10.4）
```

**PPORolloutV2 と compute_reward_v2 の実装手順**:
```python
# rollout.py — PPORolloutV2 の追加
class PPORolloutV2(PPORollout):
    """§26.4.2: ターンレベル報酬を含む拡張ロールアウト"""
    turn_rewards: dict[str, float]    # {"phase2": 0.8, "phase3": 1.0, ...}
    turn_log_probs: dict[str, float]  # Phase毎のlog_prob

# reward.py — compute_reward_v2 の追加
def compute_reward_v2(node, turn_rewards, plan_spec) -> float:
    """§26.4.2: R = Σ_t(w_t * r_turn_t) - penalties"""
    # TurnRewardSpec は PlanSpec.turn_rewards に格納
    # plan_spec.turn_rewards.enabled == False の場合は compute_reward にフォールバック
```

**ppo_trainer.py の実装戦略**:

```python
class PPOTrainer:
    """
    trl ライブラリの PPOTrainer をラップ。
    ただし trl の API は頻繁に変わるため、以下の自前実装も許容:

    自前実装の場合の手順:
    1. rollouts から (prompt, response, reward) のバッチを構成
    2. model.generate() で response を再生成し、new_log_prob を取得
    3. old_log_prob との比率 r = exp(new_log_prob - old_log_prob) を計算
    4. GAE で advantage を計算
    5. PPO クリッピング損失を計算
    6. optimizer.step()（LoRA パラメータのみ）
    7. KL 制御

    trl 使用の場合:
    - trl.PPOConfig + trl.PPOTrainer を使い、peft モデルをそのまま渡す
    - trl が LoRA パラメータのみを更新することを確認
    """
```

**lineage_manager.py — materialize のテスト戦略**:
```python
# テスト: GPU不要（CPU上で小さなテンソルで検証）
def test_materialize_chain():
    """
    root(zeros) → child1(+0.1) → child2(+0.2) の3段系譜で、
    materialize(child2) == zeros + 0.1 + 0.2 を検証
    """

def test_materialize_with_snapshot():
    """
    root → child1 → child2(snapshot) → child3(+delta) で、
    materialize(child3) == snapshot + delta を検証（root/child1 は参照しない）
    """

def test_squash_creates_snapshot():
    """squash_depth=2 のとき、depth=2 のノードにスナップショットが生成されることを検証"""
```

---

### 22.12 Step 10: Phase 7-8 — 論文生成・評価（`src/sera/paper/`）

```text
1. evidence_store.py      — EvidenceStore（§11.2）
2. figure_generator.py    — FigureGenerator（matplotlib/seaborn/graphviz）
3. vlm_reviewer.py        — VLMReviewer（§11.4：図記述・キャプションレビュー・重複検出）
4. citation_searcher.py   — CitationSearcher（§11.5：Semantic Scholar自動引用検索ループ）
5. paper_composer.py      — PaperComposer（§11.3：6ステップ + ライティング内反省ループ）
6. paper_evaluator.py     — PaperEvaluator + PaperScoreResult（§12.1：アンサンブル+レビュアー反省）
```

**figure_generator.py の実装**:
```python
class FigureGenerator:
    def __init__(self, output_dir: Path, max_figures: int = 12, dpi: int = 300): ...

    def ci_bar_chart(self, nodes: list[SearchNode], output_name: str) -> Path:
        """各ノードの μ ± CI を棒グラフで描画。matplotlib.errorbar 使用"""

    def convergence_curve(self, data: list[tuple[int, float]], output_name: str) -> Path:
        """step vs best_lcb の折れ線グラフ"""

    def search_tree(self, nodes: dict[str, SearchNode], top_n: int, output_name: str) -> Path:
        """graphviz でツリー描画。LCB値をノードラベルに表示"""

    def ablation_table(self, data: dict, output_name: str) -> Path:
        """アブレーション結果をヒートマップまたはテーブル画像で出力"""

    def aggregate_plots(self, evidence: EvidenceStore, agent_llm: AgentLLM,
                        n_reflections: int = 5) -> list[Path]:
        """
        LLMが実験結果を統合した追加集約図スクリプトを生成。
        反省ループで改善（最大 n_reflections 回）。
        AI-Scientist-v2 の plot aggregation に相当。
        """
```

**vlm_reviewer.py の実装**:
```python
class VLMReviewer:
    """§11.4 参照。VLM による図の視覚的レビュー。"""
    # ModelSpec.vlm が null の場合、PaperComposer は本クラスを None として扱う
    # describe_figures(), review_figure_caption_refs(), detect_duplicate_figures() を実装
```

**citation_searcher.py の実装**:
```python
class CitationSearcher:
    """§11.5 参照。Semantic Scholar API を使った自動引用検索ループ。"""
    # Phase 0 の SemanticScholarClient を再利用
    # 各ラウンドを citation_search_log.jsonl に記録（再現性）
```

**paper_evaluator.py のアンサンブル実装**:
```python
class PaperEvaluator:
    """§12.1 参照。以下の機能を実装:
    - 単体レビュー生成（Few-shot + bias_mode 対応）
    - レビュアー反省ループ（num_reviewer_reflections 回）
    - アンサンブル集約（num_reviews_ensemble > 1 の場合）
    - メタレビュー生成（Area Chair モード）
    """
```

**テスト**:
- ダミーデータで図が生成され、PNG ファイルが存在することを検証（画像内容は検証不要）
- CitationSearcher: Semantic Scholar API をモックし、検索ループが正しく動作することを検証
- VLMReviewer: VLM API をモックし、図記述・レビュー・重複検出の出力形式を検証
- PaperEvaluator: LLM をモックし、アンサンブル集約・メタレビュー・反省ループのフローを検証
- PaperComposer: ライティング内反省ループが最大回数で停止することを検証

---

### 22.13 Step 11: CLI（`src/sera/cli.py`）

**Typer で全コマンドを定義。各コマンドは対応するモジュールを呼び出すだけの薄いラッパー。**

```python
import typer
from pathlib import Path

app = typer.Typer(name="sera", help="Self-Evolving Research Agent")

@app.command()
def init(input1_path: Path, work_dir: Path = Path("./sera_workspace")):
    """Input-1 を読み込み、workspace を初期化"""
    # 1. work_dir 作成（§14 のディレクトリ構造）
    # 2. input1.yaml を specs/ にコピー
    # 3. 成功メッセージ

@app.command()
def phase0_related_work(
    work_dir: Path = Path("./sera_workspace"),
    topk: int = 10,
    teacher_papers: int = 5,
    citation_depth: int = 1,
    years_bias: int = 5,
    api_priority: str = "semantic_scholar,crossref,arxiv,web",
):
    """Phase 0: 先行研究収集"""
    # 1. Input-1 ロード
    # 2. AgentLLM 初期化
    # 3. RelatedWorkEngine.run()
    # 4. 結果を specs/ に保存

@app.command()
def freeze_specs(work_dir: Path = Path("./sera_workspace"), auto: bool = False):
    """Phase 1: 全Spec確定、ExecutionSpec固定"""
    # 1. Phase 0 出力ロード
    # 2. SpecBuilder で ProblemSpec, PlanSpec 生成
    # 3. auto=false なら specs/ を開いてユーザ確認待ち
    # 4. SpecFreezer.freeze()

@app.command()
def research(work_dir: Path = Path("./sera_workspace"), resume: bool = False):
    """Phase 2-6: 研究ループ"""
    # 1. AllSpecs ロード
    # 2. ExecutionSpec ハッシュ検証（失敗なら exit(2)）
    # 3. resume なら checkpoint ロード
    # 4. SearchManager.run()
    # 5. export_best 自動実行

@app.command()
def export_best(work_dir: Path = Path("./sera_workspace")):
    """best成果物を outputs/best/ に集約"""

@app.command()
def generate_paper(work_dir: Path = Path("./sera_workspace")):
    """Phase 7: 論文生成"""

@app.command()
def evaluate_paper(work_dir: Path = Path("./sera_workspace")):
    """Phase 8: 論文評価・改善ループ"""

@app.command()
def status(work_dir: Path = Path("./sera_workspace")):
    """現在の探索状態サマリ表示"""

@app.command()
def show_node(node_id: str, work_dir: Path = Path("./sera_workspace")):
    """ノード詳細表示"""

@app.command()
def replay(node_id: str, seed: int, work_dir: Path = Path("./sera_workspace")):
    """特定ノードの実験再実行"""

@app.command()
def validate_specs(work_dir: Path = Path("./sera_workspace")):
    """Spec整合性チェック"""
```

**テスト**: `typer.testing.CliRunner` で各コマンドの呼び出しテスト。`sera init` → `sera validate-specs` の最小フロー。

---

### 22.14 Step 12: 統合テスト + docs

**統合テスト**（`tests/test_integration/`）:
```python
def test_full_pipeline_mock():
    """
    全APIとLLMをモックして、以下のフローが完走することを検証:
    1. sera init（サンプル Input-1）
    2. sera phase0-related-work（モックAPI）
    3. sera freeze-specs --auto
    4. sera research（max_nodes=3, repeats=1 の最小設定、モック実験）
    5. sera export-best
    6. sera generate-paper（モックLLM）
    7. sera evaluate-paper（モックLLM）

    検証項目:
    - specs/ に全9ファイル + .lock が存在
    - logs/ に全 JSONL ファイルが存在し、各1エントリ以上
    - outputs/best/ に best_node.json, report.json が存在
    - paper/paper.md が存在し、空でない
    - exit code が全て 0
    """
```

**docs/**:
- §15 の内容に従い、各 .md ファイルを作成
- quickstart.md は実際にコマンドを実行できるチュートリアル形式
- architecture.md は Mermaid 図を含める

---

### 22.15 Step 13: Phase A — MT-GRPO統合

**作業内容**: ターンレベル報酬システムを実装し、PPOに統合する。

**実装ファイル**:
```text
1. src/sera/learning/turn_reward.py  — Phase毎のターンレベル報酬評価器
2. src/sera/specs/plan_spec.py       — TurnRewardSpec の追加
3. src/sera/learning/reward.py       — compute_reward_v2 の追加
4. src/sera/learning/rollout.py      — PPORolloutV2 の追加
```

**turn_reward.py の実装**:
```python
class TurnRewardEvaluator:
    """各Phaseの出力品質を評価するターンレベル報酬評価器"""

    def __init__(self, turn_reward_spec: TurnRewardSpec):
        self.spec = turn_reward_spec
        self._evaluators = {
            "citation_relevance": self._eval_citation_relevance,
            "hypothesis_novelty": self._eval_hypothesis_novelty,
            "code_executability": self._eval_code_executability,
            "metric_improvement": self._eval_metric_improvement,
            "paper_score_delta": self._eval_paper_score_delta,
        }

    def evaluate_turn(self, phase: str, phase_output: dict) -> float:
        """指定Phaseの出力品質を 0.0〜1.0 で評価"""
        evaluator_name = self.spec.phase_rewards[phase].evaluator
        return self._evaluators[evaluator_name](phase_output)

    def evaluate_all_turns(self, phase_outputs: dict[str, dict]) -> dict[str, float]:
        """全Phaseのターン報酬を一括計算"""
        return {phase: self.evaluate_turn(phase, output)
                for phase, output in phase_outputs.items()
                if phase in self.spec.phase_rewards}

    def _eval_citation_relevance(self, output: dict) -> float:
        """Phase 0: 収集論文の関連性スコア（0.0〜1.0）"""
        pass

    def _eval_hypothesis_novelty(self, output: dict) -> float:
        """Phase 2: 仮説の新規性（既存ノードとの類似度逆数）"""
        pass

    def _eval_code_executability(self, output: dict) -> float:
        """Phase 3: コードが正常実行できたか（binary: 0.0 or 1.0）"""
        return 1.0 if output.get("exit_code") == 0 else 0.0

    def _eval_metric_improvement(self, output: dict) -> float:
        """Phase 4: 親ノード比のメトリクス改善率"""
        pass

    def _eval_paper_score_delta(self, output: dict) -> float:
        """Phase 7: PaperScore改善幅"""
        pass
```

**PlanSpec への TurnRewardSpec 統合**:
```python
# plan_spec.py に追加
class PhaseRewardConfig(BaseModel):
    evaluator: str
    weight: float

class TurnRewardSpec(BaseModel):
    enabled: bool = True
    phase_rewards: dict[str, PhaseRewardConfig] = {
        "phase0": PhaseRewardConfig(evaluator="citation_relevance", weight=0.1),
        "phase2": PhaseRewardConfig(evaluator="hypothesis_novelty", weight=0.15),
        "phase3": PhaseRewardConfig(evaluator="code_executability", weight=0.25),
        "phase4": PhaseRewardConfig(evaluator="metric_improvement", weight=0.35),
        "phase7": PhaseRewardConfig(evaluator="paper_score_delta", weight=0.15),
    }

class PlanSpecModel(BaseModel):
    # ... 既存フィールド ...
    turn_rewards: TurnRewardSpec = TurnRewardSpec()
```

**テスト**: `tests/test_learning/test_turn_reward.py`
- 各評価器が 0.0〜1.0 の範囲で値を返すこと
- compute_reward_v2 がターン報酬の重み付き和を正しく計算すること
- turn_rewards.enabled=False の場合に compute_reward にフォールバックすること

**完了条件**:
- TurnRewardEvaluator の5つの評価器が実装済み
- PPORolloutV2 が turn_rewards を保持
- compute_reward_v2 が正しく計算
- ppo_log.jsonl に turn_rewards が記録される

---

### 22.16 Step 14: Phase B — ECHO軽量版統合

**作業内容**: 失敗ノードからの知識抽出と兄弟ノードへの注入を実装する。

**実装ファイル**:
```text
1. src/sera/search/failure_extractor.py  — FailureKnowledgeExtractor
2. src/sera/search/search_node.py        — failure_context フィールド追加
3. src/sera/search/search_manager.py     — build_context() 拡張
4. src/sera/search/tree_ops.py           — improve プロンプトに failure_context 注入
```

**failure_extractor.py の実装**:
```python
from dataclasses import dataclass

@dataclass
class FailureSummary:
    """失敗ノードから抽出された知識の要約"""
    node_id: str
    hypothesis: str
    error_type: str           # "runtime_error" | "logic_error" | "oom" | "timeout"
    error_message: str
    failure_analysis: str     # LLMが生成した失敗原因分析
    partial_successes: str    # 部分的に成功した箇所の特定
    avoidance_advice: str     # 兄弟ノードが避けるべき事項

class FailureKnowledgeExtractor:
    """ECHO軽量版: 失敗ノードから知識を抽出し兄弟ノードに注入（§26.4.3）"""

    def __init__(self, agent_llm: "AgentLLM"):
        self.agent_llm = agent_llm

    def extract(self, failed_node: "SearchNode") -> FailureSummary:
        """
        失敗ノードの実験結果を要約する。
        - 何を試みたか（hypothesis + experiment_config）
        - なぜ失敗したか（stderr, metrics, exit_code分析）
        - 何が部分的に成功したか（実行可能だった部分の特定）

        注意: 完全なECHOと異なり、目標の再解釈は行わない。
        木構造の兄弟展開が代替手段の探索を担う。
        """
        prompt = FAILURE_ANALYSIS_PROMPT.format(
            hypothesis=failed_node.hypothesis,
            config=failed_node.experiment_config,
            stderr=read_stderr(failed_node),
            metrics=read_metrics(failed_node),
        )
        analysis = self.agent_llm.generate(prompt, purpose="failure_analysis")
        return FailureSummary(
            node_id=failed_node.node_id,
            hypothesis=failed_node.hypothesis,
            error_type=classify_error(failed_node),
            error_message=failed_node.error_message or "",
            failure_analysis=analysis.text,
            partial_successes=extract_partial_successes(analysis.text),
            avoidance_advice=extract_avoidance_advice(analysis.text),
        )

    def inject(self, summary: FailureSummary, sibling_nodes: list["SearchNode"]):
        """
        失敗知識を兄弟ノードのコンテキストに注入する。
        SearchManager.build_context()（§6.8）に追加情報として渡す。
        """
        for node in sibling_nodes:
            if not hasattr(node, 'failure_context'):
                node.failure_context = []
            node.failure_context.append(summary)
```

**SearchNode への failure_context フィールド追加**:
```python
# search_node.py — 追加フィールド
failure_context: list[FailureSummary] = field(default_factory=list)
```

**SearchManager.build_context() 拡張**:
```python
# search_manager.py — improve プロンプト構築時に failure_context を含める
def build_context(self, parent: SearchNode, siblings: list[SearchNode]) -> str:
    context = self._build_sibling_context(parent, siblings)  # 既存
    # 失敗知識コンテキストの追加（§6.8.2）
    if parent.failure_context:
        context += "\n\n## 失敗した兄弟ノード（避けるべきアプローチ）\n"
        for fs in parent.failure_context:
            context += f"- 仮説: {fs.hypothesis}\n"
            context += f"  エラー: {fs.error_message}\n"
            context += f"  分析: {fs.failure_analysis}\n"
            context += f"  回避事項: {fs.avoidance_advice}\n"
    return context
```

**テスト**: `tests/test_search/test_failure_extractor.py`
- extract() がモックLLMで FailureSummary を返すこと
- inject() が兄弟ノードの failure_context に追加すること
- build_context() に failure_context が含まれること
- failure_context が空の場合に既存動作が変わらないこと

**完了条件**:
- FailureKnowledgeExtractor.extract() が失敗ノードから FailureSummary を抽出
- inject() が兄弟ノードのコンテキストに知識を注入
- improve プロンプトに失敗知識が含まれる
- 失敗なしの場合に既存動作に影響しない

---

### 22.17 Step 15: Phase C — HiPER + Tool-Calling

> **実装状況**: ✅ HiPER 3層Advantage分解、Agent Function System（§28）、Tool Execution Engine（§29）は全て実装済み。以下のコード例は初期設計時のもの。実際の実装は `src/sera/agent/` 以下（`agent_functions.py`, `agent_loop.py`, `tool_executor.py`, `tool_policy.py`, `tools/`, `functions/`）を参照。ツール・関数の有効化リストは PlanSpec §5.8 `agent_commands` で Phase 1 に凍結される。

**作業内容**: AgentLLMのtool-calling対応と階層的PPOを実装する。

**実装ファイル**:
```text
1. src/sera/agent/tool_registry.py       — ToolRegistry（ツール定義・実行管理）
2. src/sera/learning/hierarchical_ppo.py — HiPER 3層階層PPO
3. src/sera/agent/agent_llm.py           — generate_with_tools() 実装
```

**tool_registry.py の実装**:
```python
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class ToolDefinition:
    """ツール定義（LLMに渡すスキーマ）"""
    name: str
    description: str
    parameters: dict[str, Any]       # JSON Schema 形式
    handler: Callable                 # ツール実行関数
    phase_availability: list[str]     # 使用可能なPhase（["phase0", "phase2", "phase3"]）

class ToolRegistry:
    """ツール定義の管理と実行"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """ツールを登録"""
        self._tools[tool.name] = tool

    def get_tools_for_phase(self, phase: str) -> list[dict]:
        """指定Phaseで利用可能なツール定義をLLM向けスキーマ形式で返す"""
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
            if phase in t.phase_availability
        ]

    def execute(self, tool_name: str, arguments: dict) -> Any:
        """ツールを実行し結果を返す"""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name].handler(**arguments)

    def register_defaults(self) -> None:
        """SERAの標準ツールを登録"""
        self.register(ToolDefinition(
            name="search_api",
            description="Search Semantic Scholar for relevant papers",
            parameters={"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
            handler=self._search_api_handler,
            phase_availability=["phase0", "phase7"],
        ))
        # 他の標準ツール...
```

**hierarchical_ppo.py の実装**:
```python
class HierarchicalPPOTrainer:
    """
    HiPER（§26.5）の3層階層的PPO。
    - Switch Level: ノード選択方策のアドバンテージ推定
    - High Level: Phase戦略のアドバンテージ推定
    - Low Level: テキスト/ツール出力のアドバンテージ推定

    各層に独立したAdvantage推定を持ち、境界でのBootstrappingにより分散を低減する。
    """

    def __init__(self, specs: "AllSpecs"):
        self.switch_optimizer = ...  # Switch Level パラメータ
        self.high_optimizer = ...    # High Level パラメータ
        self.low_optimizer = ...     # Low Level パラメータ（= AgentLLMのLoRA）

    def update(self, rollouts: list["PPORolloutV2"], agent_llm: "AgentLLM",
               specs: "AllSpecs"):
        """
        3層の階層的更新:
        1. Low Level: テキスト/ツール出力のPPO更新（既存PPOTrainerと同等）
        2. High Level: Phase戦略の更新（ターン報酬ベース）
        3. Switch Level: ノード選択方策の更新（最終メトリクスベース）

        各層の境界でBootstrapping:
        - Low → High: Low Level の価値推定をHigh Levelの報酬に加算
        - High → Switch: High Level の価値推定をSwitch Levelの報酬に加算
        """
        pass
```

**AgentLLM.generate_with_tools() の実装**:
```python
# agent_llm.py — generate_with_tools の具体実装
async def generate_with_tools(self, prompt, available_tools, purpose, ...):
    if self._provider == "local":
        # ツール定義をシステムプロンプトに含め、構造化出力を要求
        tool_prompt = self._format_tool_prompt(prompt, available_tools)
        output = self._generate_local(tool_prompt)
        return self._parse_tool_output(output, purpose)
    elif self._provider == "openai":
        # OpenAI Function Calling API を使用
        response = await self._client.chat.completions.create(
            model=self.model_spec.agent_llm.model_id,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": t} for t in available_tools],
        )
        return self._parse_openai_tool_response(response, purpose)
    elif self._provider == "anthropic":
        # Anthropic Tool Use API を使用
        response = await self._client.messages.create(
            model=self.model_spec.agent_llm.model_id,
            messages=[{"role": "user", "content": prompt}],
            tools=available_tools,
        )
        return self._parse_anthropic_tool_response(response, purpose)
```

**テスト**: `tests/test_agent/test_tool_calling.py`
- ToolRegistry にツールを登録し、Phase毎のフィルタリングが動作すること
- generate_with_tools がモックで GenerationOutput(tool_calls=[...]) を返すこと
- ToolRegistry.execute() がハンドラを正しく呼び出すこと
- HierarchicalPPOTrainer のモックテスト（3層の更新フローが完走すること）

**完了条件**:
- ToolRegistry がツールの登録・取得・実行を管理
- AgentLLM.generate_with_tools() が全3プロバイダで動作
- HierarchicalPPOTrainer の基本フレームワークが実装済み
- E2Eテスト: ツール呼び出し→実行→結果取得のフルパスが動作

---

### 22.18 テスト戦略まとめ

| テスト種別 | 対象 | モック範囲 | テストファイル |
|-----------|------|----------|--------------|
| 単体 | utils/* | なし | tests/test_utils/ |
| 単体 | specs/* | なし | tests/test_specs/ |
| 単体 | evaluation/* | Executor | tests/test_evaluation/ |
| 単体 | lineage/* | なし（CPUテンソル） | tests/test_lineage/ |
| 単体 | search/priority.py | なし | tests/test_search/ |
| 単体 | learning/turn_reward.py | なし | tests/test_learning/test_turn_reward.py |
| 単体 | search/failure_extractor.py | LLM | tests/test_search/test_failure_extractor.py |
| 単体 | agent/tool_registry.py | なし | tests/test_agent/test_tool_registry.py |
| モック統合 | phase0/* | HTTP（respx） + LLM | tests/test_phase0/ |
| モック統合 | search/* | LLM + Executor | tests/test_search/ |
| モック統合 | paper/* | LLM | tests/test_paper/ |
| モック統合 | agent/tool_calling | LLM + ToolRegistry | tests/test_agent/test_tool_calling.py |
| E2E | 全パイプライン | HTTP + LLM + Executor | tests/test_integration/ |
| CLI | cli.py | 全モック | tests/test_cli/ |

**テスト実行コマンド**:
```bash
# 全テスト（GPU不要テストのみ）
pytest -m "not gpu"

# GPU テスト含む
pytest

# カバレッジ
pytest --cov=sera --cov-report=html
```

**conftest.py に追加するマーカー**:
```python
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "slow: slow integration tests")
    config.addinivalue_line("markers", "network: requires network access")
```

---

### 22.19 実装時の注意点・よくある落とし穴

```text
1. trl のバージョン依存:
   - trl の PPOTrainer API は頻繁に変わる。import エラーが出たら trl のバージョンを確認
   - 最悪自前PPOで対処（§9.3 の手順に従う）

2. peft と transformers のバージョン互換:
   - peft が新しすぎると transformers と非互換な場合がある
   - pip install 時にバージョン制約を確認

3. safetensors のキー名:
   - peft が保存するキー名と、本書で定義する delta キー名が異なる可能性
   - materialize 時にキー名マッピングが必要になることがある

4. httpx の async:
   - Phase 0 の API クライアントは async で実装するが、CLI から呼ぶ際は asyncio.run() でラップ
   - テストでは pytest-asyncio を使用

5. heapq は min-heap:
   - Best-First は max-priority なので、priority の負値を heapq に入れる
   - または heapq._heapify_max は非公開APIなので使わない

6. Ctrl+C ハンドリング:
   - signal.signal(signal.SIGINT, handler) で登録
   - handler 内で checkpoint 保存してから sys.exit(20)
   - PPO 更新中の中断は次回 resume で再実行

7. ExecutionSpec 改竄検知:
   - research コマンド開始時に必ず verify_spec_hash() を呼ぶ
   - 不一致なら即座に exit(2)

8. LLM JSON パース失敗:
   - LLM の出力が正しい JSON でないことがある
   - json.loads() の前に ```json ... ``` ブロックを抽出する前処理を入れる
   - 失敗時は最大3回リトライ（temperature += 0.1）

9. メモリ管理:
   - LoRA キャッシュ（cache_max_entries）を超えないよう LRU で管理
   - 大きなモデルは 4bit/8bit 量子化を推奨

10. ファイルパスの一貫性:
    - 全てのパスは work_dir からの相対パスとして管理
    - Path オブジェクトを使い、文字列結合は避ける
```

---

### 22.20 実装チェックリスト

実装完了後、以下を全て確認すること：

```text
=== 基盤（Step 0-4） ===
[ ] pyproject.toml が正しく、pip install -e ".[dev]" が通る
[ ] sera コマンドが実行でき、--help が表示される
[ ] 全 Spec の Pydantic モデルが定義され、YAML ラウンドトリップが通る
[ ] ExecutionSpec のハッシュ固定と検証が動作する

=== Phase 0-1（Step 3-4） ===
[ ] Phase 0: モック API でフォールバック検索が動作し、queries.jsonl にログが記録される
[ ] Phase 1: LLM モックで ProblemSpec, PlanSpec が生成され、freeze-specs が完了する
[ ] Phase 1: PlanSpec に TurnRewardSpec が含まれ、YAML ラウンドトリップが通る

=== Phase 2-4（Step 6-8） ===
[ ] Phase 2: Best-First 探索で子ノードが生成され、優先度でソートされる
[ ] Phase 3: LocalExecutor で実験スクリプトが実行され、metrics.json が出力される
[ ] Phase 4: μ, SE, LCB が正しく計算され、逐次評価で Top-k のみ追加実行される

=== Phase 5-6（Step 9） ===
[ ] Phase 5: PPO 更新で LoRA パラメータのみが更新され、delta が safetensors で保存される
[ ] Phase 6: materialize が正しく累積復元し、squash でスナップショットが生成される
[ ] Phase 6: Pareto 剪定が動作し、保護リストのノードは残る

=== Phase 7-8（Step 10） ===
[ ] Phase 7: PaperComposer が CI 付き図表を含む paper.md を生成する
[ ] Phase 7: 自動引用検索ループが Semantic Scholar API で動作する
[ ] Phase 7: エージェント自己修正サイクルが構文エラー・未使用図等を自己修正する
[ ] Phase 7: VLM が有効な場合、図記述生成・キャプションレビュー・重複図検出が動作する
[ ] Phase 8: PaperEvaluator がマルチエージェントレビューアンサンブル（Few-shot + エージェント自己改善ループ）で採点する
[ ] Phase 8: アンサンブル時にオーケストレーションエージェント（Area Chair）メタレビューが生成される
[ ] Phase 8: 改善ループが Phase 7（内部反省）と Phase 8（外部評価）の二重構造で回る

=== Agent拡張（Step 5, 13-15） ===
[ ] AgentLLM.generate() が GenerationOutput 型を返す
[ ] AgentLLM.generate_with_tools() がツール定義を受け取り構造化出力を返す（Phase C）

=== Phase A: MT-GRPO（Step 13） ===
[ ] TurnRewardEvaluator の5つの評価器が実装済み
[ ] PPORolloutV2 が turn_rewards を保持し、compute_reward_v2 で報酬を計算
[ ] ppo_log.jsonl に turn_rewards フィールドが記録される
[ ] turn_rewards.enabled=False 時に compute_reward にフォールバックする

=== Phase B: ECHO軽量版（Step 14） ===
[ ] FailureKnowledgeExtractor.extract() が失敗ノードから FailureSummary を抽出する
[ ] inject() が兄弟ノードの failure_context に FailureSummary を追加する
[ ] improve プロンプトに失敗知識コンテキストが含まれる（§6.8.2）
[ ] 失敗なしの場合に既存動作に影響しない

=== Phase C: HiPER + Tool-Calling（Step 15） ===
[x] ToolExecutor が18ツールのディスパッチ・ToolPolicy・レート制限を管理する（§29）
[x] AgentFunctionRegistry が19関数を登録し call_function() で統一呼び出し（§28）
[x] AgentLoop が ReAct 型反復ループで allowed_tools を制限しツール実行（§29）
[x] AgentLLM.generate_with_tools() が全3プロバイダ（local/openai/anthropic）で動作する
[x] HierarchicalPPOTrainer の基本フレームワークが実装済み
[x] PlanSpec §5.8 agent_commands でツール・関数の有効化リストが Phase 1 で凍結される
[ ] E2Eテスト: ツール呼び出し→実行→結果取得のフルパスが動作する

=== 共通 ===
[ ] outputs/best/ に best_node.json, adapter.safetensors, report.json が出力される
[ ] logs/ に全 JSONL ファイル（search, eval, ppo, paper, agent_llm, turn_reward）が記録される
[ ] agent_llm_log.jsonl に tool_calls, turn_rewards フィールドが記録される
[ ] docs/ に §15 の全ファイルが存在する
[ ] pytest が全テスト通過する（GPU不要テストのみでも可）
[ ] sera init → sera phase0-related-work → sera freeze-specs --auto → sera research → sera generate-paper → sera evaluate-paper の一連のフローがモック環境で完走する
```

---
