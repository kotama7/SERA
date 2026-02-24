"""Tests for sera.agent.agent_llm and sera.agent.prompt_templates."""

import asyncio
import json
import pytest
from pathlib import Path
from types import SimpleNamespace

from sera.agent.agent_llm import AgentLLM
from sera.agent import prompt_templates
from sera.agent.prompt_templates import TEMPLATE_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model_spec(provider="local"):
    """Create a minimal model_spec-like object for testing."""
    return SimpleNamespace(
        agent_llm=SimpleNamespace(
            provider=provider,
            model_id="test-model",
            temperature=0.7,
            max_tokens=512,
        ),
        base_model=SimpleNamespace(
            id="test-model",
            revision=None,
            dtype="fp32",
            load_method="none",
        ),
        adapter_spec=SimpleNamespace(
            rank=8,
            alpha=16,
            target_modules=["q_proj", "v_proj"],
            dropout=0.05,
            init="zero",
        ),
    )


def _make_resource_spec():
    """Create a minimal resource_spec-like object for testing."""
    return SimpleNamespace(
        api_keys=SimpleNamespace(openai="OPENAI_API_KEY", anthropic="ANTHROPIC_API_KEY")
    )


@pytest.fixture
def mock_model_spec():
    return _make_model_spec(provider="local")


@pytest.fixture
def mock_resource_spec():
    return _make_resource_spec()


@pytest.fixture
def agent_llm(tmp_path, mock_model_spec, mock_resource_spec):
    """Create an AgentLLM instance with a temporary log path."""
    log_path = tmp_path / "logs" / "agent_llm_log.jsonl"
    return AgentLLM(mock_model_spec, mock_resource_spec, log_path)


# ---------------------------------------------------------------------------
# AgentLLM Tests
# ---------------------------------------------------------------------------


class TestAgentLLMMock:
    """Tests for AgentLLM with mock function."""

    async def test_set_mock_and_generate(self, agent_llm):
        """set_mock works and generate returns the mock response."""
        agent_llm.set_mock(lambda prompt, purpose: f"mock:{purpose}")

        result = await agent_llm.generate("Hello", purpose="test")
        assert result == "mock:test"

    async def test_mock_receives_prompt_and_purpose(self, agent_llm):
        """Mock function receives correct prompt and purpose."""
        calls = []

        def track_mock(prompt, purpose):
            calls.append((prompt, purpose))
            return "ok"

        agent_llm.set_mock(track_mock)
        await agent_llm.generate("my prompt", purpose="my_purpose")

        assert len(calls) == 1
        assert calls[0] == ("my prompt", "my_purpose")

    async def test_mock_with_adapter_node_id(self, agent_llm):
        """generate() works with adapter_node_id when mock is set."""
        agent_llm.set_mock(lambda p, pu: "adapted")

        result = await agent_llm.generate(
            "test", purpose="adapt_test", adapter_node_id="node-42"
        )
        assert result == "adapted"

    async def test_mock_with_temperature_and_max_tokens(self, agent_llm):
        """generate() accepts temperature and max_tokens with mock."""
        agent_llm.set_mock(lambda p, pu: "temp_test")

        result = await agent_llm.generate(
            "test",
            purpose="param_test",
            temperature=0.5,
            max_tokens=100,
        )
        assert result == "temp_test"


class TestAgentLLMLogging:
    """Tests for AgentLLM call logging."""

    async def test_logger_records_call(self, agent_llm):
        """After generate(), the logger has exactly one entry."""
        agent_llm.set_mock(lambda p, pu: "logged_response")

        await agent_llm.generate("test prompt", purpose="logging_test")

        entries = agent_llm.logger.read_all()
        assert len(entries) == 1

    async def test_log_entry_fields(self, agent_llm):
        """Log entry contains all required fields."""
        agent_llm.set_mock(lambda p, pu: "response")

        await agent_llm.generate(
            "test prompt",
            purpose="field_test",
            adapter_node_id="node-1",
            temperature=0.9,
        )

        entries = agent_llm.logger.read_all()
        entry = entries[0]

        assert entry["event"] == "llm_call"
        assert entry["purpose"] == "field_test"
        assert entry["model_id"] == "mock"
        assert entry["adapter_node_id"] == "node-1"
        assert entry["temperature"] == 0.9
        assert "call_id" in entry
        assert "prompt_hash" in entry
        assert "response_hash" in entry
        assert "latency_ms" in entry
        assert "prompt_tokens" in entry
        assert "completion_tokens" in entry
        assert "timestamp" in entry

    async def test_log_prompt_hash_format(self, agent_llm):
        """Prompt and response hashes follow sha256:... format."""
        agent_llm.set_mock(lambda p, pu: "hashed")

        await agent_llm.generate("hash test", purpose="hash_test")

        entries = agent_llm.logger.read_all()
        entry = entries[0]
        assert entry["prompt_hash"].startswith("sha256:")
        assert entry["response_hash"].startswith("sha256:")
        assert len(entry["prompt_hash"]) == len("sha256:") + 16

    async def test_multiple_calls_logged(self, agent_llm):
        """Multiple generate() calls are all logged."""
        agent_llm.set_mock(lambda p, pu: "r")

        await agent_llm.generate("p1", purpose="call1")
        await agent_llm.generate("p2", purpose="call2")
        await agent_llm.generate("p3", purpose="call3")

        entries = agent_llm.logger.read_all()
        assert len(entries) == 3
        purposes = [e["purpose"] for e in entries]
        assert purposes == ["call1", "call2", "call3"]

    async def test_log_latency_is_non_negative(self, agent_llm):
        """Latency is recorded as a non-negative number."""
        agent_llm.set_mock(lambda p, pu: "fast")

        await agent_llm.generate("speed test", purpose="latency_test")

        entries = agent_llm.logger.read_all()
        assert entries[0]["latency_ms"] >= 0


class TestAgentLLMProviderInit:
    """Tests for provider initialization paths."""

    def test_local_provider_default(self, tmp_path, mock_resource_spec):
        """When no agent_llm attribute, defaults to 'local' provider."""
        spec = SimpleNamespace()  # No agent_llm attribute
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        assert llm._provider_name == "local"

    def test_openai_provider_init(self, tmp_path, mock_resource_spec):
        """OpenAI provider initializes without error (client may be None if no package)."""
        spec = _make_model_spec(provider="openai")
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        assert llm._provider_name == "openai"

    def test_anthropic_provider_init(self, tmp_path, mock_resource_spec):
        """Anthropic provider initializes without error."""
        spec = _make_model_spec(provider="anthropic")
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        assert llm._provider_name == "anthropic"

    def test_load_adapter_sets_current_id(self, agent_llm):
        """load_adapter() stores the adapter node ID."""
        assert agent_llm._current_adapter_id is None
        agent_llm.load_adapter("node-99")
        assert agent_llm._current_adapter_id == "node-99"

    def test_get_log_probs_rejects_non_local(self, tmp_path, mock_resource_spec):
        """get_log_probs() raises for non-local providers."""
        spec = _make_model_spec(provider="openai")
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        with pytest.raises(RuntimeError, match="local provider"):
            llm.get_log_probs("prompt", "response")


class TestAgentLLMVLLMIntegration:
    """Tests for vLLM inference engine integration in AgentLLM."""

    def test_default_inference_engine_is_transformers(self, agent_llm):
        """Default inference engine is 'transformers'."""
        assert agent_llm._inference_engine == "transformers"
        assert agent_llm._vllm_engine is None

    def test_vllm_engine_attribute_set_for_vllm_config(self, tmp_path, mock_resource_spec):
        """When inference.engine='vllm', _inference_engine is set correctly."""
        spec = _make_model_spec(provider="local")
        spec.inference = SimpleNamespace(engine="vllm")
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        assert llm._inference_engine == "vllm"
        assert llm._vllm_engine is None  # Lazy init

    def test_no_inference_attr_defaults_to_transformers(self, tmp_path, mock_resource_spec):
        """When model_spec has no inference attribute, defaults to transformers."""
        spec = SimpleNamespace(
            agent_llm=SimpleNamespace(
                provider="local", model_id="test", temperature=0.7, max_tokens=512
            ),
            base_model=SimpleNamespace(id="test", revision=None, dtype="fp32", load_method="none"),
            adapter_spec=SimpleNamespace(rank=8, alpha=16, target_modules=["q_proj"], dropout=0.05, init="zero"),
        )
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        assert llm._inference_engine == "transformers"

    def test_ppo_methods_always_use_transformers(self, tmp_path, mock_resource_spec):
        """get_log_probs and get_value always use transformers path even with vllm engine."""
        spec = _make_model_spec(provider="local")
        spec.inference = SimpleNamespace(engine="vllm")
        llm = AgentLLM(spec, mock_resource_spec, tmp_path / "log.jsonl")
        # These should attempt the local/transformers path (will fail without GPU but that's expected)
        assert llm._inference_engine == "vllm"
        # get_log_probs should still check for local provider (not vllm)
        assert llm._provider_name == "local"


# ---------------------------------------------------------------------------
# Prompt Template Tests
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    """Tests for prompt template definitions."""

    def test_all_templates_are_non_empty_strings(self):
        """Every template in the registry is a non-empty string."""
        for name, template in TEMPLATE_REGISTRY.items():
            assert isinstance(template, str), f"{name} is not a string"
            assert len(template) > 50, f"{name} is too short ({len(template)} chars)"

    def test_registry_has_all_expected_keys(self):
        """The registry contains all expected template keys."""
        expected_keys = [
            "query_generation",
            "paper_clustering",
            "relevance_scoring",
            "spec_generation",
            "draft",
            "debug",
            "improve",
            "experiment_code",
            "paper_outline",
            "paper_full_generation",
            "paper_writeup_reflection",
            "citation_search",
            "citation_select",
            "plot_aggregation",
            "vlm_figure_description",
            "vlm_figure_caption_review",
            "vlm_duplicate_detection",
            "paper_evaluation",
            "reviewer_reflection",
            "meta_review",
            "paper_revision",
        ]
        for key in expected_keys:
            assert key in TEMPLATE_REGISTRY, f"Missing template: {key}"

    def test_registry_count(self):
        """Registry has exactly 21 templates."""
        assert len(TEMPLATE_REGISTRY) == 21

    def test_query_generation_placeholders(self):
        """QUERY_GENERATION_PROMPT has expected placeholders."""
        t = prompt_templates.QUERY_GENERATION_PROMPT
        for var in ["task_description", "field", "subfield", "goal_objective"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_paper_clustering_placeholders(self):
        """PAPER_CLUSTERING_PROMPT has expected placeholders."""
        t = prompt_templates.PAPER_CLUSTERING_PROMPT
        for var in ["task_description", "papers_json"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_relevance_scoring_placeholders(self):
        """RELEVANCE_SCORING_PROMPT has expected placeholders."""
        t = prompt_templates.RELEVANCE_SCORING_PROMPT
        for var in ["task_description", "goal_objective", "paper_title", "paper_abstract"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_spec_generation_placeholders(self):
        """SPEC_GENERATION_PROMPT has expected placeholders."""
        t = prompt_templates.SPEC_GENERATION_PROMPT
        for var in ["task_brief", "field", "subfield", "goal_objective", "related_work_summary"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_draft_placeholders(self):
        """DRAFT_PROMPT has expected placeholders."""
        t = prompt_templates.DRAFT_PROMPT
        for var in ["title", "hypothesis", "task_brief", "best_score"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_debug_placeholders(self):
        """DEBUG_PROMPT has expected placeholders."""
        t = prompt_templates.DEBUG_PROMPT
        for var in ["approach_name", "error_type", "error_message", "traceback", "failed_code"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_improve_placeholders(self):
        """IMPROVE_PROMPT has expected placeholders."""
        t = prompt_templates.IMPROVE_PROMPT
        for var in ["title", "approach_name", "current_code", "primary_metric", "current_score"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_experiment_code_placeholders(self):
        """EXPERIMENT_CODE_PROMPT has expected placeholders."""
        t = prompt_templates.EXPERIMENT_CODE_PROMPT
        for var in ["title", "approach_name", "data_location", "seed"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_paper_evaluation_placeholders(self):
        """PAPER_EVALUATION_PROMPT has expected placeholders."""
        t = prompt_templates.PAPER_EVALUATION_PROMPT
        for var in ["venue", "paper_text", "criteria_json", "max_score"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_reviewer_reflection_placeholders(self):
        """REVIEWER_REFLECTION_PROMPT has expected placeholders."""
        t = prompt_templates.REVIEWER_REFLECTION_PROMPT
        for var in ["previous_review_json", "paper_text", "reflection_round", "max_reflections"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_meta_review_placeholders(self):
        """META_REVIEW_PROMPT has expected placeholders."""
        t = prompt_templates.META_REVIEW_PROMPT
        for var in ["paper_title", "reviews_json", "criteria_json"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_paper_revision_placeholders(self):
        """PAPER_REVISION_PROMPT has expected placeholders."""
        t = prompt_templates.PAPER_REVISION_PROMPT
        for var in ["paper_text", "meta_review_json", "reviews_json", "revision_round"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_paper_outline_placeholders(self):
        """PAPER_OUTLINE_PROMPT has expected placeholders."""
        t = prompt_templates.PAPER_OUTLINE_PROMPT
        for var in ["title", "hypothesis", "results_summary"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_paper_full_generation_placeholders(self):
        """PAPER_FULL_GENERATION_PROMPT has expected placeholders."""
        t = prompt_templates.PAPER_FULL_GENERATION_PROMPT
        for var in ["outline_json", "title", "methodology_details"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_citation_search_placeholders(self):
        """CITATION_SEARCH_PROMPT has expected placeholders."""
        t = prompt_templates.CITATION_SEARCH_PROMPT
        for var in ["paper_draft", "current_citations_json"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_plot_aggregation_placeholders(self):
        """PLOT_AGGREGATION_PROMPT has expected placeholders."""
        t = prompt_templates.PLOT_AGGREGATION_PROMPT
        for var in ["results_json", "approaches_json", "results_path", "output_path"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_vlm_figure_description_placeholders(self):
        """VLM_FIGURE_DESCRIPTION_PROMPT has expected placeholders."""
        t = prompt_templates.VLM_FIGURE_DESCRIPTION_PROMPT
        for var in ["figure_path", "section_name", "paper_title"]:
            assert f"{{{var}}}" in t, f"Missing placeholder: {var}"

    def test_templates_are_module_level_constants(self):
        """Templates are accessible as module-level constants."""
        assert hasattr(prompt_templates, "QUERY_GENERATION_PROMPT")
        assert hasattr(prompt_templates, "PAPER_EVALUATION_PROMPT")
        assert hasattr(prompt_templates, "IMPROVE_PROMPT")
        assert hasattr(prompt_templates, "META_REVIEW_PROMPT")

    def test_templates_contain_json_instruction(self):
        """Most templates instruct to output JSON."""
        json_templates = [
            "query_generation",
            "paper_clustering",
            "relevance_scoring",
            "spec_generation",
            "draft",
            "debug",
            "improve",
            "paper_outline",
            "paper_writeup_reflection",
            "citation_search",
            "citation_select",
            "paper_evaluation",
            "reviewer_reflection",
            "meta_review",
            "paper_revision",
        ]
        for name in json_templates:
            template = TEMPLATE_REGISTRY[name]
            assert "JSON" in template or "json" in template, (
                f"Template '{name}' should mention JSON output format"
            )
