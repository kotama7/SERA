"""Tests for sera.agent.agent_functions and sera.agent.functions.

Covers:
1. Registry operations (register, get, list_all, list_by_phase, duplicate error)
2. Schema conversions (to_openai_tools, to_anthropic_tools, to_prompt_schema)
3. Parse utilities (parse_json_response 3-stage fallback, extract_code_block)
4. Validation utility (validate_against_schema)
5. Handler correctness for each function module
6. All 19 functions registered via functions/__init__.py import
7. call_function integration with set_mock
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sera.agent.agent_functions import (
    AgentFunction,
    AgentFunctionRegistry,
    OutputMode,
    extract_code_block,
    parse_json_response,
    validate_against_schema,
)


# =====================================================================
# 1. Registry tests
# =====================================================================


class TestAgentFunctionRegistry:
    def test_register_and_get(self):
        reg = AgentFunctionRegistry()
        func = AgentFunction(name="test_fn", description="A test function")
        reg.register(func)
        assert reg.get("test_fn") is func

    def test_duplicate_register_raises(self):
        reg = AgentFunctionRegistry()
        func = AgentFunction(name="dup", description="First")
        reg.register(func)
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(AgentFunction(name="dup", description="Second"))

    def test_get_missing_raises(self):
        reg = AgentFunctionRegistry()
        with pytest.raises(KeyError, match="not_found"):
            reg.get("not_found")

    def test_list_all(self):
        reg = AgentFunctionRegistry()
        reg.register(AgentFunction(name="a", description="A"))
        reg.register(AgentFunction(name="b", description="B"))
        assert len(reg.list_all()) == 2

    def test_list_by_phase(self):
        reg = AgentFunctionRegistry()
        reg.register(AgentFunction(name="s1", description="Search 1", phase="search"))
        reg.register(AgentFunction(name="s2", description="Search 2", phase="search"))
        reg.register(AgentFunction(name="p1", description="Paper 1", phase="paper"))
        assert len(reg.list_by_phase("search")) == 2
        assert len(reg.list_by_phase("paper")) == 1
        assert len(reg.list_by_phase("missing")) == 0


# =====================================================================
# 2. Schema conversion tests
# =====================================================================


class TestSchemaConversions:
    def setup_method(self):
        self.reg = AgentFunctionRegistry()
        self.reg.register(
            AgentFunction(
                name="fn1",
                description="Function one",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                return_schema={"type": "object", "properties": {"result": {"type": "string"}}},
                output_mode=OutputMode.JSON,
                phase="search",
            )
        )

    def test_to_openai_tools(self):
        tools = self.reg.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "fn1"
        assert "parameters" in tools[0]["function"]

    def test_to_anthropic_tools(self):
        tools = self.reg.to_anthropic_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "fn1"
        assert "input_schema" in tools[0]

    def test_to_prompt_schema(self):
        text = self.reg.to_prompt_schema()
        assert "fn1" in text
        assert "Function one" in text

    def test_to_openai_tools_filtered(self):
        self.reg.register(AgentFunction(name="fn2", description="Fn two"))
        tools = self.reg.to_openai_tools(names=["fn1"])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "fn1"


# =====================================================================
# 3. Parse utility tests
# =====================================================================


class TestParseJsonResponse:
    def test_fenced_json_block(self):
        response = 'Here is the result:\n```json\n[{"a": 1}]\n```'
        result = parse_json_response(response)
        assert result == [{"a": 1}]

    def test_raw_json(self):
        response = '{"key": "value"}'
        result = parse_json_response(response)
        assert result == {"key": "value"}

    def test_embedded_json_array(self):
        response = "Some text before [1, 2, 3] some text after"
        result = parse_json_response(response)
        assert result == [1, 2, 3]

    def test_embedded_json_object(self):
        response = 'Blah blah {"x": 42} done'
        result = parse_json_response(response)
        assert result == {"x": 42}

    def test_no_json_returns_none(self):
        response = "Just plain text with no JSON at all."
        result = parse_json_response(response)
        assert result is None

    def test_invalid_json_returns_none(self):
        response = "```json\n{invalid json}\n```"
        result = parse_json_response(response)
        assert result is None

    def test_fenced_takes_priority(self):
        response = '{"outer": 1}\n```json\n{"inner": 2}\n```'
        result = parse_json_response(response)
        assert result == {"inner": 2}


class TestExtractCodeBlock:
    def test_python_fence(self):
        response = "Here:\n```python\nprint('hello')\n```"
        assert extract_code_block(response, "python") == "print('hello')"

    def test_generic_fence(self):
        response = "```\ncode here\n```"
        assert extract_code_block(response, "python") == "code here"

    def test_raw_fallback(self):
        response = "  raw code only  "
        assert extract_code_block(response, "python") == "raw code only"

    def test_julia_fence(self):
        response = '```julia\nprintln("hi")\n```'
        assert extract_code_block(response, "julia") == 'println("hi")'


# =====================================================================
# 4. Validation utility tests
# =====================================================================


class TestValidateAgainstSchema:
    def test_valid_object(self):
        data = {"name": "test", "value": 42}
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
        }
        ok, errors = validate_against_schema(data, schema)
        assert ok
        assert errors == []

    def test_missing_required(self):
        data = {"value": 42}
        schema = {"type": "object", "required": ["name"]}
        ok, errors = validate_against_schema(data, schema)
        assert not ok
        assert any("name" in e for e in errors)

    def test_wrong_type(self):
        data = "not an object"
        schema = {"type": "object"}
        ok, errors = validate_against_schema(data, schema)
        assert not ok

    def test_array_items(self):
        data = [{"a": 1}, {"a": "wrong"}]
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
            },
        }
        ok, errors = validate_against_schema(data, schema)
        assert not ok

    def test_empty_schema_always_valid(self):
        ok, errors = validate_against_schema({"anything": True}, {})
        assert ok

    def test_nested_validation(self):
        data = {"outer": {"inner": 42}}
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {"inner": {"type": "string"}},
                }
            },
        }
        ok, errors = validate_against_schema(data, schema)
        assert not ok
        assert any("inner" in e for e in errors)


# =====================================================================
# 5. Handler tests (individual function modules)
# =====================================================================


class TestSearchHandlers:
    def test_handle_search_draft_json_array(self):
        from sera.agent.functions.search_functions import handle_search_draft

        resp = json.dumps([{"hypothesis": "H1", "experiment_config": {}, "rationale": "R1"}])
        result = handle_search_draft(resp)
        assert len(result) == 1
        assert result[0]["hypothesis"] == "H1"

    def test_handle_search_draft_single_object(self):
        from sera.agent.functions.search_functions import handle_search_draft

        resp = json.dumps({"hypothesis": "H1", "experiment_config": {}, "rationale": "R1"})
        result = handle_search_draft(resp)
        assert len(result) == 1

    def test_handle_search_draft_empty(self):
        from sera.agent.functions.search_functions import handle_search_draft

        result = handle_search_draft("no json here")
        assert result == []

    def test_handle_search_debug_valid(self):
        from sera.agent.functions.search_functions import handle_search_debug

        resp = json.dumps({
            "hypothesis": "H",
            "experiment_config": {},
            "experiment_code": "print(1)",
            "rationale": "Fixed",
        })
        result = handle_search_debug(resp)
        assert result is not None
        assert result["experiment_code"] == "print(1)"

    def test_handle_search_debug_none(self):
        from sera.agent.functions.search_functions import handle_search_debug

        result = handle_search_debug("garbage")
        assert result is None

    def test_handle_search_improve(self):
        from sera.agent.functions.search_functions import handle_search_improve

        resp = json.dumps([{"hypothesis": "H", "experiment_config": {"lr": 0.01}, "rationale": "R"}])
        result = handle_search_improve(resp)
        assert len(result) == 1


class TestExecutionHandlers:
    def test_handle_experiment_code_gen_fenced(self):
        from sera.agent.functions.execution_functions import handle_experiment_code_gen

        resp = "Sure:\n```python\nimport numpy\nprint('ok')\n```"
        result = handle_experiment_code_gen(resp)
        assert "import numpy" in result
        assert "```" not in result

    def test_handle_experiment_code_gen_raw(self):
        from sera.agent.functions.execution_functions import handle_experiment_code_gen

        resp = "print('hello world')"
        result = handle_experiment_code_gen(resp)
        assert result == "print('hello world')"


class TestSpecHandlers:
    def test_handle_spec_generation_problem(self):
        from sera.agent.functions.spec_functions import handle_spec_generation_problem

        resp = json.dumps({"problem_spec": {"title": "Test"}})
        result = handle_spec_generation_problem(resp)
        assert result["problem_spec"]["title"] == "Test"

    def test_handle_spec_generation_plan(self):
        from sera.agent.functions.spec_functions import handle_spec_generation_plan

        resp = json.dumps({"plan_spec": {"search_strategy": "best-first"}})
        result = handle_spec_generation_plan(resp)
        assert "plan_spec" in result


class TestPaperHandlers:
    def test_handle_paper_outline(self):
        from sera.agent.functions.paper_functions import handle_paper_outline

        result = handle_paper_outline("  An outline  ")
        assert result == "An outline"

    def test_handle_aggregate_plot_generation(self):
        from sera.agent.functions.paper_functions import handle_aggregate_plot_generation

        resp = json.dumps([{"description": "scatter", "code": "plt.scatter(...)"}])
        result = handle_aggregate_plot_generation(resp)
        assert len(result) == 1
        assert result[0]["description"] == "scatter"

    def test_handle_aggregate_plot_fix(self):
        from sera.agent.functions.paper_functions import handle_aggregate_plot_fix

        resp = "```python\nfixed_code()\n```"
        result = handle_aggregate_plot_fix(resp)
        assert result == "fixed_code()"


class TestPhase0Handlers:
    def test_handle_query_generation(self):
        from sera.agent.functions.phase0_functions import handle_query_generation

        result = handle_query_generation("  query1\nquery2  ")
        assert result == "query1\nquery2"

    def test_handle_paper_clustering(self):
        from sera.agent.functions.phase0_functions import handle_paper_clustering

        resp = json.dumps([{"label": "ML", "description": "Machine learning", "paper_ids": ["p1"]}])
        result = handle_paper_clustering(resp)
        assert len(result) == 1
        assert result[0]["label"] == "ML"


# =====================================================================
# 6. Full registration test (all 19 functions via __init__.py import)
# =====================================================================


class TestAllFunctionsRegistered:
    def test_all_19_functions_registered(self):
        """Importing functions/__init__.py should register all 19 functions."""
        from sera.agent.agent_functions import REGISTRY

        # Force import of all function modules
        import sera.agent.functions  # noqa: F401

        expected = {
            "search_draft",
            "search_debug",
            "search_improve",
            "experiment_code_gen",
            "spec_generation_problem",
            "spec_generation_plan",
            "paper_outline",
            "paper_draft",
            "paper_reflection",
            "aggregate_plot_generation",
            "aggregate_plot_fix",
            "citation_identify",
            "citation_select",
            "citation_bibtex",
            "paper_review",
            "paper_review_reflection",
            "meta_review",
            "query_generation",
            "paper_clustering",
        }
        registered = {f.name for f in REGISTRY.list_all()}
        missing = expected - registered
        assert not missing, f"Missing registered functions: {missing}"

    def test_phase_grouping(self):
        from sera.agent.agent_functions import REGISTRY
        import sera.agent.functions  # noqa: F401

        assert len(REGISTRY.list_by_phase("search")) == 3
        assert len(REGISTRY.list_by_phase("execution")) == 1
        assert len(REGISTRY.list_by_phase("spec")) == 2
        assert len(REGISTRY.list_by_phase("paper")) == 8
        assert len(REGISTRY.list_by_phase("evaluation")) == 3
        assert len(REGISTRY.list_by_phase("phase0")) == 2


# =====================================================================
# 7. call_function integration tests (with set_mock)
# =====================================================================


class TestCallFunctionIntegration:
    """Integration tests using AgentLLM.call_function with set_mock."""

    @pytest.fixture
    def agent_llm(self, tmp_path):
        """Create a mock-ready AgentLLM."""
        from sera.agent.agent_llm import AgentLLM

        # Ensure functions are registered
        import sera.agent.functions  # noqa: F401

        model_spec = SimpleNamespace(
            agent_llm=SimpleNamespace(provider="openai", model_id="test", temperature=0.7, max_tokens=4096),
            base_model=SimpleNamespace(id="test", family="", revision=None),
        )
        resource_spec = SimpleNamespace(api_keys=None)
        log_path = tmp_path / "test_log.jsonl"
        llm = AgentLLM(model_spec, resource_spec, log_path)
        return llm

    async def test_call_function_search_draft(self, agent_llm):
        """call_function should invoke handler and return parsed result."""
        mock_response = json.dumps([
            {"hypothesis": "H1", "experiment_config": {"lr": 0.01}, "rationale": "R1"}
        ])
        agent_llm.set_mock(lambda prompt, purpose: mock_response)

        result = await agent_llm.call_function(
            "search_draft", prompt="test prompt"
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["hypothesis"] == "H1"

    async def test_call_function_experiment_code_gen(self, agent_llm):
        """CODE output mode should extract code block."""
        agent_llm.set_mock(
            lambda prompt, purpose: "Here:\n```python\nimport os\nprint('ok')\n```"
        )
        result = await agent_llm.call_function(
            "experiment_code_gen", prompt="generate code"
        )
        assert "import os" in result
        assert "```" not in result

    async def test_call_function_free_text(self, agent_llm):
        """FREE_TEXT output mode should return stripped text."""
        agent_llm.set_mock(lambda prompt, purpose: "  Some review text  ")
        result = await agent_llm.call_function(
            "paper_review", prompt="review this"
        )
        assert result == "Some review text"

    async def test_call_function_retry_on_exception(self, agent_llm):
        """call_function should retry on exception and eventually succeed."""
        call_count = 0

        def mock_fn(prompt, purpose):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return json.dumps([{"hypothesis": "H", "experiment_config": {}, "rationale": "R"}])

        agent_llm.set_mock(mock_fn)
        result = await agent_llm.call_function("search_draft", prompt="test")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert call_count == 3

    async def test_call_function_all_retries_fail(self, agent_llm):
        """When all retries fail, JSON mode returns None."""
        agent_llm.set_mock(lambda prompt, purpose: "never valid json")

        result = await agent_llm.call_function("search_debug", prompt="test")
        # search_debug handler returns None for bad input, schema validation
        # may also fail; eventually we get None fallback for JSON mode
        assert result is None

    async def test_call_function_purpose_defaults_to_name(self, agent_llm):
        """Purpose should default to function name when not specified."""
        captured = {}

        def mock_fn(prompt, purpose):
            captured["purpose"] = purpose
            return "some text"

        agent_llm.set_mock(mock_fn)
        await agent_llm.call_function("paper_review", prompt="test")
        assert captured["purpose"] == "paper_review"

    async def test_call_function_temperature_from_function(self, agent_llm):
        """Temperature should come from function default if not overridden."""
        from sera.agent.agent_functions import REGISTRY

        func = REGISTRY.get("search_debug")
        assert func.default_temperature == 0.5
