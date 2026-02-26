"""Tests for Phase 1 SpecBuilder."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sera.phase1.spec_builder import SpecBuilder


@pytest.fixture
def mock_agent_llm():
    """Mock AgentLLM that returns valid JSON responses."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"problem_spec": {"title": "Test"}}')
    llm.call_function = AsyncMock(return_value={"title": "Test"})
    return llm


@pytest.fixture
def sample_input1_dict():
    return {
        "task": {"brief": "Classify iris", "type": "prediction"},
        "goal": {"metric": "accuracy", "direction": "maximize", "objective": "Maximize accuracy"},
        "domain": {"field": "ML"},
    }


class TestSpecBuilderModelSpec:
    def test_build_model_spec_defaults(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_model_spec({})
        assert "base_model" in result
        assert result["base_model"]["id"] == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_build_model_spec_custom_model(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_model_spec({"base_model": "meta-llama/Llama-3.1-8B"})
        assert result["base_model"]["id"] == "meta-llama/Llama-3.1-8B"

    def test_build_model_spec_agent_llm_provider(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_model_spec({"agent_llm": "openai:gpt-4o"})
        assert result["agent_llm"]["provider"] == "openai"
        assert result["agent_llm"]["model_id"] == "gpt-4o"

    def test_build_model_spec_adapter(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_model_spec({"rank": 32, "alpha": 64})
        assert result["adapter_spec"]["rank"] == 32
        assert result["adapter_spec"]["alpha"] == 64


class TestSpecBuilderResourceSpec:
    def test_build_resource_spec_defaults(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_resource_spec({})
        assert result["compute"]["executor_type"] == "local"
        assert result["compute"]["gpu_required"] is True

    def test_build_resource_spec_timeout(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_resource_spec({"timeout": 7200})
        assert result["sandbox"]["experiment_timeout_sec"] == 7200

    def test_build_resource_spec_no_web(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_resource_spec({"no_web": True})
        assert result["network"]["allow_internet"] is False

    def test_build_resource_spec_slurm(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_resource_spec({"executor": "slurm", "gpu_count": 4})
        assert result["compute"]["executor_type"] == "slurm"
        assert result["compute"]["gpu_count"] == 4


class TestSpecBuilderExecutionSpec:
    def test_build_execution_spec_defaults(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_execution_spec({})
        assert result["search"]["max_nodes"] == 100
        assert result["evaluation"]["repeats"] == 3
        assert result["learning"]["lr"] == 1e-4

    def test_build_execution_spec_custom(self):
        builder = SpecBuilder(agent_llm=None)
        result = builder.build_execution_spec({
            "max_nodes": 200,
            "max_depth": 20,
            "branch_factor": 5,
            "lambda_cost": 0.2,
            "beta": 0.1,
            "repeats": 5,
            "lcb_coef": 2.58,
            "lr": 5e-5,
            "clip": 0.3,
        })
        assert result["search"]["max_nodes"] == 200
        assert result["search"]["max_depth"] == 20
        assert result["search"]["branch_factor"] == 5
        assert result["evaluation"]["repeats"] == 5
        assert result["evaluation"]["lcb_coef"] == 2.58
        assert result["learning"]["lr"] == 5e-5
        assert result["learning"]["clip_range"] == 0.3


class TestSpecBuilderProblemSpec:
    async def test_build_problem_spec_fallback(self, sample_input1_dict):
        """When LLM fails, fallback should map input1 fields correctly."""
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="invalid json")
        builder = SpecBuilder(agent_llm=llm)
        result = await builder.build_problem_spec(sample_input1_dict, {})
        assert result["title"] == "Classify iris"
        assert result["objective"]["metric_name"] == "accuracy"
        assert result["objective"]["direction"] == "maximize"
        assert result["objective"]["description"] == "Maximize accuracy"

    async def test_build_problem_spec_with_call_function(self, mock_agent_llm):
        """When call_function is available, it should be used."""
        mock_agent_llm.call_function = AsyncMock(return_value={
            "problem_spec": {
                "title": "LLM Generated",
                "objective": {"description": "test", "metric_name": "f1", "direction": "maximize"},
            }
        })
        builder = SpecBuilder(agent_llm=mock_agent_llm)
        result = await builder.build_problem_spec({"task": {"brief": "Test"}}, {})
        assert result["title"] == "LLM Generated"


class TestSpecBuilderPlanSpec:
    async def test_build_plan_spec_fallback(self):
        """When LLM fails, should return defaults."""
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="invalid")
        builder = SpecBuilder(agent_llm=llm)
        result = await builder.build_plan_spec({}, {})
        assert "search_strategy" in result or "reward" in result
