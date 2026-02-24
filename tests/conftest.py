import pytest
from pathlib import Path


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "slow: slow integration tests")
    config.addinivalue_line("markers", "network: requires network access")


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary sera_workspace directory."""
    dirs = [
        "specs",
        "related_work/results",
        "related_work/teacher_papers",
        "lineage/nodes",
        "runs",
        "logs",
        "checkpoints",
        "outputs/best",
        "paper/figures",
        "docs/modules",
    ]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_input1():
    """Test Input-1 data."""
    return {
        "version": 1,
        "data": {
            "description": "UCI Iris dataset",
            "location": "./data/iris.csv",
            "format": "csv",
            "size_hint": "small(<1GB)",
        },
        "domain": {"field": "ML", "subfield": "classification"},
        "task": {"brief": "Classify iris species", "type": "prediction"},
        "goal": {
            "objective": "maximize accuracy",
            "direction": "maximize",
            "baseline": "0.95",
        },
        "constraints": [{"name": "inference_time_ms", "type": "le", "threshold": 100}],
        "notes": "",
    }


@pytest.fixture
def mock_llm_response():
    """LLM response mock generator."""

    def _mock(content: str):
        return content

    return _mock
