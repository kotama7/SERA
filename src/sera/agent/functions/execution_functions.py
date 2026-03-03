"""Agent Function definitions for Phase 3 experiment execution.

Registers: experiment_code_gen.
"""

from __future__ import annotations

from sera.agent.agent_functions import (
    OutputMode,
    extract_code_block,
    register_function,
)


@register_function(
    name="experiment_code_gen",
    description="Generate a complete experiment script from a hypothesis and config.",
    parameters={
        "type": "object",
        "properties": {
            "language_name": {"type": "string"},
            "problem_description": {"type": "string"},
            "hypothesis": {"type": "string"},
            "experiment_config": {"type": "object"},
        },
        "required": ["language_name", "problem_description", "hypothesis"],
    },
    return_schema=None,
    output_mode=OutputMode.CODE,
    phase="execution",
    default_temperature=0.5,
    max_retries=3,
    allowed_tools=["web_search", "run_shell_command", "read_file", "execute_code_snippet"],
    loop_config={"max_steps": 8, "tool_call_budget": 15, "timeout_sec": 180},
)
def handle_experiment_code_gen(response: str) -> str:
    """Extract code from LLM response.

    Tries python-fenced block first, then generic fence, then raw.
    """
    return extract_code_block(response, "python")
