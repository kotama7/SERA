"""Phase 1: Spec freezing with SHA-256 integrity verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from sera.utils.hashing import compute_spec_hash

logger = logging.getLogger(__name__)


class SpecFreezer:
    """Freeze all specs to disk and lock ExecutionSpec with SHA-256 hash."""

    def freeze(self, specs: Any, specs_dir: Path) -> None:
        """
        1. Save all specs to specs_dir as YAML files.
        2. Compute ExecutionSpec SHA-256 hash.
        3. Write hash to execution_spec.yaml.lock.
        """
        specs_dir.mkdir(parents=True, exist_ok=True)

        spec_mapping = {
            "input1.yaml": "input1",
            "related_work_spec.yaml": "related_work",
            "paper_spec.yaml": "paper",
            "paper_score_spec.yaml": "paper_score",
            "teacher_paper_set.yaml": "teacher_paper_set",
            "problem_spec.yaml": "problem",
            "model_spec.yaml": "model",
            "resource_spec.yaml": "resource",
            "plan_spec.yaml": "plan",
            "execution_spec.yaml": "execution",
        }

        for filename, attr in spec_mapping.items():
            spec = getattr(specs, attr, None)
            if spec is None:
                continue
            data = spec.model_dump() if hasattr(spec, "model_dump") else spec
            path = specs_dir / filename
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Saved {filename}")

        # Auto-populate model metadata
        model_spec = getattr(specs, "model", None)
        if model_spec is not None:
            bm = getattr(model_spec, "base_model", None)
            if bm is not None and not getattr(bm, "revision", None):
                try:
                    from transformers import AutoConfig

                    config = AutoConfig.from_pretrained(bm.id)
                    bm.revision = getattr(config, "_commit_hash", "unknown")
                except Exception:
                    bm.revision = "unknown"

            # Compute adapter_spec_hash
            adapter_data = getattr(model_spec, "adapter_spec", None)
            compat = getattr(model_spec, "compatibility", None)
            if adapter_data is not None and compat is not None:
                try:
                    adapter_dict = adapter_data.model_dump() if hasattr(adapter_data, "model_dump") else adapter_data
                    compat.adapter_spec_hash = compute_spec_hash(adapter_dict)
                except Exception:
                    pass

            # Re-save model_spec with updated metadata
            data = model_spec.model_dump() if hasattr(model_spec, "model_dump") else model_spec
            path = specs_dir / "model_spec.yaml"
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            logger.info("Updated model_spec.yaml with revision and adapter_spec_hash")

        # Validate agent_commands (§5.8.4)
        plan_spec = getattr(specs, "plan", None)
        if plan_spec is not None:
            self._validate_agent_commands(plan_spec)

        # Lock ExecutionSpec
        exec_spec = getattr(specs, "execution", None)
        if exec_spec is not None:
            exec_data = exec_spec.model_dump() if hasattr(exec_spec, "model_dump") else exec_spec
            spec_hash = compute_spec_hash(exec_data)
            lock_path = specs_dir / "execution_spec.yaml.lock"
            with open(lock_path, "w") as f:
                f.write(spec_hash)
            logger.info(f"ExecutionSpec locked: {spec_hash}")

    @staticmethod
    def _validate_agent_commands(plan_spec: Any) -> None:
        """Validate agent_commands consistency (§5.8.4).

        Checks:
        1. function_tool_bindings tools exist in available_tools
        2. function_tool_bindings tools are subset of phase_tool_map for that function's phase
        3. available_functions are registered (soft check - log warning only)
        """
        ac = getattr(plan_spec, "agent_commands", None)
        if ac is None:
            return

        tools_cfg = getattr(ac, "tools", None)
        funcs_cfg = getattr(ac, "functions", None)
        if tools_cfg is None or funcs_cfg is None:
            return

        # Flatten available_tools into a set
        all_tools: set[str] = set()
        available_tools = getattr(tools_cfg, "available_tools", {})
        for cat_tools in available_tools.values():
            all_tools.update(cat_tools)

        # Build phase_tool_map lookup
        phase_tool_map = getattr(tools_cfg, "phase_tool_map", {})

        # Build function→phase mapping
        # Map function category names to actual phase_tool_map keys
        _CATEGORY_TO_PHASE = {
            "search": "phase2",
            "execution": "phase3",
            "spec": "phase1",
            "paper": "phase7",
            "evaluation": "phase8",
            "phase0": "phase0",
        }
        func_to_phase: dict[str, str] = {}
        available_functions = getattr(funcs_cfg, "available_functions", {})
        for category, func_list in available_functions.items():
            phase = _CATEGORY_TO_PHASE.get(category, category)
            for fn in func_list:
                func_to_phase[fn] = phase

        # Validate function_tool_bindings
        bindings = getattr(funcs_cfg, "function_tool_bindings", {})
        for func_name, bound_tools in bindings.items():
            # Check 1: bound tools exist in available_tools
            for tool in bound_tools:
                if tool not in all_tools:
                    logger.warning(
                        "agent_commands validation: function '%s' binds tool '%s' which is not in available_tools",
                        func_name,
                        tool,
                    )

            # Check 2: bound tools are subset of phase_tool_map for function's phase
            phase = func_to_phase.get(func_name)
            if phase:
                phase_tools = set(phase_tool_map.get(phase, []))
                for tool in bound_tools:
                    if tool not in phase_tools:
                        logger.warning(
                            "agent_commands validation: function '%s' (phase=%s) binds "
                            "tool '%s' which is not in phase_tool_map[%s]",
                            func_name,
                            phase,
                            tool,
                            phase,
                        )

        # Check 3: available_functions are registered in AgentFunctionRegistry
        try:
            import sera.agent.functions  # noqa: F401  — trigger registration
            from sera.agent.agent_functions import REGISTRY

            for phase, func_list in available_functions.items():
                for fn in func_list:
                    try:
                        REGISTRY.get(fn)
                    except KeyError:
                        logger.warning(
                            "agent_commands validation: function '%s' (phase=%s) is not "
                            "registered in AgentFunctionRegistry",
                            fn,
                            phase,
                        )
        except ImportError:
            logger.debug("Could not import agent functions for validation")

        logger.info("agent_commands validation completed")

    def verify(self, specs_dir: Path) -> bool:
        """
        Verify ExecutionSpec hash matches lock file.
        Returns True if valid, False if tampered.
        """
        spec_path = specs_dir / "execution_spec.yaml"
        lock_path = specs_dir / "execution_spec.yaml.lock"

        if not spec_path.exists() or not lock_path.exists():
            logger.error("ExecutionSpec or lock file not found")
            return False

        with open(spec_path) as f:
            data = yaml.safe_load(f)

        with open(lock_path) as f:
            stored_hash = f.read().strip()

        computed_hash = compute_spec_hash(data)
        if computed_hash != stored_hash:
            logger.error(f"ExecutionSpec tampered! Computed={computed_hash}, Stored={stored_hash}")
            return False

        logger.info("ExecutionSpec integrity verified")
        return True
