"""Experiment config validation per section 6.6.1.

Whitelist validation ensures the agent only proposes experiment configs
that use declared manipulated variables with valid types and ranges.
"""

from __future__ import annotations

from typing import Any


def validate_experiment_config(config: dict, problem_spec: Any) -> tuple[bool, list[str]]:
    """Validate an experiment configuration against the problem spec.

    Performs whitelist validation:
    1. Check for unknown keys not in manipulated_variables.
    2. Check type/range for each variable:
       - float: value must be numeric and within [range[0], range[1]]
       - int: value must be int and within [range[0], range[1]]
       - categorical: value must be in choices list

    Parameters
    ----------
    config : dict
        The experiment configuration to validate.
    problem_spec : object
        Problem spec with ``manipulated_variables`` attribute, where each
        variable has ``name``, ``type``, ``range``, and ``choices`` fields.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, error_messages). is_valid is True when error_messages
        is empty.
    """
    errors: list[str] = []

    # Build lookup of allowed variables
    variables = getattr(problem_spec, "manipulated_variables", [])
    var_lookup: dict[str, Any] = {}
    for var in variables:
        # Support both dict and object access
        name = var.name if hasattr(var, "name") else var["name"]
        var_lookup[name] = var

    # 1. Check for unknown keys
    allowed_keys = set(var_lookup.keys())
    for key in config:
        if key not in allowed_keys:
            errors.append(
                f"Unknown config key '{key}' is not in manipulated_variables. Allowed keys: {sorted(allowed_keys)}"
            )

    # 2. Validate type and range for each known key
    for key, value in config.items():
        if key not in var_lookup:
            continue  # already reported as unknown

        var = var_lookup[key]
        var_type = var.type if hasattr(var, "type") else var["type"]
        var_range = var.range if hasattr(var, "range") else var.get("range")
        var_choices = var.choices if hasattr(var, "choices") else var.get("choices")

        if var_type == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"Variable '{key}' expects float, got {type(value).__name__}: {value!r}")
            elif var_range is not None and len(var_range) == 2:
                lo, hi = var_range[0], var_range[1]
                if value < lo or value > hi:
                    errors.append(f"Variable '{key}' value {value} out of range [{lo}, {hi}]")

        elif var_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"Variable '{key}' expects int, got {type(value).__name__}: {value!r}")
            elif var_range is not None and len(var_range) == 2:
                lo, hi = int(var_range[0]), int(var_range[1])
                if value < lo or value > hi:
                    errors.append(f"Variable '{key}' value {value} out of range [{lo}, {hi}]")

        elif var_type == "categorical":
            if var_choices is not None and value not in var_choices:
                errors.append(f"Variable '{key}' value {value!r} not in choices: {var_choices}")

    is_valid = len(errors) == 0
    return is_valid, errors
