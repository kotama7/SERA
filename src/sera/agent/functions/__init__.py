"""SERA Agent Functions — imports all function modules to trigger registration.

Importing this package ensures that all ``@register_function`` decorators
in the sub-modules execute and populate ``REGISTRY``.
"""

from sera.agent.functions import (  # noqa: F401
    evaluation_functions,
    execution_functions,
    paper_functions,
    phase0_functions,
    search_functions,
    spec_functions,
)
