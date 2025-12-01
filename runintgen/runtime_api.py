"""Public runtime API for runintgen.

This module provides the user-facing API to compile runtime integrals
from UFL forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ufl


@dataclass
class RuntimeKernelInfo:
    """Information about a compiled runtime kernel.

    Attributes:
        name: The C function name for this kernel.
        integral_type: The type of integral ("cell", "exterior_facet", etc.).
        subdomain_id: The subdomain identifier.
        c_declaration: The C function declaration (header).
        c_definition: The C function definition (implementation).
        tensor_shape: The shape of the output tensor, if known.
        table_info: List of dicts describing required FE tables.
    """

    name: str
    integral_type: str
    subdomain_id: int
    c_declaration: str
    c_definition: str
    tensor_shape: tuple[int, ...] | None = None
    table_info: list[dict[str, Any]] | None = None


@dataclass
class RunintModule:
    """Container for all runtime kernels compiled from a form.

    Attributes:
        kernels: List of RuntimeKernelInfo objects for each runtime integral.
        meta: Additional metadata (e.g. form name, function spaces, etc.).
    """

    kernels: list[RuntimeKernelInfo] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def compile_runtime_integrals(
    form: ufl.Form,
    options: dict[str, Any] | None = None,
) -> RunintModule:
    """Compile all runtime integrals in `form` to C kernels.

    Only integrals marked as runtime (via subdomain_data/metadata) are handled.
    Standard integrals are ignored.

    Args:
        form: A UFL Form containing one or more runtime integrals.
        options: Optional dictionary of compilation options, passed to FFCX.

    Returns:
        A RunintModule containing the compiled kernels and metadata.

    Example:
        >>> from runintgen import compile_runtime_integrals, runtime_dx
        >>> # ... define form with runtime_dx ...
        >>> module = compile_runtime_integrals(form)
        >>> for kernel in module.kernels:
        ...     print(kernel.name)
    """
    # Import here to avoid circular imports
    from .analysis import build_runtime_info
    from .codegeneration.C.integrals import generate_C_runtime_kernels

    options = dict(options or {})
    runtime_info = build_runtime_info(form, options)
    kernels = generate_C_runtime_kernels(runtime_info, options)
    return RunintModule(kernels=kernels, meta=runtime_info.meta)
