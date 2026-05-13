"""Public runtime API for runintgen.

This module provides the user-facing API to compile runtime integrals
from UFL forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ufl

from .form_metadata import FormRuntimeMetadata


@dataclass
class RuntimeKernelInfo:
    """Information about a compiled runtime kernel.

    Attributes:
        name: The C function name for this kernel.
        integral_type: The type of integral ("cell", "exterior_facet", etc.).
        subdomain_id: The subdomain identifier.
        ir_index: Index in FFCx IR for this integral type.
        c_declaration: The C function declaration (header).
        c_definition: The C function definition (implementation).
        tensor_shape: The shape of the output tensor, if known.
        table_info: List of dicts mapping FFCx table references to runtime
            Basix element tabulation slots.
        table_slots: Map from FFCx table name to runtime element table slot.
        scalar_type: NumPy scalar type used for A, w, and c.
        geometry_type: Real NumPy type used for coordinate_dofs, following FFCx.
    """

    name: str
    integral_type: str
    subdomain_id: int
    ir_index: int = 0
    c_declaration: str = ""
    c_definition: str = ""
    tensor_shape: tuple[int, ...] | None = None
    table_info: list[dict[str, Any]] | None = None
    table_slots: dict[str, int] | None = None
    domain: str | None = None
    kernel_id: int = 0
    scalar_type: str | None = None
    geometry_type: str | None = None


@dataclass
class RunintModule:
    """Container for all runtime kernels compiled from a form.

    Attributes:
        kernels: List of RuntimeKernelInfo objects for each runtime integral.
        meta: Additional metadata (e.g. form name, function spaces, etc.).
    form_metadata: Form-level runtime metadata (Plan v2).
        quadrature_provider: Runtime quadrature provider carried by the UFL
            measure when all generated kernels share one provider.
    """

    kernels: list[RuntimeKernelInfo] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    form_metadata: FormRuntimeMetadata | None = None
    quadrature_provider: Any | None = None

    def create_custom_data(
        self,
        quadrature: Any | None = None,
        *,
        is_cut: Any | None = None,
    ) -> Any:
        """Build a Basix-only runtime ``custom_data`` owner for this module."""
        from .basix_runtime import CustomData

        if quadrature is None:
            quadrature = self.quadrature_provider
        if quadrature is None:
            raise ValueError("No runtime quadrature provider was supplied.")
        return CustomData(self.form_metadata, quadrature=quadrature, is_cut=is_cut)


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
        A RunintModule containing the compiled kernels, metadata, and
        form-level runtime metadata (Plan v2).

    Example:
        >>> from runintgen import compile_runtime_integrals
        >>> # ... define form with runtime measure ...
        >>> module = compile_runtime_integrals(form)
        >>> for kernel in module.kernels:
        ...     print(kernel.name)
        >>> # Access form-level metadata
        >>> for elem in module.form_metadata.unique_elements:
        ...     print(f"Element: {elem.role.name}[{elem.index}]")
    """
    # Import here to avoid circular imports
    from .analysis import build_runtime_info
    from .codegeneration.C.integrals import generate_C_runtime_kernels
    from .form_metadata import build_form_runtime_metadata

    options = dict(options or {})
    options["sum_factorization"] = False
    runtime_info = build_runtime_info(form, options)

    # Build form-level metadata (Plan v2)
    form_metadata = build_form_runtime_metadata(runtime_info)

    # Generate kernels with metadata
    kernels = generate_C_runtime_kernels(runtime_info, options, form_metadata)

    providers = [
        group.quadrature_provider
        for group in runtime_info.groups
        if group.quadrature_provider is not None
    ]
    unique_provider_ids = {id(provider) for provider in providers}
    quadrature_provider = providers[0] if len(unique_provider_ids) == 1 else None

    return RunintModule(
        kernels=kernels,
        meta=runtime_info.meta,
        form_metadata=form_metadata,
        quadrature_provider=quadrature_provider,
    )
