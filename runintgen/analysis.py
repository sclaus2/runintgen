"""Analysis module for runintgen.

This module handles the detection and analysis of runtime integrals
in UFL forms, and delegates to FFCX for IR computation.

Key types:
- ElementInfo: Basix-related information for one element in a runtime integral.
- ArgumentRole: Enum for TEST, TRIAL, COEFFICIENT, GEOMETRY.
- ArgumentInfo: Usage of one logical argument in a runtime integral.
- RuntimeIntegralInfo: Collected analysis info for one runtime integral.
- RuntimeAnalysisInfo: Global result of runtime analysis for a form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import ufl
from ffcx.analysis import UFLData
from ffcx.ir.elementtables import get_modified_terminal_element
from ffcx.ir.representation import compute_ir
from ffcx.options import get_options

from .measures import is_runtime_integral

# Type aliases
DerivTuple = tuple[int, ...]  # e.g. (1, 0) in 2D
IntegralKey = tuple[str, int]  # (integral_type, ir_index)
Key = tuple[Any, str, tuple[Any, ...]]  # (domain, integral_type, subdomain_ids)
ProviderKey = tuple[Any, str, Any]  # (domain, integral_type, subdomain_id)


# -----------------------------------------------------------------------------
# Core Data Structures
# -----------------------------------------------------------------------------


class ArgumentRole(Enum):
    """Role of an argument in the form."""

    TEST = auto()
    TRIAL = auto()
    COEFFICIENT = auto()
    GEOMETRY = auto()


@dataclass
class ElementInfo:
    """Basix-related information for one element in a runtime integral."""

    element: Any
    element_id: str
    max_derivative_order: int = 0
    derivative_tuples: set[DerivTuple] = field(default_factory=set)

    def register_derivative(self, deriv: DerivTuple) -> None:
        """Register that this derivative tuple is needed."""
        self.derivative_tuples.add(deriv)
        self.max_derivative_order = max(self.max_derivative_order, sum(deriv))


@dataclass
class ArgumentInfo:
    """Usage of one logical argument in a runtime integral."""

    role: ArgumentRole
    index: int  # argument index, coefficient index, or 0 for geometry
    element_id: str
    element: Any
    max_derivative_order: int = 0
    derivative_tuples: set[DerivTuple] = field(default_factory=set)

    def register_derivative(self, deriv: DerivTuple) -> None:
        """Register that this derivative tuple is needed."""
        self.derivative_tuples.add(deriv)
        self.max_derivative_order = max(self.max_derivative_order, sum(deriv))


@dataclass
class RuntimeIntegralInfo:
    """Collected analysis info for one runtime integral in the FFCx IR."""

    integral_type: str  # "cell", "exterior_facet", ...
    ir_index: int  # index in DataIR.integrals for this type
    subdomain_id: int

    # (role, index) -> ArgumentInfo
    arguments: dict[tuple[ArgumentRole, int], ArgumentInfo] = field(
        default_factory=dict
    )

    # element_id -> ElementInfo
    elements: dict[str, ElementInfo] = field(default_factory=dict)

    def get_or_add_argument(
        self,
        role: ArgumentRole,
        index: int,
        element: Any,
        element_id: str,
    ) -> ArgumentInfo:
        """Get existing or create new ArgumentInfo."""
        key = (role, index)
        if key not in self.arguments:
            self.arguments[key] = ArgumentInfo(
                role=role,
                index=index,
                element=element,
                element_id=element_id,
            )
        return self.arguments[key]

    def get_or_add_element(
        self,
        element: Any,
        element_id: str,
    ) -> ElementInfo:
        """Get existing or create new ElementInfo."""
        if element_id not in self.elements:
            self.elements[element_id] = ElementInfo(
                element=element,
                element_id=element_id,
            )
        return self.elements[element_id]


@dataclass(frozen=True)
class RuntimeGroup:
    """A group of integrals sharing runtime quadrature.

    Attributes:
        domain: The UFL mesh domain.
        integral_type: The type of integral ("cell", "exterior_facet", etc.).
        subdomain_ids: Tuple of subdomain identifiers.
        quadrature_provider: The quadrature provider object (e.g., C++ class).
    """

    domain: Any  # ufl.Mesh - use Any for hashability
    integral_type: str
    subdomain_ids: tuple[Any, ...]
    quadrature_provider: Any = None


@dataclass
class RuntimeAnalysisInfo:
    """Global result of runtime analysis for a form.

    This is the main output of the analysis phase, containing everything
    needed for code generation.
    """

    ir: Any  # FFCx DataIR
    groups: list[RuntimeGroup]
    integral_infos: dict[IntegralKey, RuntimeIntegralInfo]
    form_data: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _get_element_id(element: Any) -> str:
    """Get a stable identifier for an element."""
    return str(hash(element))


def _compute_unscaled_ffcx_form_data(
    form: ufl.Form,
    scalar_type: np.dtype,
) -> Any:
    """Compute FFCx-compatible form data without measure scaling."""
    complex_mode = np.issubdtype(scalar_type, np.complexfloating)
    form_data = ufl.algorithms.compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=False,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(ufl.classes.Jacobian,),
        do_apply_restrictions=True,
        do_append_everywhere_integrals=False,
        complex_mode=complex_mode,
    )

    for integral_data in form_data.integral_data:
        for i, integral in enumerate(integral_data.integrals):
            metadata = dict(integral.metadata() or {})
            if metadata.get("quadrature_rule") == "runtime":
                metadata["quadrature_rule"] = "default"
            elif "quadrature_rule" not in metadata:
                metadata["quadrature_rule"] = "default"
            if (
                "quadrature_degree" not in metadata
                or metadata["quadrature_degree"] < 0
            ):
                qd = metadata.get("estimated_polynomial_degree", 0)
                if isinstance(qd, (tuple, list)):
                    qd = max(qd) if qd else 0
                metadata["quadrature_degree"] = int(qd)
            integral_data.integrals[i] = integral.reconstruct(metadata=metadata)

    return form_data


def _build_ffcx_analysis(form_data: Any) -> UFLData:
    """Build the FFCx analysis wrapper from already processed form data."""
    elements = list(form_data.unique_sub_elements)
    coordinate_elements = list(form_data.coordinate_elements)
    elements += ufl.algorithms.analysis.extract_sub_elements(elements)

    unique_elements = ufl.algorithms.sort_elements(set(elements))
    unique_coordinate_elements = sorted(set(coordinate_elements), key=lambda x: repr(x))
    element_numbers = {element: i for i, element in enumerate(unique_elements)}

    return UFLData(
        form_data=(form_data,),
        unique_elements=unique_elements,
        element_numbers=element_numbers,
        unique_coordinate_elements=unique_coordinate_elements,
        expressions=[],
    )


def _normalised_subdomain_ids(subdomain_id: Any) -> tuple[Any, ...]:
    """Return subdomain ids in the form used by FFCx form_data."""
    if isinstance(subdomain_id, (tuple, list)):
        subdomain_ids = tuple(subdomain_id)
    else:
        subdomain_ids = (subdomain_id,)
    return tuple(
        "otherwise" if sid == "everywhere" else sid for sid in subdomain_ids
    )


def _lookup_runtime_provider(
    domain: Any,
    integral_type: str,
    subdomain_id: Any,
    original_providers: dict[ProviderKey, Any],
    original_providers_by_id: dict[tuple[str, Any], Any],
) -> tuple[bool, Any]:
    """Look up runtime-integral presence and provider for an integral group."""
    provider_key = (domain, integral_type, subdomain_id)
    if provider_key in original_providers:
        return True, original_providers[provider_key]

    fallback_key = (integral_type, subdomain_id)
    if fallback_key in original_providers_by_id:
        return True, original_providers_by_id[fallback_key]

    return False, None


def _build_runtime_groups_and_map(
    form: ufl.Form,
    form_data: Any,
    ir: Any,
) -> tuple[list[RuntimeGroup], dict[IntegralKey, RuntimeGroup]]:
    """Build RuntimeGroup objects and a mapping to IR indices.

    Returns:
        Tuple of (groups list, mapping from (integral_type, ir_index) to RuntimeGroup)
    """
    # Extract quadrature providers from the ORIGINAL form before processing
    original_providers: dict[ProviderKey, Any] = {}
    original_providers_by_id: dict[tuple[str, Any], Any] = {}
    for integral in form.integrals():
        if is_runtime_integral(integral):
            for sid in _normalised_subdomain_ids(integral.subdomain_id()):
                provider_key = (integral.ufl_domain(), integral.integral_type(), sid)
                original_providers[provider_key] = integral.subdomain_data()
                original_providers_by_id[(integral.integral_type(), sid)] = (
                    integral.subdomain_data()
                )

    # Scan integral_data for runtime integrals
    groups_dict: dict[Key, RuntimeGroup] = {}

    for itg_data in form_data.integral_data:
        domain = itg_data.domain
        itype = itg_data.integral_type
        subdomain_ids = tuple(itg_data.subdomain_id)
        key: Key = (domain, itype, subdomain_ids)

        quadrature_provider: Any = None
        runtime_integrals = [
            integral for integral in itg_data.integrals if is_runtime_integral(integral)
        ]

        for integral in runtime_integrals:
            sd = integral.subdomain_data()
            if sd is not None:
                quadrature_provider = sd
                break

        is_runtime_group = bool(runtime_integrals)
        if quadrature_provider is None:
            for sid in subdomain_ids:
                found, provider = _lookup_runtime_provider(
                    itg_data.domain,
                    itype,
                    sid,
                    original_providers,
                    original_providers_by_id,
                )
                is_runtime_group = is_runtime_group or found
                if provider is not None:
                    quadrature_provider = provider
                    break

        if is_runtime_group:
            groups_dict[key] = RuntimeGroup(
                domain=domain,
                integral_type=itype,
                subdomain_ids=subdomain_ids,
                quadrature_provider=quadrature_provider,
            )

    groups = list(groups_dict.values())

    # FFCx creates one IntegralIR per form_data.integral_data entry, with an
    # index counted per integral type. Preserve that ordering so standard and
    # runtime integrals of the same type can coexist without being conflated.
    ir_to_group: dict[IntegralKey, RuntimeGroup] = {}
    ir_type_counts: dict[str, int] = {}
    for itg_data in form_data.integral_data:
        itype = itg_data.integral_type
        idx = ir_type_counts.get(itype, 0)
        ir_type_counts[itype] = idx + 1

        key: Key = (itg_data.domain, itype, tuple(itg_data.subdomain_id))
        if key in groups_dict:
            ir_to_group[(itype, idx)] = groups_dict[key]

    return groups, ir_to_group


def _analyse_single_runtime_integral(
    integral_ir: Any,
    integral_type: str,
    ir_index: int,
    form_data: Any,
    subdomain_id: int,
) -> RuntimeIntegralInfo:
    """Extract ArgumentInfo + ElementInfo for one runtime integral."""

    info = RuntimeIntegralInfo(
        integral_type=integral_type,
        ir_index=ir_index,
        subdomain_id=subdomain_id,
    )

    expr_ir = integral_ir.expression
    coefficient_numbering = expr_ir.coefficient_numbering

    # Scan all integrand data
    for (cell_type, qrule), integrand_data in expr_ir.integrand.items():
        # 1. Extract from modified_arguments
        modified_arguments = integrand_data.get("modified_arguments", [])
        for mt in modified_arguments:
            _process_modified_terminal(mt, coefficient_numbering, info)

        # 2. Extract from factorization graph
        factorization = integrand_data.get("factorization")
        if factorization and hasattr(factorization, "nodes"):
            for node_id, node_data in factorization.nodes.items():
                mt = node_data.get("mt")
                if mt is not None:
                    _process_modified_terminal(mt, coefficient_numbering, info)

    return info


def _process_modified_terminal(
    mt: Any,
    coefficient_numbering: dict,
    info: RuntimeIntegralInfo,
) -> None:
    """Process a ModifiedTerminal and register element/derivative usage."""
    mte = get_modified_terminal_element(mt)
    if mte is None:
        return

    element, averaged, local_derivatives, flat_component = mte
    terminal = mt.terminal

    # Determine role and index
    if isinstance(terminal, ufl.Argument):
        if terminal.number() == 0:
            role = ArgumentRole.TEST
        else:
            role = ArgumentRole.TRIAL
        index = terminal.number()
    elif isinstance(terminal, ufl.Coefficient):
        role = ArgumentRole.COEFFICIENT
        index = coefficient_numbering.get(terminal, -1)
    elif isinstance(terminal, (ufl.classes.SpatialCoordinate, ufl.classes.Jacobian)):
        role = ArgumentRole.GEOMETRY
        index = 0
    else:
        return

    element_id = _get_element_id(element)
    deriv = tuple(local_derivatives)

    # Register in both argument and element
    arg_info = info.get_or_add_argument(role, index, element, element_id)
    elem_info = info.get_or_add_element(element, element_id)

    arg_info.register_derivative(deriv)
    elem_info.register_derivative(deriv)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def build_runtime_analysis(
    form: ufl.Form,
    options: dict[str, Any] | None = None,
) -> RuntimeAnalysisInfo:
    """Analyse a UFL form and collect info for runtime integrals.

    This is the main entry point for the analysis phase. It:
    1. Computes UFL form_data with integral scaling disabled
    2. Runs FFCX analysis and IR computation
    3. Builds runtime groups and maps IR integrals to groups
    4. For each runtime integral, extracts element and derivative info

    Args:
        form: A UFL Form to analyze.
        options: Compilation options dictionary.

    Returns:
        RuntimeAnalysisInfo containing IR, runtime groups, and integral analysis.
    """
    options = dict(options or {})
    options["sum_factorization"] = False
    scalar_type = np.dtype(options.get("scalar_type", "float64"))

    # Compute the single form_data object used by runintgen and FFCx IR. It is
    # unscaled so runtime weights remain the only measure scaling source.
    form_data = _compute_unscaled_ffcx_form_data(form, scalar_type)
    analysis = _build_ffcx_analysis(form_data)

    ffcx_options = get_options(options)
    ffcx_options["sum_factorization"] = False
    ir = compute_ir(analysis, {}, "runint", ffcx_options, visualise=False)

    # 3. Build runtime groups and IR mapping
    groups, ir_to_group = _build_runtime_groups_and_map(form, form_data, ir)

    # 4. For each runtime integral, build RuntimeIntegralInfo
    integral_infos: dict[IntegralKey, RuntimeIntegralInfo] = {}

    ir_type_counts: dict[str, int] = {}
    for integral_ir in ir.integrals:
        itype = integral_ir.expression.integral_type
        idx = ir_type_counts.get(itype, 0)
        ir_type_counts[itype] = idx + 1

        key: IntegralKey = (itype, idx)
        group = ir_to_group.get(key)
        if group is None:
            continue  # Not a runtime integral
        subdomain_id = group.subdomain_ids[0] if group.subdomain_ids else 0
        subdomain_id = -1 if subdomain_id == "otherwise" else int(subdomain_id)

        info = _analyse_single_runtime_integral(
            integral_ir=integral_ir,
            integral_type=itype,
            ir_index=idx,
            form_data=form_data,
            subdomain_id=subdomain_id,
        )
        integral_infos[key] = info

    # 5. Pack metadata
    meta = {
        "form_rank": len(form.arguments()),
        "num_runtime_groups": len(groups),
    }

    return RuntimeAnalysisInfo(
        ir=ir,
        groups=groups,
        integral_infos=integral_infos,
        form_data=form_data,
        meta=meta,
    )


# Backward compatibility aliases
RuntimeInfo = RuntimeAnalysisInfo


def build_runtime_info(form: ufl.Form, options: dict[str, Any]) -> RuntimeAnalysisInfo:
    """Backward-compatible wrapper for build_runtime_analysis."""
    return build_runtime_analysis(form, options)
