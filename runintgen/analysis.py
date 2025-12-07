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

from ffcx.analysis import analyze_ufl_objects
from ffcx.ir.representation import compute_ir
from ffcx.ir.elementtables import get_modified_terminal_element
from ffcx.options import get_options

from .measures import is_runtime_integral

# Type aliases
DerivTuple = tuple[int, ...]  # e.g. (1, 0) in 2D
IntegralKey = tuple[str, int]  # (integral_type, ir_index)
Key = tuple[Any, str, tuple[Any, ...]]  # (domain, integral_type, subdomain_ids)


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
    """Usage of one logical argument (test/trial/coef/geometry) in a runtime integral."""

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


def _strip_runtime_metadata(form: ufl.Form) -> ufl.Form:
    """Strip quadrature_rule='runtime' metadata so FFCX doesn't choke on it.

    FFCX tries to interpret quadrature_rule as a basix quadrature type.
    We need to replace 'runtime' with a valid default (or remove it).
    """
    new_integrals = []
    for integral in form.integrals():
        md = dict(integral.metadata())
        if md.get("quadrature_rule") == "runtime":
            md["quadrature_rule"] = "default"
        new_integral = integral.reconstruct(metadata=md)
        new_integrals.append(new_integral)

    return ufl.Form(new_integrals)


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
    original_providers: dict[tuple[str, int], Any] = {}
    for integral in form.integrals():
        if is_runtime_integral(integral):
            orig_key = (integral.integral_type(), integral.subdomain_id())
            original_providers[orig_key] = integral.subdomain_data()

    # Scan integral_data for runtime integrals
    groups_dict: dict[Key, RuntimeGroup] = {}

    for itg_data in form_data.integral_data:
        domain = itg_data.domain
        itype = itg_data.integral_type
        subdomain_ids = tuple(itg_data.subdomain_id)
        key: Key = (domain, itype, subdomain_ids)

        quadrature_provider: Any = None

        for integral in itg_data.integrals:
            if is_runtime_integral(integral):
                sd = integral.subdomain_data()
                if sd is not None:
                    quadrature_provider = sd
                else:
                    for sid in subdomain_ids:
                        lookup_key = (itype, sid)
                        if lookup_key in original_providers:
                            quadrature_provider = original_providers[lookup_key]
                            break
                if quadrature_provider is not None:
                    break

        if quadrature_provider is not None or any(
            is_runtime_integral(i) for i in itg_data.integrals
        ):
            groups_dict[key] = RuntimeGroup(
                domain=domain,
                integral_type=itype,
                subdomain_ids=subdomain_ids,
                quadrature_provider=quadrature_provider,
            )

    groups = list(groups_dict.values())

    # Build mapping from IR integrals to runtime groups
    ir_to_group: dict[IntegralKey, RuntimeGroup] = {}

    if not ir.forms:
        return groups, ir_to_group

    # Map each IR integral to a runtime group
    ir_type_counts: dict[str, int] = {}
    for integral_ir in ir.integrals:
        itype = integral_ir.expression.integral_type
        idx = ir_type_counts.get(itype, 0)
        ir_type_counts[itype] = idx + 1

        # Match based on integral type
        for group in groups:
            if group.integral_type == itype:
                ir_to_group[(itype, idx)] = group
                break

    return groups, ir_to_group


def _analyse_single_runtime_integral(
    integral_ir: Any,
    integral_type: str,
    ir_index: int,
    form_data: Any,
) -> RuntimeIntegralInfo:
    """Extract ArgumentInfo + ElementInfo for one runtime integral."""

    subdomain_id = getattr(integral_ir.expression, "subdomain_id", 0)

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
    complex_mode = options.get("scalar_type", "float64") in ("complex64", "complex128")

    # 1. Compute form_data with disabled integral scaling
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

    # 2. Let FFCX do the standard analysis/IR
    form_for_ffcx = _strip_runtime_metadata(form)
    scalar_type = np.dtype(options.get("scalar_type", "float64"))
    analysis = analyze_ufl_objects([form_for_ffcx], scalar_type)

    ffcx_options = get_options(options)
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
        if key not in ir_to_group:
            continue  # Not a runtime integral

        info = _analyse_single_runtime_integral(
            integral_ir=integral_ir,
            integral_type=itype,
            ir_index=idx,
            form_data=form_data,
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
