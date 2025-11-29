"""FE table metadata extraction for runtime integrals.

This module extracts element, component, and derivative information
from FFCX IR for runtime integrals, following the approach in AGENTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import basix.ufl
import ufl

from ffcx.ir.elementtables import get_modified_terminal_element

from .analysis import RuntimeGroup, RuntimeInfo


@dataclass(frozen=True)
class ComponentRequest:
    """Request for a specific component of an element.

    Attributes:
        element: The basix/UFL element.
        element_hash: Hash of the element for identification.
        role: "argument" or "coefficient".
        index: Argument number (0=test, 1=trial) or coefficient index.
        component: Flat component index.
        max_deriv: Maximum derivative order needed (sum of derivative counts).
        local_derivatives: Tuple of derivative counts per reference direction.
    """

    element: Any  # basix.ufl._ElementBase
    element_hash: int
    role: str  # "argument" or "coefficient"
    index: int  # argument number or coefficient index
    component: int  # flat component index
    max_deriv: int  # max derivative order
    local_derivatives: tuple[int, ...]  # derivative counts per direction

    def __hash__(self) -> int:
        return hash(
            (
                self.element_hash,
                self.role,
                self.index,
                self.component,
                self.local_derivatives,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComponentRequest):
            return False
        return (
            self.element_hash == other.element_hash
            and self.role == other.role
            and self.index == other.index
            and self.component == other.component
            and self.local_derivatives == other.local_derivatives
        )


@dataclass
class IntegralRuntimeMeta:
    """Runtime metadata for a single integral.

    Attributes:
        integral_type: "cell", "exterior_facet", etc.
        subdomain_id: The subdomain identifier.
        component_requests: List of ComponentRequest for this integral.
        unique_elements: Set of unique elements used.
        max_derivatives: Dict mapping element hash to max derivative order.
    """

    integral_type: str
    subdomain_id: int
    component_requests: list[ComponentRequest] = field(default_factory=list)

    @property
    def unique_elements(self) -> list[Any]:
        """Get unique elements used in this integral."""
        seen: dict[int, Any] = {}
        for cr in self.component_requests:
            if cr.element_hash not in seen:
                seen[cr.element_hash] = cr.element
        return list(seen.values())

    @property
    def max_derivative_per_element(self) -> dict[int, int]:
        """Get maximum derivative order for each element (by hash)."""
        result: dict[int, int] = {}
        for cr in self.component_requests:
            current = result.get(cr.element_hash, 0)
            result[cr.element_hash] = max(current, cr.max_deriv)
        return result

    @property
    def components_per_element(self) -> dict[int, set[int]]:
        """Get set of components used for each element (by hash)."""
        result: dict[int, set[int]] = {}
        for cr in self.component_requests:
            if cr.element_hash not in result:
                result[cr.element_hash] = set()
            result[cr.element_hash].add(cr.component)
        return result


def _get_element_hash(element: Any) -> int:
    """Get a stable hash for an element."""
    # Use basix's element hash if available
    if hasattr(element, "__hash__"):
        return hash(element)
    return id(element)


def _collect_modified_terminal(
    mt,
    coefficient_numbering: dict,
    seen: set[tuple],
) -> ComponentRequest | None:
    """Extract ComponentRequest from a single ModifiedTerminal.

    Args:
        mt: ModifiedTerminal object.
        coefficient_numbering: Mapping from ufl.Coefficient to index.
        seen: Set of already-seen (element_hash, role, index, comp, derivs) tuples.

    Returns:
        ComponentRequest if terminal has an element and is not duplicate, else None.
    """
    mte = get_modified_terminal_element(mt)
    if mte is None:
        # Skip terminals without elements (FloatValue, IntValue, etc.)
        return None

    element, averaged, local_derivatives, flat_component = mte
    terminal = mt.terminal

    # Determine role and index
    if isinstance(terminal, ufl.Argument):
        role = "argument"
        index = terminal.number()
    elif isinstance(terminal, ufl.Coefficient):
        role = "coefficient"
        index = coefficient_numbering.get(terminal, -1)
    elif isinstance(terminal, ufl.classes.SpatialCoordinate):
        role = "coordinate"
        index = 0
    elif isinstance(terminal, ufl.classes.Jacobian):
        role = "jacobian"
        index = 0
    else:
        return None

    max_deriv = sum(local_derivatives)
    element_hash = _get_element_hash(element)

    # Create unique key to avoid duplicates
    key = (element_hash, role, index, flat_component, local_derivatives)
    if key in seen:
        return None
    seen.add(key)

    return ComponentRequest(
        element=element,
        element_hash=element_hash,
        role=role,
        index=index,
        component=flat_component,
        max_deriv=max_deriv,
        local_derivatives=local_derivatives,
    )


def _collect_component_requests_from_modified_arguments(
    modified_arguments: list,
    coefficient_numbering: dict,
) -> list[ComponentRequest]:
    """Collect ComponentRequest objects from modified_arguments list.

    Args:
        modified_arguments: List of ModifiedTerminal objects from integrand data.
        coefficient_numbering: Mapping from ufl.Coefficient to index.

    Returns:
        List of ComponentRequest objects.
    """
    requests: list[ComponentRequest] = []
    seen: set[tuple] = set()

    for mt in modified_arguments:
        req = _collect_modified_terminal(mt, coefficient_numbering, seen)
        if req is not None:
            requests.append(req)

    return requests


def _collect_component_requests_from_graph(
    graph_nodes: dict[int, dict],
    coefficient_numbering: dict,
) -> list[ComponentRequest]:
    """Collect ComponentRequest objects from factorization graph nodes.

    This extracts Coefficients, Jacobians, and other non-Argument terminals
    from the expression graph.

    Args:
        graph_nodes: The nodes of the factorization graph (factorization.nodes).
        coefficient_numbering: Mapping from ufl.Coefficient to index.

    Returns:
        List of ComponentRequest objects.
    """
    requests: list[ComponentRequest] = []
    seen: set[tuple] = set()

    for node_id, node_data in graph_nodes.items():
        mt = node_data.get("mt")
        if mt is None:
            continue

        req = _collect_modified_terminal(mt, coefficient_numbering, seen)
        if req is not None:
            requests.append(req)

    return requests


def extract_integral_metadata(
    runtime_info: RuntimeInfo,
) -> dict[RuntimeGroup, IntegralRuntimeMeta]:
    """Extract FE table metadata for each runtime group.

    This function analyzes the FFCX IR to extract:
    - Which elements are used in each integral
    - Which components of each element are accessed
    - What derivative orders are needed

    Args:
        runtime_info: RuntimeInfo from analysis phase.

    Returns:
        Dictionary mapping RuntimeGroup to IntegralRuntimeMeta.
    """
    ir = runtime_info.ir
    result: dict[RuntimeGroup, IntegralRuntimeMeta] = {}

    if not ir.integrals:
        return result

    # Build a mapping from integral_type to list of IntegralIR
    ir_by_type: dict[str, list] = {}
    for integral_ir in ir.integrals:
        itype = integral_ir.expression.integral_type
        if itype not in ir_by_type:
            ir_by_type[itype] = []
        ir_by_type[itype].append(integral_ir)

    for group in runtime_info.groups:
        itype = group.integral_type
        subdomain_id = group.subdomain_ids[0] if group.subdomain_ids else 0

        # Find matching IntegralIR(s) - for now, take the first matching type
        # TODO: Better matching when multiple subdomains exist
        matching_integrals = ir_by_type.get(itype, [])

        if not matching_integrals:
            # Create empty metadata if no IR found
            result[group] = IntegralRuntimeMeta(
                integral_type=itype,
                subdomain_id=subdomain_id,
            )
            continue

        matching_integral = matching_integrals[0]

        expr_ir = matching_integral.expression
        coefficient_numbering = expr_ir.coefficient_numbering

        # Collect component requests from integrand
        all_requests: list[ComponentRequest] = []

        # The integrand is dict[(basix.CellType, QuadratureRule), dict]
        for (cell_type, qrule), integrand_data in expr_ir.integrand.items():
            # 1. Extract from 'modified_arguments' - list of ModifiedTerminal
            modified_arguments = integrand_data.get("modified_arguments", [])
            if modified_arguments:
                requests = _collect_component_requests_from_modified_arguments(
                    modified_arguments,
                    coefficient_numbering,
                )
                all_requests.extend(requests)

            # 2. Extract from factorization graph (Coefficients, Jacobians, etc.)
            factorization = integrand_data.get("factorization")
            if factorization and hasattr(factorization, "nodes"):
                requests = _collect_component_requests_from_graph(
                    factorization.nodes,
                    coefficient_numbering,
                )
                all_requests.extend(requests)

        # Deduplicate requests
        unique_requests: list[ComponentRequest] = []
        seen_keys: set[tuple] = set()
        for req in all_requests:
            key = (
                req.element_hash,
                req.role,
                req.index,
                req.component,
                req.local_derivatives,
            )
            if key not in seen_keys:
                seen_keys.add(key)
                unique_requests.append(req)

        result[group] = IntegralRuntimeMeta(
            integral_type=itype,
            subdomain_id=subdomain_id,
            component_requests=unique_requests,
        )

    return result


def print_integral_metadata(meta: IntegralRuntimeMeta) -> None:
    """Print human-readable summary of integral metadata."""
    print(f"Integral: {meta.integral_type}, subdomain_id={meta.subdomain_id}")
    print(f"  Unique elements: {len(meta.unique_elements)}")
    print(f"  Component requests: {len(meta.component_requests)}")

    for i, cr in enumerate(meta.component_requests):
        print(
            f"  [{i}] {cr.role}[{cr.index}], component={cr.component}, "
            f"max_deriv={cr.max_deriv}, local_derivs={cr.local_derivatives}"
        )
        print(f"      element: {cr.element}")

    print("  Max derivatives per element:")
    for elem_hash, max_d in meta.max_derivative_per_element.items():
        print(f"    hash={elem_hash}: max_deriv={max_d}")

    print("  Components per element:")
    for elem_hash, comps in meta.components_per_element.items():
        print(f"    hash={elem_hash}: components={sorted(comps)}")
