"""Quadrature-point coefficient terminals for runintgen forms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any

import basix.ufl
import numpy as np
import numpy.typing as npt
import ufl

QuadratureFunctionCallable = Callable[[npt.NDArray[np.float64]], npt.ArrayLike]


@dataclass(frozen=True)
class QuadratureFunctionSpec:
    """Metadata attached to a UFL coefficient with quadrature semantics."""

    name: str | None
    value_shape: tuple[int, ...]
    value_size: int


@dataclass(frozen=True)
class QuadratureFunctionInfo:
    """Compiler/runtime information for one quadrature function."""

    terminal: ufl.Coefficient
    coefficient_number: int
    slot: int
    name: str | None
    label: str
    value_shape: tuple[int, ...]
    value_size: int


def _is_function_space(value: Any) -> bool:
    """Return whether a value behaves like a UFL function space."""
    return hasattr(value, "ufl_domain") and hasattr(value, "ufl_element")


def _as_ufl_domain(value: Any) -> Any:
    """Return a UFL domain from a UFL/DOLFINx mesh-like object."""
    if isinstance(value, ufl.Mesh):
        return value
    if hasattr(value, "ufl_domain"):
        return value.ufl_domain()
    return value


def quadrature_function_space(
    domain: Any,
    shape: tuple[int, ...] = (),
) -> ufl.FunctionSpace:
    """Create the default DG0 UFL space for a quadrature function."""
    ufl_domain = _as_ufl_domain(domain)
    cell = ufl_domain.ufl_cell()
    cell_name = cell.cellname() if callable(cell.cellname) else cell.cellname
    element = basix.ufl.element(
        "DG",
        cell_name,
        0,
        shape=shape,
    )
    return ufl.FunctionSpace(ufl_domain, element)


class QuadratureFunction(ufl.Coefficient):
    """UFL coefficient whose runtime values live at quadrature points."""

    def __init__(
        self,
        space_or_domain: Any,
        source: QuadratureFunctionCallable | None = None,
        *,
        name: str | None = None,
        shape: tuple[int, ...] = (),
    ) -> None:
        """Initialise a quadrature-backed UFL coefficient.

        Args:
            space_or_domain: UFL function space, UFL mesh/domain, or a mesh-like
                object exposing ``ufl_domain``. Passing a mesh/domain creates a
                DG0 UFL function space with optional ``shape``.
            source: Optional callable evaluated as ``source(physical_points)``
                during custom-data creation.
            name: Optional diagnostic label. It is not used for identity.
            shape: Optional value shape for the default DG0 space.
        """
        if _is_function_space(space_or_domain):
            space = space_or_domain
        else:
            space = quadrature_function_space(space_or_domain, shape)

        super().__init__(space)

        value_shape = tuple(shape or self.ufl_shape)
        value_size = int(prod(value_shape)) if value_shape else 1
        self._runintgen_quadrature_function = QuadratureFunctionSpec(
            name=name,
            value_shape=value_shape,
            value_size=value_size,
        )
        self._runintgen_source = source
        self._runintgen_values: dict[str, npt.ArrayLike] = {}

    def set_values(self, quadrature: Any, values: npt.ArrayLike) -> None:
        """Attach explicit provider-owned values for one quadrature rule set."""
        rule_id = getattr(quadrature, "rule_id", None)
        if rule_id is None:
            raise TypeError("quadrature must carry a stable rule_id.")
        self._runintgen_values[str(rule_id)] = values

    def is_cellwise_constant(self) -> bool:
        """Return false so UFL does not simplify derivatives to zero.

        The default symbolic space may be DG0 for FEniCSx compatibility, but a
        quadrature function is not interpreted as that finite-element field in
        generated kernels. Derivatives are therefore rejected explicitly instead
        of being silently simplified by UFL.
        """
        return False


def is_quadrature_function(value: Any) -> bool:
    """Return whether a UFL terminal is a runintgen quadrature function."""
    return hasattr(value, "_runintgen_quadrature_function")


def quadrature_function_spec(value: Any) -> QuadratureFunctionSpec:
    """Return quadrature-function metadata for a tagged UFL coefficient."""
    return getattr(value, "_runintgen_quadrature_function")


def quadrature_function_source(value: Any) -> QuadratureFunctionCallable | None:
    """Return the optional callable source attached to a quadrature function."""
    return getattr(value, "_runintgen_source", None)


def quadrature_function_values(value: Any) -> dict[str, npt.ArrayLike]:
    """Return explicit rule-bound values attached to a quadrature function."""
    return getattr(value, "_runintgen_values", {})


def expression_quadrature_functions(value: Any) -> tuple[ufl.Coefficient, ...]:
    """Return quadrature-function coefficients found in a UFL expression."""
    coefficients = ufl.algorithms.extract_coefficients(value)
    return tuple(coef for coef in coefficients if is_quadrature_function(coef))


def integral_quadrature_functions(value: Any) -> tuple[ufl.Coefficient, ...]:
    """Return quadrature-function coefficients found in a UFL integral."""
    return expression_quadrature_functions(value.integrand())


def form_quadrature_functions(value: ufl.Form) -> tuple[ufl.Coefficient, ...]:
    """Return quadrature-function coefficients found anywhere in a UFL form."""
    functions: dict[ufl.Coefficient, None] = {}
    for integral in value.integrals():
        for coefficient in integral_quadrature_functions(integral):
            functions.setdefault(coefficient, None)
    return tuple(functions)


_DISALLOWED_OPERATORS = (
    ufl.classes.CellAvg,
    ufl.classes.Div,
    ufl.classes.FacetAvg,
    ufl.classes.Grad,
    ufl.classes.NablaDiv,
    ufl.classes.NablaGrad,
    ufl.classes.ReferenceDiv,
    ufl.classes.ReferenceGrad,
)


def _contains_quadrature_function(value: Any) -> bool:
    """Return whether an expression tree contains a quadrature function."""
    if is_quadrature_function(value):
        return True
    return any(
        _contains_quadrature_function(operand)
        for operand in getattr(value, "ufl_operands", ())
    )


def _quadrature_function_labels(value: Any) -> list[str]:
    """Return diagnostic labels for quadrature functions in an expression."""
    labels = []
    for coefficient in expression_quadrature_functions(value):
        spec = quadrature_function_spec(coefficient)
        labels.append(spec.name or "<unnamed>")
    return labels


def validate_quadrature_function_expression(value: Any) -> None:
    """Reject v1-unsupported quadrature-function expression constructs."""
    if isinstance(value, _DISALLOWED_OPERATORS) and _contains_quadrature_function(
        value
    ):
        labels = _quadrature_function_labels(value)
        raise NotImplementedError(
            "Derivatives and averages of QuadratureFunction are not supported "
            "in v1. Create a separate QuadratureFunction for the precomputed "
            "quantity and supply its quadrature-point values. Affected "
            f"quadrature functions: {', '.join(labels)}."
        )
    for operand in getattr(value, "ufl_operands", ()):
        validate_quadrature_function_expression(operand)


def validate_quadrature_function_form(value: ufl.Form) -> None:
    """Reject unsupported quadrature-function constructs in all integrands."""
    for integral in value.integrals():
        validate_quadrature_function_expression(integral.integrand())


def collect_quadrature_function_infos(ir: Any) -> list[QuadratureFunctionInfo]:
    """Collect quadrature functions from FFCx integral coefficient numbering."""
    by_number: dict[int, ufl.Coefficient] = {}
    for integral_ir in getattr(ir, "integrals", []):
        numbering = getattr(integral_ir.expression, "coefficient_numbering", {})
        for terminal, number in numbering.items():
            if is_quadrature_function(terminal):
                by_number.setdefault(int(number), terminal)

    infos: list[QuadratureFunctionInfo] = []
    for slot, coefficient_number in enumerate(sorted(by_number)):
        terminal = by_number[coefficient_number]
        spec = quadrature_function_spec(terminal)
        label = spec.name or f"quadrature_function_{slot}"
        infos.append(
            QuadratureFunctionInfo(
                terminal=terminal,
                coefficient_number=coefficient_number,
                slot=slot,
                name=spec.name,
                label=label,
                value_shape=spec.value_shape,
                value_size=spec.value_size,
            )
        )
    return infos


__all__ = [
    "QuadratureFunction",
    "QuadratureFunctionCallable",
    "QuadratureFunctionInfo",
    "QuadratureFunctionSpec",
    "collect_quadrature_function_infos",
    "expression_quadrature_functions",
    "form_quadrature_functions",
    "integral_quadrature_functions",
    "is_quadrature_function",
    "quadrature_function_space",
    "quadrature_function_source",
    "quadrature_function_spec",
    "quadrature_function_values",
    "validate_quadrature_function_expression",
    "validate_quadrature_function_form",
]
