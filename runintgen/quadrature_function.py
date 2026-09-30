"""Quadrature-point coefficient terminals for runintgen forms."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Any

import basix.ufl
import numpy as np
import numpy.typing as npt
import ufl

from .measures import RuntimeIntegralMode, runtime_integral_mode

QuadratureFunctionCallable = Callable[[npt.NDArray[np.float64]], npt.ArrayLike]
QuadratureFunctionSource = QuadratureFunctionCallable | Any

_RESTRICTIONS = (None, "+", "-")


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
    restriction: str | None
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


class QuadratureFunctionMixin:
    """Quadrature-point value semantics for UFL coefficient classes.

    The runintgen :class:`QuadratureFunction` and backend coefficients, such as
    a DOLFINx ``Function`` in CutFEMx, share this implementation. A subclass
    lists the mixin after its ``ufl.Coefficient`` base (FFCx finds terminal
    handlers through the first base class), binds
    ``is_cellwise_constant = QuadratureFunctionMixin.is_cellwise_constant`` in
    its body (the coefficient's method comes first in the MRO), and calls
    :meth:`_init_quadrature_function` after the coefficient constructor.

    Values are resolved per quadrature rule set (``rule_id``) and side
    (``restriction``), in this order: arrays attached with :meth:`set_values`, a
    context-aware evaluator ``source.evaluate(context)``, a callable
    ``source(physical_points)``, and a backend fallback evaluator.
    """

    def _init_quadrature_function(
        self,
        source: QuadratureFunctionSource | None,
        *,
        name: str | None,
        value_shape: tuple[int, ...],
    ) -> None:
        """Attach quadrature-function metadata and empty value storage."""
        value_shape = tuple(int(i) for i in value_shape)
        self._runintgen_quadrature_function = QuadratureFunctionSpec(
            name=name,
            value_shape=value_shape,
            value_size=int(prod(value_shape)) if value_shape else 1,
        )
        self._runintgen_source = source
        self._runintgen_values: dict[tuple[str, str | None], npt.ArrayLike] = {}
        # Evaluated values live only as long as some custom_data uses them, so
        # rule sets rebuilt in a time or optimisation loop do not accumulate.
        self._runintgen_cache: weakref.WeakValueDictionary[Any, np.ndarray] = (
            weakref.WeakValueDictionary()
        )

    def set_values(
        self,
        quadrature: Any,
        values: npt.ArrayLike,
        *,
        restriction: str | None = None,
    ) -> None:
        """Attach explicit provider-owned values for one quadrature rule set.

        Args:
            quadrature: Rule set of the measure, identified by its ``rule_id``.
            values: Point-major values with shape ``(total_nq,)`` or
                ``(total_nq, value_size)``. The array is borrowed, not copied.
            restriction: ``"+"`` or ``"-"`` to supply the values of one side of
                an interior-facet measure. Values attached without a restriction
                are used for every side that has no side-specific values.
        """
        rule_id = getattr(quadrature, "rule_id", None)
        if rule_id is None:
            raise TypeError("quadrature must carry a stable rule_id.")
        if restriction not in _RESTRICTIONS:
            raise ValueError(
                f"restriction must be None, '+' or '-', got {restriction!r}."
            )
        self._runintgen_values[(str(rule_id), restriction)] = values

    def set_evaluator(self, evaluator: Any) -> None:
        """Attach a context-aware evaluator source and clear cached values."""
        self._runintgen_source = evaluator
        self._runintgen_cache.clear()

    def invalidate(self) -> None:
        """Clear cached quadrature values for this coefficient."""
        self._runintgen_cache.clear()
        invalidate = getattr(self._runintgen_source, "invalidate", None)
        if invalidate is not None:
            invalidate()

    def update(self, *, version: int | None = None) -> None:
        """Notify the evaluator that its source data changed."""
        update = getattr(self._runintgen_source, "update", None)
        if update is not None:
            update(version=version)
        self._runintgen_cache.clear()

    def is_cellwise_constant(self) -> bool:
        """Return false so UFL does not simplify derivatives to zero.

        The default symbolic space may be DG0 for FEniCSx compatibility, but a
        quadrature function is not interpreted as that finite-element field in
        generated kernels. Derivatives are therefore rejected explicitly instead
        of being silently simplified by UFL.
        """
        return False


class QuadratureFunction(ufl.Coefficient, QuadratureFunctionMixin):
    """UFL coefficient whose runtime values live at quadrature points."""

    is_cellwise_constant = QuadratureFunctionMixin.is_cellwise_constant

    def __init__(
        self,
        space_or_domain: Any,
        source: QuadratureFunctionSource | None = None,
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
                during custom-data creation, or a context-aware evaluator with
                an ``evaluate(context)`` method.
            name: Optional diagnostic label. It is not used for identity.
            shape: Optional value shape for the default DG0 space.
        """
        if _is_function_space(space_or_domain):
            space = space_or_domain
        else:
            space = quadrature_function_space(space_or_domain, shape)

        super().__init__(space)
        self._init_quadrature_function(
            source, name=name, value_shape=tuple(shape or self.ufl_shape)
        )


def is_quadrature_function(value: Any) -> bool:
    """Return whether a UFL terminal is a runintgen quadrature function."""
    return hasattr(value, "_runintgen_quadrature_function")


def quadrature_function_spec(value: Any) -> QuadratureFunctionSpec:
    """Return quadrature-function metadata for a tagged UFL coefficient."""
    return getattr(value, "_runintgen_quadrature_function")


def quadrature_function_source(value: Any) -> QuadratureFunctionSource | None:
    """Return the optional callable source attached to a quadrature function."""
    return getattr(value, "_runintgen_source", None)


def quadrature_function_values(
    value: Any,
) -> dict[tuple[str, str | None], npt.ArrayLike]:
    """Return explicit values keyed by ``(rule_id, restriction)``."""
    return getattr(value, "_runintgen_values", {})


def quadrature_function_explicit_values(
    value: Any,
    rule_id: Any,
    restriction: str | None = None,
) -> npt.ArrayLike | None:
    """Return explicit values for one rule set and side, if attached.

    Side-specific values take precedence over values attached without a
    restriction, which serve both sides of an interior facet.
    """
    values = quadrature_function_values(value)
    for key in ((str(rule_id), restriction), (str(rule_id), None)):
        if key in values:
            return values[key]
    return None


def quadrature_function_cache(value: Any) -> Any:
    """Return evaluator cache storage attached to a quadrature function."""
    cache = getattr(value, "_runintgen_cache", None)
    if cache is None:
        cache = weakref.WeakValueDictionary()
        setattr(value, "_runintgen_cache", cache)
    return cache


def quadrature_function_layout(
    form: ufl.Form,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return ``(coefficient position, value shape)`` of quadrature functions.

    Positions index ``form.coefficients()``, the numbering UFL uses for form
    signatures. A QuadratureFunction and an ordinary coefficient on the same
    element have equal signatures, so runintgen adds this layout to its JIT
    cache key.
    """
    return tuple(
        (position, quadrature_function_spec(coefficient).value_shape)
        for position, coefficient in enumerate(form.coefficients())
        if is_quadrature_function(coefficient)
    )


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
    """Reject unsupported quadrature-function constructs in a form.

    Only runtime integrals can load quadrature functions. Standard kernels,
    including the standard entities of mixed integrals, would read one as an
    ordinary coefficient from ``w``, which holds no values for it.
    """
    for integral in value.integrals():
        validate_quadrature_function_expression(integral.integrand())
        mode = runtime_integral_mode(integral)
        if mode is RuntimeIntegralMode.RUNTIME:
            continue
        functions = integral_quadrature_functions(integral)
        if not functions:
            continue
        labels = ", ".join(
            quadrature_function_spec(f).name or "<unnamed>" for f in functions
        )
        if mode is RuntimeIntegralMode.MIXED:
            raise NotImplementedError(
                "QuadratureFunction in mixed standard/runtime integrals is not "
                "implemented yet: the standard entities would read it as an "
                "ordinary coefficient. Integrate it with a runtime-only measure. "
                f"Affected quadrature functions: {labels}."
            )
        raise NotImplementedError(
            "QuadratureFunction in standard integrals is not implemented yet. "
            "Use a runtime measure with QuadratureRules for now. Affected "
            f"quadrature functions: {labels}."
        )


def collect_quadrature_function_infos(ir: Any) -> list[QuadratureFunctionInfo]:
    """Collect quadrature functions from FFCx integral coefficient numbering."""
    by_key: dict[tuple[ufl.Coefficient, str | None], int] = {}
    for integral_ir in getattr(ir, "integrals", []):
        expression = integral_ir.expression
        numbering = getattr(expression, "coefficient_numbering", {})
        for integrand_data in getattr(expression, "integrand", {}).values():
            factorization = integrand_data.get("factorization")
            if factorization is None:
                continue
            for node_data in factorization.nodes.values():
                mt = node_data.get("mt")
                terminal = getattr(mt, "terminal", None)
                if not is_quadrature_function(terminal):
                    continue
                if terminal not in numbering:
                    continue
                restriction = getattr(mt, "restriction", None)
                by_key.setdefault((terminal, restriction), int(numbering[terminal]))

    restriction_rank = {None: 0, "+": 1, "-": 2}

    infos: list[QuadratureFunctionInfo] = []
    entries = sorted(
        ((number, restriction_rank.get(restriction, 99), terminal, restriction)
         for (terminal, restriction), number in by_key.items()),
        key=lambda item: (item[0], item[1], repr(item[2])),
    )
    for slot, (coefficient_number, _, terminal, restriction) in enumerate(entries):
        spec = quadrature_function_spec(terminal)
        infos.append(
            QuadratureFunctionInfo(
                terminal=terminal,
                restriction=restriction,
                coefficient_number=coefficient_number,
                slot=slot,
                name=spec.name,
                label=_quadrature_function_label(spec.name, slot, restriction),
                value_shape=spec.value_shape,
                value_size=spec.value_size,
            )
        )
    return infos


def _quadrature_function_label(
    name: str | None, slot: int, restriction: str | None
) -> str:
    """Return the diagnostic label of one quadrature-function slot."""
    label = name or f"quadrature_function_{slot}"
    return f"{label}({restriction})" if restriction is not None else label


def rebind_quadrature_function_infos(
    infos: Sequence[QuadratureFunctionInfo],
    template_form: ufl.Form,
    form: ufl.Form,
) -> list[QuadratureFunctionInfo]:
    """Point the infos of a cached module at the terminals of ``form``.

    A JIT module compiled for ``template_form`` is reused for any form with an
    equal signature. Such forms number their coefficients identically, since
    the signature renumbers coefficients by position in ``form.coefficients()``.
    Each info therefore moves to the coefficient at the same position, whatever
    the names of the two quadrature functions.
    """
    if not infos:
        return list(infos)

    positions = {id(c): i for i, c in enumerate(template_form.coefficients())}
    coefficients = form.coefficients()
    rebound = []
    for info in infos:
        position = positions.get(id(info.terminal))
        if position is None or position >= len(coefficients):
            raise RuntimeError(
                f"Cached QuadratureFunction {info.label!r} is not a coefficient "
                "of the form the cached module was compiled for."
            )
        terminal = coefficients[position]
        if not is_quadrature_function(terminal):
            raise RuntimeError(
                f"Coefficient {position} of the form is not a QuadratureFunction, "
                f"but the cached module loads it as {info.label!r}."
            )
        spec = quadrature_function_spec(terminal)
        if spec.value_shape != info.value_shape:
            raise RuntimeError(
                f"QuadratureFunction at coefficient {position} has value shape "
                f"{spec.value_shape}; the cached module expects {info.value_shape}."
            )
        rebound.append(
            replace(
                info,
                terminal=terminal,
                name=spec.name,
                label=_quadrature_function_label(
                    spec.name, info.slot, info.restriction
                ),
            )
        )
    return rebound


__all__ = [
    "QuadratureFunction",
    "QuadratureFunctionCallable",
    "QuadratureFunctionInfo",
    "QuadratureFunctionMixin",
    "QuadratureFunctionSpec",
    "QuadratureFunctionSource",
    "collect_quadrature_function_infos",
    "expression_quadrature_functions",
    "form_quadrature_functions",
    "integral_quadrature_functions",
    "is_quadrature_function",
    "quadrature_function_cache",
    "quadrature_function_explicit_values",
    "quadrature_function_layout",
    "quadrature_function_space",
    "quadrature_function_source",
    "quadrature_function_spec",
    "quadrature_function_values",
    "rebind_quadrature_function_infos",
    "validate_quadrature_function_expression",
    "validate_quadrature_function_form",
]
