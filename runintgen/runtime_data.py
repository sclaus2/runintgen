"""Runtime C ABI helpers for runintgen kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .cpp_headers import runtime_abi_cdef

if TYPE_CHECKING:
    import cffi


CFFI_DEF = runtime_abi_cdef()


@dataclass
class RuntimeQuadratureRule:
    """Reference quadrature carried by ``custom_data``.

    Weights are expected to be pre-scaled by the runtime integration provider,
    including the relevant cell/facet/cut measure scaling.
    """

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Normalise quadrature arrays."""
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("Runtime quadrature points must have shape (nq, tdim).")
        if self.points.shape[0] != self.weights.shape[0]:
            raise ValueError("Runtime quadrature points and weights disagree on nq.")

    @property
    def nq(self) -> int:
        """Number of quadrature points."""
        return int(self.weights.shape[0])

    @property
    def tdim(self) -> int:
        """Reference point dimension."""
        return int(self.points.shape[1])


@dataclass
class RuntimeBasixElement:
    """Opaque Basix element handle carried by ``custom_data``."""

    handle: int = 0
    tabulate: int = 0


@dataclass
class RuntimeTableRequest:
    """Generated request for one FFCx table-reference slot."""

    slot: int
    element_index: int
    derivative_counts: tuple[int, ...] = ()
    flat_component: int | None = None
    num_permutations: int = 1
    num_entities: int = 1
    num_dofs: int = 0
    block_size: int | None = None
    offset: int | None = None
    is_uniform: bool = False
    is_permuted: bool = False

    @property
    def derivative_order(self) -> int:
        """Total derivative order."""
        return int(sum(self.derivative_counts))


@dataclass
class RuntimeTableView:
    """One raw Basix tabulation view.

    Values are flattened in Basix tabulate order:
    ``[derivative][point][dof][component]``.
    """

    values: npt.NDArray[np.float64]
    num_derivatives: int
    num_points: int
    num_dofs: int
    num_components: int

    def __post_init__(self) -> None:
        """Normalise values to contiguous float64 storage."""
        self.values = np.ascontiguousarray(self.values, dtype=np.float64)
        expected = (
            self.num_derivatives
            * self.num_points
            * self.num_dofs
            * self.num_components
        )
        if self.values.size != expected:
            raise ValueError(
                f"Runtime table has {self.values.size} values, expected {expected}."
            )


class RuntimeContextBuilder:
    """Pack a runtime context into CFFI structures.

    This helper is intended for tests and lightweight JIT use. Production
    DOLFINx/CutFEMx integration should populate the same C ABI from C++ and
    implement each element's `tabulate` function pointer using real Basix
    elements and wrapper-owned scratch storage.
    """

    def __init__(self, ffi: "cffi.FFI") -> None:
        """Initialise the builder."""
        self.ffi = ffi
        self._refs: list[object] = []

    def build_context(
        self,
        quadrature: RuntimeQuadratureRule | list[RuntimeQuadratureRule],
        basix_elements: list[RuntimeBasixElement] | None = None,
        form_context: "cffi.CData | None" = None,
        is_cut: npt.ArrayLike | None = None,
        scratch: "cffi.CData | None" = None,
    ) -> "cffi.CData":
        """Build a ``runintgen_context*``."""
        ffi = self.ffi
        rules = quadrature if isinstance(quadrature, list) else [quadrature]
        c_rules = ffi.new(f"runintgen_quadrature_rule[{len(rules)}]")
        self._refs.append(c_rules)
        for i, rule in enumerate(rules):
            points = np.ascontiguousarray(rule.points.ravel(), dtype=np.float64)
            weights = np.ascontiguousarray(rule.weights, dtype=np.float64)
            self._refs.extend([points, weights])
            c_rules[i].nq = rule.nq
            c_rules[i].tdim = rule.tdim
            c_rules[i].points = ffi.cast("const double*", points.ctypes.data)
            c_rules[i].weights = ffi.cast("const double*", weights.ctypes.data)

        if form_context is None:
            form_context = self.build_form_context(
                basix_elements=basix_elements,
                scratch=scratch,
            )

        ctx = ffi.new("runintgen_context*")
        self._refs.append(ctx)
        ctx.num_rules = len(rules)
        ctx.rules = c_rules
        if is_cut is None:
            is_cut_array = np.ones(len(rules), dtype=np.uint8)
        else:
            is_cut_array = np.ascontiguousarray(is_cut, dtype=np.uint8)
            if is_cut_array.size != len(rules):
                raise ValueError("is_cut must have one entry per quadrature rule.")
        self._refs.append(is_cut_array)
        ctx.is_cut = ffi.cast("const uint8_t*", is_cut_array.ctypes.data)
        ctx.form = form_context
        return ctx

    def build_form_context(
        self,
        basix_elements: list[RuntimeBasixElement] | None = None,
        descriptor: "cffi.CData | None" = None,
        scratch: "cffi.CData | None" = None,
    ) -> "cffi.CData":
        """Build and retain a ``runintgen_form_context*``."""
        ffi = self.ffi
        elements = basix_elements or []
        c_elements = ffi.new(f"runintgen_basix_element[{len(elements)}]")
        self._refs.append(c_elements)
        for i, element in enumerate(elements):
            c_elements[i].handle = ffi.cast("const void*", element.handle)
            c_elements[i].tabulate = (
                ffi.cast("runintgen_element_tabulate_fn", element.tabulate)
                if element.tabulate
                else ffi.NULL
            )

        form = ffi.new("runintgen_form_context*")
        self._refs.append(form)
        form.num_elements = len(elements)
        form.elements = c_elements
        form.descriptor = descriptor or ffi.NULL
        form.scratch = scratch or ffi.NULL
        return form

    def build_request(self, request: RuntimeTableRequest) -> "cffi.CData":
        """Build and retain a ``runintgen_table_request*``."""
        ffi = self.ffi
        c_request = ffi.new("runintgen_table_request*")
        self._refs.append(c_request)
        c_request.slot = request.slot
        c_request.element_index = request.element_index
        c_request.derivative_order = request.derivative_order
        for i, count in enumerate(request.derivative_counts[:4]):
            c_request.derivative_counts[i] = int(count)
        c_request.flat_component = (
            -1 if request.flat_component is None else int(request.flat_component)
        )
        c_request.num_permutations = request.num_permutations
        c_request.num_entities = request.num_entities
        c_request.num_dofs = request.num_dofs
        c_request.block_size = -1 if request.block_size is None else request.block_size
        c_request.offset = -1 if request.offset is None else request.offset
        c_request.is_uniform = int(request.is_uniform)
        c_request.is_permuted = int(request.is_permuted)
        return c_request


def to_intptr(ffi: "cffi.FFI", data: "cffi.CData") -> int:
    """Return an ``intptr_t`` representation of a CFFI pointer."""
    return int(ffi.cast("intptr_t", data))
