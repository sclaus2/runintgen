"""Runtime data structures for runintgen kernels.

This module provides Python classes and helpers to create and manage
the runtime data structures (runintgen_data, etc.) that are passed
to generated kernels via the custom_data pointer.

The design supports:
1. Single-config mode: All cells use the same quadrature rule
2. Multi-config mode: Different cells can use different quadrature rules
3. Table caching: Cells sharing the same quadrature share tabulated tables

Usage with DOLFINx:
    # Create runtime data
    builder = RuntimeDataBuilder(element_info)
    builder.add_config(quadrature_points, quadrature_weights)
    data_ptr = builder.build()

    # Create wrapper kernel that captures the runtime data
    wrapper = create_kernel_wrapper(kernel_ptr, data_ptr)

    # Use with DOLFINx Form
    integrals = {IntegralType.cell: [(0, wrapper, cells, active_coeffs)]}
    form = Form(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cffi


@dataclass
class ElementTableInfo:
    """Information about a finite element's tabulated values.

    Attributes:
        ndofs: Number of degrees of freedom.
        nderivs: Number of derivative components in the table.
        table: Tabulated values with shape [nderivs, nq, ndofs].
    """

    ndofs: int
    nderivs: int
    table: npt.NDArray[np.float64]


@dataclass
class QuadratureConfig:
    """A single quadrature configuration with tabulated element values.

    Attributes:
        points: Quadrature points on the reference element, shape [nq, tdim].
        weights: Quadrature weights, shape [nq].
        elements: List of ElementTableInfo for each unique element.
    """

    points: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    elements: list[ElementTableInfo] = field(default_factory=list)

    @property
    def nq(self) -> int:
        """Number of quadrature points."""
        return len(self.weights)


class RuntimeDataBuilder:
    """Builder for runintgen_data structures.

    This class helps construct the C structures needed by runtime kernels,
    handling memory management and layout for CFFI.

    Example (single-config mode):
        builder = RuntimeDataBuilder(ffi)
        config = QuadratureConfig(points, weights)
        config.elements.append(ElementTableInfo(3, 3, table))
        builder.add_config(config)
        data = builder.build()
        # data can be passed to kernel via custom_data

    Example (multi-config mode for per-cell quadrature):
        builder = RuntimeDataBuilder(ffi)
        # Add different quadrature configs
        builder.add_config(config_low_order)   # index 0
        builder.add_config(config_high_order)  # index 1
        # Map cells to configs
        cell_config_map = np.array([0, 0, 1, 0, 1, ...])
        data = builder.build(cell_config_map=cell_config_map)
    """

    def __init__(self, ffi: cffi.FFI) -> None:
        """Initialize the builder.

        Args:
            ffi: CFFI FFI instance with runintgen struct definitions.
        """
        self.ffi = ffi
        self.configs: list[QuadratureConfig] = []
        # Keep references to prevent garbage collection
        self._refs: list = []

    def add_config(self, config: QuadratureConfig) -> int:
        """Add a quadrature configuration.

        Args:
            config: The quadrature configuration to add.

        Returns:
            The index of the added configuration.
        """
        idx = len(self.configs)
        self.configs.append(config)
        return idx

    def build(
        self,
        cell_config_map: npt.NDArray[np.int32] | None = None,
    ) -> "cffi.CData":
        """Build the runintgen_data structure.

        Args:
            cell_config_map: Optional array mapping cell indices to config indices.
                If None, single-config mode is used with config index 0.

        Returns:
            A CFFI pointer to the runintgen_data structure.
        """
        if not self.configs:
            raise ValueError("At least one quadrature configuration is required")

        ffi = self.ffi
        num_configs = len(self.configs)

        # Allocate array of quadrature configs
        c_configs = ffi.new(f"runintgen_quadrature_config[{num_configs}]")
        self._refs.append(c_configs)

        for i, config in enumerate(self.configs):
            # Ensure arrays are contiguous
            points = np.ascontiguousarray(config.points.flatten(), dtype=np.float64)
            weights = np.ascontiguousarray(config.weights, dtype=np.float64)
            self._refs.extend([points, weights])

            c_configs[i].nq = config.nq
            c_configs[i].points = ffi.cast("const double*", points.ctypes.data)
            c_configs[i].weights = ffi.cast("const double*", weights.ctypes.data)

            # Allocate element array
            nelements = len(config.elements)
            c_elements = ffi.new(f"runintgen_element[{nelements}]")
            self._refs.append(c_elements)

            for j, elem in enumerate(config.elements):
                table = np.ascontiguousarray(elem.table.flatten(), dtype=np.float64)
                self._refs.append(table)

                c_elements[j].ndofs = elem.ndofs
                c_elements[j].nderivs = elem.nderivs
                c_elements[j].table = ffi.cast("const double*", table.ctypes.data)

            c_configs[i].nelements = nelements
            c_configs[i].elements = c_elements

        # Allocate main structure
        c_data = ffi.new("runintgen_data*")
        self._refs.append(c_data)

        c_data.num_configs = num_configs
        c_data.configs = c_configs

        if cell_config_map is not None:
            # Multi-config mode
            cell_map = np.ascontiguousarray(cell_config_map, dtype=np.int32)
            self._refs.append(cell_map)
            c_data.active_config = -1
            c_data.cell_config_map = ffi.cast("const int*", cell_map.ctypes.data)
        else:
            # Single-config mode
            c_data.active_config = 0
            c_data.cell_config_map = ffi.NULL

        return c_data

    def build_single_config(self, config_index: int = 0) -> "cffi.CData":
        """Build for single-config mode with explicit config selection.

        Args:
            config_index: Index of the configuration to use.

        Returns:
            A CFFI pointer to the runintgen_data structure.
        """
        if config_index < 0 or config_index >= len(self.configs):
            raise ValueError(f"Invalid config index: {config_index}")

        c_data = self.build(cell_config_map=None)
        c_data.active_config = config_index
        return c_data


def tabulate_element(
    element,  # basix.finite_element
    points: npt.NDArray[np.float64],
    max_deriv_order: int = 1,
) -> ElementTableInfo:
    """Tabulate a basix element at given points.

    Args:
        element: A basix finite element.
        points: Quadrature points, shape [nq, tdim].
        max_deriv_order: Maximum derivative order to tabulate.

    Returns:
        ElementTableInfo with tabulated values.
    """
    # basix.tabulate returns [nderivs, nq, ndofs, ncomps]
    tables = element.tabulate(max_deriv_order, points)

    # For scalar elements, ncomps=1, so squeeze that dimension
    if tables.shape[-1] == 1:
        tables = tables[..., 0]  # [nderivs, nq, ndofs]

    return ElementTableInfo(
        ndofs=element.dim,
        nderivs=tables.shape[0],
        table=tables,
    )


def create_quadrature_config(
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    elements: list,  # List of basix elements
    max_deriv_order: int = 1,
) -> QuadratureConfig:
    """Create a QuadratureConfig by tabulating elements at quadrature points.

    Args:
        points: Quadrature points, shape [nq, tdim].
        weights: Quadrature weights, shape [nq].
        elements: List of basix finite elements to tabulate.
        max_deriv_order: Maximum derivative order.

    Returns:
        QuadratureConfig with tabulated element values.
    """
    config = QuadratureConfig(
        points=np.asarray(points),
        weights=np.asarray(weights),
    )

    for elem in elements:
        config.elements.append(tabulate_element(elem, points, max_deriv_order))

    return config


# CFFI definition string for the runtime structures
# This should be passed to ffi.cdef() before using RuntimeDataBuilder
CFFI_DEF = """
typedef struct {
  int ndofs;
  int nderivs;
  const double* table;
} runintgen_element;

typedef struct {
  int nq;
  const double* points;
  const double* weights;
  int nelements;
  const runintgen_element* elements;
} runintgen_quadrature_config;

typedef struct {
  int num_configs;
  const runintgen_quadrature_config* configs;
  int active_config;
  const int* cell_config_map;
} runintgen_data;
"""
