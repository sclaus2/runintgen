"""Compatibility shims for the removed Python pretabulation workflow.

Runtime kernels now receive a ``runintgen_context`` through ``custom_data``.
That context carries reference quadrature rules and opaque Basix element
handles. Generated C calls the element tabulation function pointers directly
for the FFCx table-reference slots it needs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .runtime_data import (
    CFFI_DEF,
    RuntimeBasixElement,
    RuntimeContextBuilder,
    RuntimeEntityMap,
    RuntimeQuadraturePayload,
    RuntimeQuadratureRule,
    RuntimeQuadratureRules,
    RuntimeTableRequest,
    RuntimeTableView,
    as_runtime_quadrature_payload,
    as_runtime_quadrature_rules,
)


class LegacyRuntimeTabulationRemoved(RuntimeError):
    """Raised when callers use the removed Python pretabulation API."""


def _raise_removed(name: str) -> None:
    """Raise a consistent error for removed helpers."""
    raise LegacyRuntimeTabulationRemoved(
        f"runintgen.tabulation.{name} was removed by the runtime FFCx IR "
        "migration. Build a runintgen_context with RuntimeContextBuilder, "
        "or provide the same ABI from the C++ wrapper so generated kernels "
        "can call element tabulate function pointers."
    )


def tabulate_from_table_info(*args, **kwargs):
    """Deprecated placeholder for the removed heuristic table reconstruction."""
    _raise_removed("tabulate_from_table_info")


def prepare_runtime_data(*args, **kwargs):
    """Deprecated placeholder for the removed Python table-builder workflow."""
    _raise_removed("prepare_runtime_data")


def prepare_runtime_data_for_cell(*args, **kwargs):
    """Deprecated placeholder for the removed Python table-builder workflow."""
    _raise_removed("prepare_runtime_data_for_cell")


def compute_detJ_triangle(coords: npt.NDArray[np.float64]) -> float:
    """Compute the affine Jacobian determinant for a triangle.

    This small utility is kept for callers that used it independently of the
    removed table-builder API.
    """
    coords = np.asarray(coords)
    x0, y0 = coords[0, 0], coords[0, 1]
    x1, y1 = coords[1, 0], coords[1, 1]
    x2, y2 = coords[2, 0], coords[2, 1]
    return float((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


__all__ = [
    "CFFI_DEF",
    "LegacyRuntimeTabulationRemoved",
    "RuntimeBasixElement",
    "RuntimeContextBuilder",
    "RuntimeEntityMap",
    "RuntimeQuadraturePayload",
    "RuntimeQuadratureRule",
    "RuntimeQuadratureRules",
    "RuntimeTableRequest",
    "RuntimeTableView",
    "as_runtime_quadrature_payload",
    "as_runtime_quadrature_rules",
    "compute_detJ_triangle",
    "prepare_runtime_data",
    "prepare_runtime_data_for_cell",
    "tabulate_from_table_info",
]
