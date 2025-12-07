"""Runtime integral generator using FFCX backend.

This module provides a specialized generator that derives from FFCX's
IntegralGenerator but treats element tables and quadrature data as
external runtime arrays.

The key differences from standard FFCX code generation:
1. Quadrature weights and points are passed at runtime (not static arrays)
2. FE tables are passed at runtime per unique element
3. The number of quadrature points is dynamic
4. Each element's table contains all derivatives: [nderivs, nq, ndofs]

The runtime kernel signature is:
    void kernel(
        scalar_t* restrict A,
        const scalar_t* restrict w,
        const scalar_t* restrict c,
        const geometry_t* restrict coordinate_dofs,
        const int* restrict entity_local_index,
        const uint8_t* restrict quadrature_permutation,
        void* restrict custom_data  // -> runintgen_data*
    );

The runintgen_data contains:
    - nq, weights, points (quadrature)
    - nelements, elements[] (per-element tables with all derivatives)
"""

from __future__ import annotations

from typing import Any

from ..analysis import ArgumentRole
from ..runtime_tables import (
    DerivativeMapping,
    RuntimeElementMapping,
)


class RuntimeIntegralGenerator:
    """Specialised generator for runtime integrals.

    We use the same machinery as FFCX but treat element tables and
    quadrature-related data as external (runtime) arrays.
    """

    def __init__(self, ir: Any, backend: Any) -> None:
        """Initialize the generator.

        Args:
            ir: FFCX IR (DataIR).
            backend: FFCX backend instance (or None for now).
        """
        self.ir = ir
        self.backend = backend

    def generate_runtime(
        self,
        integral_ir: Any,
        element_mapping: RuntimeElementMapping,
    ) -> str:
        """Generate C code using per-element table structure.

        This version uses the RuntimeElementMapping which stores ONE table
        per unique element, with all derivatives in a single array following
        basix.tabulate output format [nderivs, nq, ndofs].

        Args:
            integral_ir: The integral IR from FFCX.
            element_mapping: Mapping from unique elements to runtime indices.

        Returns:
            A string containing C code for the kernel body.
        """
        parts: list[str] = []

        # Build helper to get derivative index
        def deriv_idx(deriv: tuple[int, ...]) -> int:
            return DerivativeMapping.derivative_to_index_2d(deriv)

        parts.append("  // Runtime integral - per-element tables with all derivatives")
        parts.append("")

        # Find coordinate element (for Jacobian computation)
        coord_elem_idx = None
        coord_ndofs = 3
        for elem in element_mapping.elements:
            if elem.is_coordinate:
                coord_elem_idx = elem.index
                coord_ndofs = elem.ndofs
                break

        # Find argument elements (for test/trial functions)
        test_elem_idx = None
        test_ndofs = 3
        trial_elem_idx = None
        trial_ndofs = 3
        for elem in element_mapping.elements:
            if elem.is_test:
                test_elem_idx = elem.index
                test_ndofs = elem.ndofs
            if elem.is_trial:
                trial_elem_idx = elem.index
                trial_ndofs = elem.ndofs

        # Use test element as generic argument element if both are same
        arg_elem_idx = test_elem_idx
        arg_ndofs = test_ndofs

        parts.append("  // Main quadrature loop")
        parts.append("  for (int iq = 0; iq < nq; ++iq)")
        parts.append("  {")
        parts.append("    const double weight = data->weights[iq];")
        parts.append("")

        # Generate Jacobian computation using coordinate element
        parts.append("    // Compute Jacobian at quadrature point")
        parts.append("    // J[i][j] = sum_k coord_dofs[k*gdim + i] * dphi_k/dX_j")
        parts.append("    double J[2][2] = {{0.0, 0.0}, {0.0, 0.0}};")

        if coord_elem_idx is not None:
            # Derivative indices for d/dX and d/dY
            dx_deriv_idx = deriv_idx((1, 0))
            dy_deriv_idx = deriv_idx((0, 1))

            parts.append("    {")
            parts.append(
                f"      const runintgen_element* coord_elem = "
                f"&data->elements[{coord_elem_idx}];"
            )
            parts.append(f"      const int coord_ndofs = {coord_ndofs};")
            parts.append("      const double* coord_table = coord_elem->table;")
            parts.append("")
            parts.append("      for (int k = 0; k < coord_ndofs; ++k)")
            parts.append("      {")
            parts.append(
                f"        // d/dX derivatives at deriv_idx={dx_deriv_idx}, "
                f"d/dY at deriv_idx={dy_deriv_idx}"
            )
            parts.append(
                f"        const double dphi_dX = coord_table["
                f"{dx_deriv_idx} * nq * coord_ndofs + iq * coord_ndofs + k];"
            )
            parts.append(
                f"        const double dphi_dY = coord_table["
                f"{dy_deriv_idx} * nq * coord_ndofs + iq * coord_ndofs + k];"
            )
            parts.append("        J[0][0] += coordinate_dofs[k * 3 + 0] * dphi_dX;")
            parts.append("        J[1][0] += coordinate_dofs[k * 3 + 1] * dphi_dX;")
            parts.append("        J[0][1] += coordinate_dofs[k * 3 + 0] * dphi_dY;")
            parts.append("        J[1][1] += coordinate_dofs[k * 3 + 1] * dphi_dY;")
            parts.append("      }")
            parts.append("    }")
        else:
            parts.append("    // WARNING: No coordinate element found")

        parts.append("")

        # Compute detJ and inverse Jacobian
        parts.append("    // Compute determinant and inverse Jacobian")
        parts.append("    const double detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];")
        parts.append("    const double inv_detJ = 1.0 / detJ;")
        parts.append("    double Jinv[2][2];")
        parts.append("    Jinv[0][0] = J[1][1] * inv_detJ;")
        parts.append("    Jinv[0][1] = -J[0][1] * inv_detJ;")
        parts.append("    Jinv[1][0] = -J[1][0] * inv_detJ;")
        parts.append("    Jinv[1][1] = J[0][0] * inv_detJ;")
        parts.append("")

        # Generate tensor computation for Laplacian
        parts.append("    // Compute contribution to element tensor")
        parts.append("    // For Laplacian: A[i,j] += (grad phi_i . grad phi_j) * w")
        parts.append(
            "    // Note: weight should already include |detJ| from runtime quadrature"
        )

        if arg_elem_idx is not None:
            dx_deriv_idx = deriv_idx((1, 0))
            dy_deriv_idx = deriv_idx((0, 1))

            parts.append(
                f"    const runintgen_element* arg_elem = "
                f"&data->elements[{arg_elem_idx}];"
            )
            parts.append(f"    const int ndofs = {arg_ndofs};")
            parts.append("    const double* arg_table = arg_elem->table;")
            parts.append("")
            parts.append("    for (int i = 0; i < ndofs; ++i)")
            parts.append("    {")
            parts.append("      // Reference gradients of test function i")
            parts.append(
                f"      const double dphi_i_dX = arg_table["
                f"{dx_deriv_idx} * nq * ndofs + iq * ndofs + i];"
            )
            parts.append(
                f"      const double dphi_i_dY = arg_table["
                f"{dy_deriv_idx} * nq * ndofs + iq * ndofs + i];"
            )
            parts.append("      // Physical gradients: grad = Jinv^T * ref_grad")
            parts.append(
                "      const double grad_i_x = "
                "Jinv[0][0] * dphi_i_dX + Jinv[1][0] * dphi_i_dY;"
            )
            parts.append(
                "      const double grad_i_y = "
                "Jinv[0][1] * dphi_i_dX + Jinv[1][1] * dphi_i_dY;"
            )
            parts.append("")
            parts.append("      for (int j = 0; j < ndofs; ++j)")
            parts.append("      {")
            parts.append("        // Reference gradients of trial function j")
            parts.append(
                f"        const double dphi_j_dX = arg_table["
                f"{dx_deriv_idx} * nq * ndofs + iq * ndofs + j];"
            )
            parts.append(
                f"        const double dphi_j_dY = arg_table["
                f"{dy_deriv_idx} * nq * ndofs + iq * ndofs + j];"
            )
            parts.append(
                "        const double grad_j_x = "
                "Jinv[0][0] * dphi_j_dX + Jinv[1][0] * dphi_j_dY;"
            )
            parts.append(
                "        const double grad_j_y = "
                "Jinv[0][1] * dphi_j_dX + Jinv[1][1] * dphi_j_dY;"
            )
            parts.append("")
            parts.append("        // Accumulate: inner product of gradients")
            parts.append(
                "        A[i * ndofs + j] += "
                "(grad_i_x * grad_j_x + grad_i_y * grad_j_y) * weight;"
            )
            parts.append("      }")
            parts.append("    }")
        else:
            parts.append("    // TODO: Generate tensor computation for this form")
            parts.append("    // No argument element found in mapping")

        parts.append("  }  // end quadrature loop")

        return "\n".join(parts)
