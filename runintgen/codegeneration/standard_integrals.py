# SPDX-FileCopyrightText: 2025 ONERA
# SPDX-License-Identifier: MIT
#
# This file subclasses FFCx code-generation APIs. See THIRD_PARTY_NOTICES.md for
# dependency and provenance notes.

"""FFCx integral generator that uses the runintgen loop optimizer."""

from __future__ import annotations

import basix
import ffcx.codegeneration.lnodes as L
from ffcx.codegeneration.definitions import create_quadrature_index
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.ir.representationutils import QuadratureRule

from .optimizer import optimize


class OptimizedIntegralGenerator(IntegralGenerator):
    """FFCx ``IntegralGenerator`` whose quadrature loops use :func:`optimize`."""

    def quadrature_index(self, quadrature_rule: QuadratureRule) -> L.MultiIndex:
        """Return the quadrature loop index for ``quadrature_rule``."""
        return create_quadrature_index(
            quadrature_rule, self.backend.symbols.quadrature_loop_index
        )

    def generate_quadrature_loop(
        self, quadrature_rule: QuadratureRule, domain: basix.CellType
    ) -> list[L.LNode]:
        """Generate the quadrature loop for ``quadrature_rule``."""
        iq = self.quadrature_index(quadrature_rule)
        definitions, intermediates_0 = self.generate_varying_partition(
            quadrature_rule, domain
        )
        tensor_comp, intermediates_fw = self.generate_dofblock_partition(
            quadrature_rule, domain
        )
        return self.quadrature_loop_code(
            iq, definitions, intermediates_0, tensor_comp, intermediates_fw
        )

    def quadrature_loop_code(
        self,
        iq: L.MultiIndex,
        definitions: list[L.LNode],
        intermediates_0: list[L.LNode],
        tensor_comp: list[L.LNode],
        intermediates_fw: list[L.VariableDecl],
    ) -> list[L.LNode]:
        """Return the quadrature loop over ``iq`` of the given sections."""
        assert all(isinstance(tc, L.Section) for tc in tensor_comp)

        inputs: list[L.Symbol] = []
        for definition in definitions:
            assert isinstance(definition, L.Section)
            inputs += definition.output

        output: list[L.Symbol] = []
        declarations: list[L.VariableDecl] = []
        for fw in intermediates_fw:
            assert isinstance(fw, L.VariableDecl)
            output += [fw.symbol]
            declarations += [L.VariableDecl(fw.symbol, 0)]
            intermediates_0 += [L.Assign(fw.symbol, fw.value)]
        intermediates = [
            L.Section("Intermediates", intermediates_0, declarations, inputs, output)
        ]

        code = optimize(definitions + intermediates + tensor_comp)
        return [L.create_nested_for_loops([iq], code)]
