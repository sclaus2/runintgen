"""Runtime integral generation using FFCx IR.

This module adapts FFCx's ``IntegralGenerator`` instead of hand-writing form
specific tensor code. Runtime integrals keep the UFCx kernel signature, but
quadrature weights, points, and Basix element handles are read from
``custom_data``. Non-piecewise FE tables are tabulated inside the generated C
kernel through a small Basix C wrapper function pointer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import basix
import ffcx.codegeneration.lnodes as L
from ffcx.codegeneration.access import FFCXBackendAccess
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.definitions import FFCXBackendDefinitions
from ffcx.codegeneration.integral_generator import IntegralGenerator
from ffcx.codegeneration.symbols import FFCXBackendSymbols
from ffcx.ir.elementtables import (
    UniqueTableReferenceT,
    get_modified_terminal_element,
)
from ffcx.ir.representation import IntegralIR
from ffcx.ir.representationutils import QuadratureRule


@dataclass(frozen=True)
class RuntimeTableReferenceInfo:
    """Runtime representation of one FFCx table reference."""

    reference_index: int
    slot: int
    name: str
    c_symbol: str
    shape: tuple[int, int, int, int]
    offset: int | None
    block_size: int | None
    ttype: str | None
    is_uniform: bool
    is_permuted: bool
    element_index: int
    element_hash: int | None = None
    averaged: str | None = None
    derivative_counts: tuple[int, ...] = ()
    derivative_index: int = 0
    flat_component: int | None = None
    role: str | None = None
    terminal_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable table description."""
        return {
            "reference_index": self.reference_index,
            "slot": self.slot,
            "name": self.name,
            "c_symbol": self.c_symbol,
            "shape": list(self.shape),
            "offset": self.offset,
            "block_size": self.block_size,
            "ttype": self.ttype,
            "is_uniform": self.is_uniform,
            "is_permuted": self.is_permuted,
            "element_index": self.element_index,
            "element_hash": self.element_hash,
            "averaged": self.averaged,
            "derivative_counts": list(self.derivative_counts),
            "derivative_index": self.derivative_index,
            "flat_component": self.flat_component,
            "role": self.role,
            "terminal_index": self.terminal_index,
        }


class RuntimeTableRegistry:
    """Assign runtime slots to non-piecewise FFCx table references."""

    def __init__(self, table_metadata: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialise an empty registry."""
        self._name_to_reference_index: dict[str, int] = {}
        self._element_to_slot: dict[int | str, int] = {}
        self.references: list[RuntimeTableReferenceInfo] = []
        self.table_metadata = table_metadata or {}

    @staticmethod
    def _derivative_index(counts: tuple[int, ...]) -> int:
        """Return the Basix derivative axis index for derivative counts."""
        if not counts:
            return 0
        if len(counts) == 1:
            return int(basix.index(counts[0]))
        if len(counts) == 2:
            return int(basix.index(counts[0], counts[1]))
        if len(counts) == 3:
            return int(basix.index(counts[0], counts[1], counts[2]))
        raise NotImplementedError(
            "Runtime integrals only support Basix derivative counts up to "
            "topological dimension 3."
        )

    def register(self, tabledata: UniqueTableReferenceT) -> RuntimeTableReferenceInfo:
        """Register a table reference and return its runtime slot info."""
        if tabledata.has_tensor_factorisation:
            raise NotImplementedError(
                "Runtime integrals do not support FFCx sum-factorized tables yet."
            )

        if tabledata.name in self._name_to_reference_index:
            return self.references[self._name_to_reference_index[tabledata.name]]

        reference_index = len(self.references)
        metadata = self.table_metadata.get(tabledata.name, {})
        element_key = metadata.get("element_hash")
        if element_key is None:
            element_key = f"table:{tabledata.name}"
        if element_key not in self._element_to_slot:
            self._element_to_slot[element_key] = len(self._element_to_slot)
        element_slot = self._element_to_slot[element_key]
        c_symbol = f"rt_element_{element_slot}"
        derivative_counts = tuple(metadata.get("derivative_counts", ()))
        info = RuntimeTableReferenceInfo(
            reference_index=reference_index,
            slot=element_slot,
            name=tabledata.name,
            c_symbol=c_symbol,
            shape=tuple(int(i) for i in tabledata.values.shape),
            offset=tabledata.offset,
            block_size=tabledata.block_size,
            ttype=tabledata.ttype,
            is_uniform=tabledata.is_uniform,
            is_permuted=tabledata.is_permuted,
            element_index=element_slot,
            element_hash=metadata.get("element_hash"),
            averaged=metadata.get("averaged"),
            derivative_counts=derivative_counts,
            derivative_index=self._derivative_index(derivative_counts),
            flat_component=metadata.get("flat_component"),
            role=metadata.get("role"),
            terminal_index=metadata.get("terminal_index"),
        )
        self._name_to_reference_index[tabledata.name] = reference_index
        self.references.append(info)
        return info


def _uses_runtime_table(tabledata: UniqueTableReferenceT, integral_type: str) -> bool:
    """Return whether a FFCx table should be read from a Basix runtime view."""
    if tabledata.ttype in {"ones", "zeros"}:
        return False
    if tabledata.ttype in {"piecewise", "fixed"} and integral_type != "cell":
        return False
    return True


def _force_runtime_tables_varying(integral_ir: IntegralIR) -> None:
    """Move runtime-backed terminal dependencies into quadrature scope.

    FFCx classifies tables using the placeholder quadrature rule present during
    IR construction. Runtime rules can contain arbitrary points, so a table that
    looks fixed for the placeholder rule can still vary at runtime, for example
    derivatives of a higher-order coordinate element. Match ffcx-runtime's
    conservative model by treating all runtime-backed table terminals and their
    dependent factorization nodes as varying.
    """
    integral_type = integral_ir.expression.integral_type

    for integrand_data in integral_ir.expression.integrand.values():
        factorization = integrand_data.get("factorization")
        if factorization is None:
            continue

        pending: list[int] = []
        seen: set[int] = set()
        for node_id, node_data in factorization.nodes.items():
            tabledata = node_data.get("tr")
            if tabledata is None:
                continue
            if not _uses_runtime_table(tabledata, integral_type):
                continue
            if node_data.get("status") == "inactive":
                continue
            pending.append(node_id)
            seen.add(node_id)

        while pending:
            node_id = pending.pop()
            factorization.nodes[node_id]["status"] = "varying"
            for dependent in factorization.in_edges.get(node_id, []):
                if dependent in seen:
                    continue
                if factorization.nodes[dependent].get("status") == "inactive":
                    continue
                seen.add(dependent)
                pending.append(dependent)


class RuntimeBackendSymbols(FFCXBackendSymbols):
    """FFCx symbols redirected to runtime view aliases."""

    def weights_table(self, quadrature_rule: QuadratureRule) -> L.Symbol:
        """Return the runtime weights pointer symbol."""
        return L.Symbol("rt_weights", dtype=L.DataType.REAL)

    def points_table(self, quadrature_rule: QuadratureRule) -> L.Symbol:
        """Return the runtime reference points pointer symbol."""
        return L.Symbol("rt_points", dtype=L.DataType.REAL)


class RuntimeBackendAccess(FFCXBackendAccess):
    """FFCx backend access with runtime FE table lookups."""

    def __init__(
        self,
        entity_type: str,
        integral_type: str,
        symbols: RuntimeBackendSymbols,
        options: dict[str, Any],
        table_registry: RuntimeTableRegistry,
    ) -> None:
        """Initialise runtime access hooks."""
        super().__init__(entity_type, integral_type, symbols, options)
        self.table_registry = table_registry

    def table_access(
        self,
        tabledata: UniqueTableReferenceT,
        entity_type: str,
        restriction: str | None,
        quadrature_index: L.MultiIndex,
        dof_index: L.MultiIndex,
    ) -> tuple[L.LExpr, list[L.Symbol]]:
        """Access an FE table from the runtime view.

        Ones/zeros tables stay static exactly as in FFCx. Other supported table
        references are exposed as raw Basix tabulations flattened as
        ``[derivative][point][dof][component]``. Multiple FFCx table references
        that come from the same Basix element share one runtime pointer.
        """
        if not _uses_runtime_table(tabledata, self.integral_type):
            return super().table_access(
                tabledata, entity_type, restriction, quadrature_index, dof_index
            )

        table_ref = self.table_registry.register(tabledata)
        table_symbol = L.Symbol(table_ref.c_symbol, dtype=L.DataType.REAL)
        self.symbols.element_tables[tabledata.name] = table_symbol

        iq = quadrature_index.global_index
        ic = dof_index.global_index
        derivative = L.LiteralInt(table_ref.derivative_index)
        component = L.LiteralInt(0)
        rt_nq = L.Symbol("rt_nq", dtype=L.DataType.INT)
        raw_num_dofs = L.Symbol(
            f"{table_ref.c_symbol}_num_dofs", dtype=L.DataType.INT
        )
        num_components = L.Symbol(
            f"{table_ref.c_symbol}_num_components", dtype=L.DataType.INT
        )

        raw_dof: L.LExpr = ic

        flat_index = ((derivative * rt_nq + iq) * raw_num_dofs + raw_dof)
        flat_index = flat_index * num_components + component
        return table_symbol[flat_index], [table_symbol]


class RuntimeFFCXBackend:
    """FFCx backend assembled from runtime-aware pieces."""

    def __init__(
        self,
        ir: IntegralIR,
        options: dict[str, Any],
        table_registry: RuntimeTableRegistry,
    ) -> None:
        """Initialise runtime backend."""
        coefficient_numbering = ir.expression.coefficient_numbering
        coefficient_offsets = ir.expression.coefficient_offsets
        original_constant_offsets = ir.expression.original_constant_offsets

        self.symbols = RuntimeBackendSymbols(
            coefficient_numbering, coefficient_offsets, original_constant_offsets
        )
        self.access = RuntimeBackendAccess(
            ir.expression.entity_type,
            ir.expression.integral_type,
            self.symbols,
            options,
            table_registry,
        )
        self.definitions = FFCXBackendDefinitions(
            ir.expression.entity_type, ir.expression.integral_type, self.access, options
        )


class RuntimeFFCXIntegralGenerator(IntegralGenerator):
    """FFCx integral generator with runtime quadrature/table sources."""

    def generate_quadrature_tables(
        self, domain: basix.CellType, _expression: Any | None = None
    ) -> list[L.LNode]:
        """Runtime kernels never emit static quadrature weight tables."""
        return []

    def generate_element_tables(self, domain: basix.CellType) -> list[L.LNode]:
        """Emit only static tables that cannot be backed by Basix runtime views."""
        parts: list[L.LNode] = []
        tables = self.ir.expression.unique_tables[domain]
        table_types = self.ir.expression.unique_table_types[domain]
        table_names = [
            name
            for name in sorted(tables)
            if not _uses_runtime_table(
                UniqueTableReferenceT(
                    name,
                    tables[name],
                    False,
                    table_types[name],
                ),
                self.ir.expression.integral_type,
            )
        ]

        for name in table_names:
            parts += self.declare_table(name, tables[name])

        return L.commented_code_list(
            parts,
            [
                "Static FE tables",
                "Runtime FE tables are supplied through custom_data per Basix element",
            ],
        )

    def generate_quadrature_loop(
        self, quadrature_rule: QuadratureRule, domain: basix.CellType
    ) -> list[L.LNode]:
        """Generate a quadrature loop whose extent is ``rt_nq``."""
        if quadrature_rule.has_tensor_factors:
            raise NotImplementedError(
                "Runtime integrals do not support tensor-factor quadrature yet."
            )

        definitions, intermediates_0 = self.generate_varying_partition(
            quadrature_rule, domain
        )
        tensor_comp, intermediates_fw = self.generate_dofblock_partition(
            quadrature_rule, domain
        )
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

        iq_symbol = self.backend.symbols.quadrature_loop_index
        iq = L.MultiIndex(
            [L.Symbol(iq_symbol.name, dtype=L.DataType.INT)],
            [L.Symbol("rt_nq", dtype=L.DataType.INT)],
        )

        # FFCx's loop optimizer assumes static FE table semantics when deciding
        # which products can be hoisted across dof loops. Runtime table accesses
        # depend on dynamic quadrature/table pointers, so keep the unoptimized
        # section structure until a runtime-aware optimizer exists.
        code = definitions + intermediates + tensor_comp
        return [L.create_nested_for_loops([iq], code)]


@dataclass
class RuntimeGeneratedKernel:
    """Generated runtime kernel body and table metadata."""

    body: str
    runtime_tables: list[RuntimeTableReferenceInfo]


class RuntimeIntegralGenerator:
    """Generate runtime C code bodies from FFCx integral IR."""

    def __init__(self, options: dict[str, Any]) -> None:
        """Initialise with FFCx options."""
        self.options = options

    def _table_metadata(self, integral_ir: IntegralIR) -> dict[str, dict[str, Any]]:
        """Extract FFCx table metadata needed by the runtime wrapper."""
        metadata: dict[str, dict[str, Any]] = {}
        expr_ir = integral_ir.expression

        for integrand_data in expr_ir.integrand.values():
            factorization = integrand_data.get("factorization")
            if factorization is None:
                continue

            for node_data in factorization.nodes.values():
                mt = node_data.get("mt")
                tr = node_data.get("tr")
                if mt is None or tr is None:
                    continue

                mte = get_modified_terminal_element(mt)
                if mte is None:
                    continue

                element, averaged, local_derivatives, flat_component = mte
                terminal = mt.terminal
                role = type(terminal).__name__.lower()
                terminal_index: int | None = None

                if hasattr(terminal, "number"):
                    number = terminal.number()
                    role = "test" if number == 0 else "trial"
                    terminal_index = int(number)
                elif terminal in expr_ir.coefficient_numbering:
                    role = "coefficient"
                    terminal_index = int(expr_ir.coefficient_numbering[terminal])
                elif "Jacobian" in type(terminal).__name__:
                    role = "geometry"
                    terminal_index = 0
                elif "SpatialCoordinate" in type(terminal).__name__:
                    role = "geometry"
                    terminal_index = 0

                element_hash = None
                if hasattr(element, "basix_hash"):
                    element_hash = int(element.basix_hash())
                elif hasattr(element, "_element") and hasattr(element._element, "hash"):
                    element_hash = int(element._element.hash())

                metadata[tr.name] = {
                    "element_hash": element_hash,
                    "averaged": averaged,
                    "derivative_counts": tuple(int(i) for i in local_derivatives),
                    "flat_component": (
                        int(flat_component) if flat_component is not None else None
                    ),
                    "role": role,
                    "terminal_index": terminal_index,
                }

        return metadata

    def generate_runtime(
        self,
        integral_ir: IntegralIR,
        domain: basix.CellType,
    ) -> RuntimeGeneratedKernel:
        """Generate a runtime kernel body for one FFCx integral/domain pair."""
        _force_runtime_tables_varying(integral_ir)
        table_registry = RuntimeTableRegistry(self._table_metadata(integral_ir))
        backend = RuntimeFFCXBackend(integral_ir, self.options, table_registry)
        generator = RuntimeFFCXIntegralGenerator(integral_ir, backend)
        parts = generator.generate(domain)
        body = Formatter(self.options["scalar_type"])(parts)

        return RuntimeGeneratedKernel(
            body=body,
            runtime_tables=table_registry.references,
        )
