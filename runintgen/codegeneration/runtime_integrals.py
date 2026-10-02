# SPDX-FileCopyrightText: 2025 ONERA
# SPDX-License-Identifier: MIT
#
# This file subclasses and composes FFCx code-generation APIs. See
# THIRD_PARTY_NOTICES.md for dependency and provenance notes.

"""Runtime integral generation using FFCx IR.

This module adapts FFCx's ``IntegralGenerator`` instead of hand-writing form
specific tensor code. Runtime integrals keep the UFCx kernel signature, but
quadrature weights, points, and Basix element handles are read from
``custom_data``. Non-piecewise FE tables are tabulated inside the generated C
kernel through a small Basix C wrapper function pointer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import basix
import ffcx.codegeneration.lnodes as L
import numpy as np
import ufl
from ffcx.codegeneration.access import FFCXBackendAccess
from ffcx.codegeneration.C.formatter import Formatter
from ffcx.codegeneration.definitions import (
    FFCXBackendDefinitions,
    create_quadrature_index,
)
from ffcx.codegeneration.symbols import FFCXBackendSymbols
from ffcx.codegeneration.utils import dtype_to_c_type, dtype_to_scalar_dtype
from ffcx.ir.elementtables import (
    UniqueTableReferenceT,
    get_modified_terminal_element,
)
from ffcx.ir.representation import IntegralIR
from ffcx.ir.representationutils import QuadratureRule

from ..form_metadata import component_element_from_mixed
from ..quadrature_function import (
    QuadratureFunctionInfo,
    is_quadrature_function,
)
from .contraction import Verbatim, chunked_contraction
from .standard_integrals import OptimizedIntegralGenerator


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
    restriction: str | None = None
    point_set: int = 0

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
            "restriction": self.restriction,
            "point_set": self.point_set,
        }


class RuntimeTableRegistry:
    """Assign runtime slots to non-piecewise FFCx table references."""

    def __init__(self, table_metadata: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialise an empty registry."""
        self._name_to_reference_index: dict[str, int] = {}
        self._element_to_slot: dict[tuple[int | str, int], int] = {}
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

    def register(
        self,
        tabledata: UniqueTableReferenceT,
        *,
        integral_type: str,
        restriction: str | None,
    ) -> RuntimeTableReferenceInfo:
        """Register a table reference and return its runtime slot info."""
        if tabledata.has_tensor_factorisation:
            raise NotImplementedError(
                "Runtime integrals do not support FFCx sum-factorized tables yet."
            )

        point_set = (
            1 if integral_type == "interior_facet" and restriction == "-" else 0
        )
        reference_key = f"{tabledata.name}:{point_set}"
        if reference_key in self._name_to_reference_index:
            return self.references[self._name_to_reference_index[reference_key]]

        reference_index = len(self.references)
        metadata = self.table_metadata.get(tabledata.name, {})
        element_key = metadata.get("element_hash")
        if element_key is None:
            element_key = f"table:{tabledata.name}"
        element_slot_key = (element_key, point_set)
        if element_slot_key not in self._element_to_slot:
            self._element_to_slot[element_slot_key] = len(self._element_to_slot)
        element_slot = self._element_to_slot[element_slot_key]
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
            restriction=restriction,
            point_set=point_set,
        )
        self._name_to_reference_index[reference_key] = reference_index
        self.references.append(info)
        return info


def _uses_runtime_table(tabledata: UniqueTableReferenceT, integral_type: str) -> bool:
    """Return whether a FFCx table should be read from a Basix runtime view."""
    if tabledata.ttype in {"ones", "zeros"}:
        return False
    if tabledata.ttype in {"piecewise", "fixed"} and integral_type != "cell":
        return False
    return True


def _varies_at_runtime(
    node_data: dict[str, Any], integral_type: str, affine_geometry: bool
) -> bool:
    """Return whether a terminal node takes different values at runtime points."""
    mt = node_data.get("mt")
    if mt is None:
        return False
    if is_quadrature_function(getattr(mt, "terminal", None)):
        return True
    tabledata = node_data.get("tr")
    if tabledata is None:
        return False
    if affine_geometry and isinstance(mt.terminal, ufl.classes.Jacobian):
        return False
    return _uses_runtime_table(tabledata, integral_type) or tabledata.ttype in (
        "varying",
        "quadrature",
        "uniform",
    )


def _is_affine_simplex_integral(integral_ir: IntegralIR) -> bool:
    """Return whether the integral's mesh has degree-1 simplex cells."""
    for integrand_data in integral_ir.expression.integrand.values():
        factorization = integrand_data.get("factorization")
        if factorization is None:
            continue
        for node_data in factorization.nodes.values():
            mt = node_data.get("mt")
            if mt is not None and isinstance(mt.terminal, ufl.classes.Jacobian):
                domain = ufl.domain.extract_unique_domain(mt.terminal)
                return bool(domain.is_piecewise_linear_simplex_domain())
    return False


def _force_runtime_tables_varying(
    integral_ir: IntegralIR, affine_geometry: bool = False
) -> None:
    """Move runtime-backed terminal dependencies into quadrature scope.

    FFCx classifies tables using the placeholder quadrature rule present during
    IR construction. Runtime rules can contain arbitrary points, so a table that
    looks fixed for the placeholder rule can still vary at runtime, for example
    derivatives of a higher-order coordinate element. Match ffcx-runtime's
    conservative model by treating all runtime-backed table terminals and their
    dependent factorization nodes as varying.

    On affine cells the Jacobian is the same at every point. With
    ``affine_geometry`` it stays cellwise constant: it is computed once, before
    the quadrature loop, and so are all nodes that depend only on it and on
    other constants.
    """
    integral_type = integral_ir.expression.integral_type

    for integrand_data in integral_ir.expression.integrand.values():
        factorization = integrand_data.get("factorization")
        if factorization is None:
            continue

        nodes = factorization.nodes
        if affine_geometry:
            for node_data in nodes.values():
                if node_data.get("status") != "inactive":
                    node_data["status"] = "active"
        pending = [
            node_id
            for node_id, node_data in nodes.items()
            if node_data.get("status") != "inactive"
            and _varies_at_runtime(node_data, integral_type, affine_geometry)
        ]
        seen = set(pending)
        while pending:
            node_id = pending.pop()
            nodes[node_id]["status"] = "varying"
            for dependent in factorization.in_edges.get(node_id, []):
                if dependent in seen:
                    continue
                if nodes[dependent].get("status") == "inactive":
                    continue
                seen.add(dependent)
                pending.append(dependent)
        if affine_geometry:
            for node_data in nodes.values():
                if node_data.get("status") == "active":
                    node_data["status"] = "piecewise"


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
        quadrature_functions: dict[tuple[Any, str | None], QuadratureFunctionInfo],
    ) -> None:
        """Initialise runtime access hooks."""
        super().__init__(entity_type, integral_type, symbols, options)
        self.table_registry = table_registry
        self.quadrature_functions = quadrature_functions

    def coefficient(
        self,
        mt: Any,
        tabledata: UniqueTableReferenceT,
        quadrature_rule: QuadratureRule,
    ) -> L.LExpr:
        """Access a coefficient, redirecting QuadratureFunction terminals."""
        if is_quadrature_function(mt.terminal):
            info = self.quadrature_functions[(mt.terminal, mt.restriction)]
            mte = get_modified_terminal_element(mt)
            if mte is None:
                raise RuntimeError("Could not analyse QuadratureFunction terminal.")
            _, averaged, local_derivatives, flat_component = mte
            if averaged:
                raise NotImplementedError(
                    "Averaged QuadratureFunction access is not supported."
                )
            if any(int(i) != 0 for i in local_derivatives):
                raise NotImplementedError(
                    "Derivatives of QuadratureFunction are not supported. "
                    "Precompute the derivative into another QuadratureFunction."
                )

            component = int(flat_component or 0)
            if component >= info.value_size:
                raise ValueError(
                    f"QuadratureFunction component {component} is outside "
                    f"value_size {info.value_size}."
                )

            iq = L.Symbol(
                self.symbols.quadrature_loop_index.name, dtype=L.DataType.INT
            )
            q0 = L.Symbol("q0", dtype=L.DataType.INT)
            values = L.Symbol(f"q_function_{info.slot}", dtype=L.DataType.REAL)
            return values[
                (q0 + iq) * L.LiteralInt(info.value_size)
                + L.LiteralInt(component)
            ]

        return super().coefficient(mt, tabledata, quadrature_rule)

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

        table_ref = self.table_registry.register(
            tabledata,
            integral_type=self.integral_type,
            restriction=restriction,
        )
        table_symbol = L.Symbol(table_ref.c_symbol, dtype=L.DataType.REAL)
        self.symbols.element_tables[tabledata.name] = table_symbol

        iq = quadrature_index.global_index
        if all(isinstance(n, int) and n == 0 for n in quadrature_index.sizes):
            # Cellwise constant definitions, emitted before the quadrature loop
            # (e.g. the Jacobian of an affine cell), read the first point.
            iq = L.LiteralInt(0)
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


class RuntimeBackendDefinitions(FFCXBackendDefinitions):
    """FFCx definitions with QuadratureFunction interpolation disabled."""

    def coefficient(
        self,
        mt: Any,
        tabledata: UniqueTableReferenceT,
        quadrature_rule: QuadratureRule,
        access: L.Symbol,
    ) -> L.Section | list:
        """Return definition code for coefficients."""
        if is_quadrature_function(mt.terminal):
            return L.Section("QuadratureFunction", [], [], [], [])
        return super().coefficient(mt, tabledata, quadrature_rule, access)


class RuntimeFFCXBackend:
    """FFCx backend assembled from runtime-aware pieces."""

    def __init__(
        self,
        ir: IntegralIR,
        options: dict[str, Any],
        table_registry: RuntimeTableRegistry,
        quadrature_functions: dict[tuple[Any, str | None], QuadratureFunctionInfo],
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
            quadrature_functions,
        )
        self.definitions = RuntimeBackendDefinitions(
            ir.expression.entity_type, ir.expression.integral_type, self.access, options
        )


class RuntimeFFCXIntegralGenerator(OptimizedIntegralGenerator):
    """FFCx integral generator with runtime quadrature/table sources."""

    def __init__(self, ir: IntegralIR, backend: Any) -> None:
        """Initialise."""
        super().__init__(ir, backend)
        self.helpers: list[str] = []

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

    def quadrature_index(self, quadrature_rule: QuadratureRule) -> L.MultiIndex:
        """Return a quadrature loop index whose extent is ``rt_nq``."""
        if quadrature_rule.has_tensor_factors:
            raise NotImplementedError(
                "Runtime integrals do not support tensor-factor quadrature yet."
            )
        iq_symbol = self.backend.symbols.quadrature_loop_index
        return L.MultiIndex(
            [L.Symbol(iq_symbol.name, dtype=L.DataType.INT)],
            [L.Symbol("rt_nq", dtype=L.DataType.INT)],
        )

    def generate_quadrature_loop(
        self, quadrature_rule: QuadratureRule, domain: basix.CellType
    ) -> list[L.LNode]:
        """Generate the quadrature loop, as chunked contraction where possible.

        The option ``runintgen_chunked_contraction=False`` keeps FFCx's loop,
        which updates the element tensor at every point.
        """
        iq = self.quadrature_index(quadrature_rule)
        definitions, intermediates = self.generate_varying_partition(
            quadrature_rule, domain
        )
        tensor_comp, weight_declarations = self.generate_dofblock_partition(
            quadrature_rule, domain
        )
        options = self.backend.access.options
        if not options.get("runintgen_chunked_contraction", True):
            return self.quadrature_loop_code(
                iq, definitions, intermediates, tensor_comp, weight_declarations
            )
        point = L.Symbol(iq.symbols[0].name, dtype=L.DataType.INT)
        # As built by FFCx's dofblock partition
        weight = self.backend.symbols.weights_table(quadrature_rule)[
            create_quadrature_index(
                quadrature_rule, self.backend.symbols.quadrature_loop_index
            ).global_index
        ]
        scalar = np.dtype(options["scalar_type"])
        real = np.dtype(dtype_to_scalar_dtype(scalar))
        chunked = chunked_contraction(
            tensor_sections=tensor_comp,
            definitions=definitions,
            intermediates=intermediates,
            weight_declarations=weight_declarations,
            weight_factors=self._weight_factors(
                quadrature_rule, domain, weight_declarations, weight
            ),
            weight=weight,
            point=point,
            num_points=L.Symbol("rt_nq", dtype=L.DataType.INT),
            c_types={
                L.DataType.REAL: dtype_to_c_type(real),
                L.DataType.SCALAR: dtype_to_c_type(scalar),
            },
            sizes={L.DataType.REAL: real.itemsize, L.DataType.SCALAR: scalar.itemsize},
        )
        if chunked is not None:
            code, helpers = chunked
            for helper in helpers:
                if helper not in self.helpers:
                    self.helpers.append(helper)
            return code
        return self.quadrature_loop_code(
            iq, definitions, intermediates, tensor_comp, weight_declarations
        )

    def _weight_factors(
        self,
        quadrature_rule: QuadratureRule,
        domain: basix.CellType,
        declarations: list[L.VariableDecl],
        weight: L.LExpr,
    ) -> dict[str, tuple[L.LExpr, bool]]:
        """Return fw name -> (f, f cellwise constant) for fw = f * weight."""
        F = self.ir.expression.integrand[(domain, quadrature_rule)]["factorization"]
        constant = {
            symbol.name: F.nodes[key[2]]["status"] == "piecewise"
            for key, symbol in self.temp_symbols.items()
            if key[0] == "fw" and key[1] is quadrature_rule
        }
        factors = {}
        for declaration in declarations:
            value = declaration.value
            args = list(value.args) if isinstance(value, L.Product) else [value]
            # LNodes define == but not !=
            rest = [a for a in args if not a == weight]
            if len(rest) == len(args):
                continue
            f = rest[0] if len(rest) == 1 else (L.Product(rest) if rest else 1.0)
            factors[declaration.symbol.name] = (
                L.as_lexpr(f),
                constant.get(declaration.symbol.name, False),
            )
        return factors


@dataclass
class RuntimeGeneratedKernel:
    """Generated runtime kernel body and table metadata."""

    body: str
    runtime_tables: list[RuntimeTableReferenceInfo]
    quadrature_function_slots: list[int]
    # File-scope C helpers the body calls (each guarded against redefinition)
    helpers: list[str] = field(default_factory=list)


class RuntimeFormatter(Formatter):
    """C formatter that also writes :class:`Verbatim` statements."""

    def __call__(self, obj: L.LNode) -> str:
        """Format an L node."""
        if isinstance(obj, Verbatim):
            return obj.text + "\n"
        return super().__call__(obj)


class RuntimeIntegralGenerator:
    """Generate runtime C code bodies from FFCx integral IR."""

    def __init__(
        self,
        options: dict[str, Any],
        quadrature_functions: list[QuadratureFunctionInfo] | None = None,
    ) -> None:
        """Initialise with FFCx options."""
        self.options = options
        self.quadrature_functions = quadrature_functions or []
        self._q_by_terminal_restriction = {
            (info.terminal, info.restriction): info for info in self.quadrature_functions
        }

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
                element = component_element_from_mixed(element, flat_component)
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
                    value = element.basix_hash()
                    element_hash = None if value is None else int(value)
                elif hasattr(element, "_element") and hasattr(element._element, "hash"):
                    value = element._element.hash()
                    element_hash = None if value is None else int(value)
                elif hasattr(element, "basix_element"):
                    value = element.basix_element.hash()
                    element_hash = None if value is None else int(value)

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
        """Generate a runtime kernel body for one FFCx integral/domain pair.

        The option ``runintgen_affine_geometry`` states that all cells are
        affine, e.g. parallelepipeds of a Cartesian hexahedral mesh. It
        defaults to whether the mesh has degree-1 simplex cells.
        """
        affine_geometry = self.options.get("runintgen_affine_geometry")
        if affine_geometry is None:
            affine_geometry = _is_affine_simplex_integral(integral_ir)
        _force_runtime_tables_varying(integral_ir, bool(affine_geometry))
        table_registry = RuntimeTableRegistry(self._table_metadata(integral_ir))
        backend = RuntimeFFCXBackend(
            integral_ir, self.options, table_registry, self._q_by_terminal_restriction
        )
        generator = RuntimeFFCXIntegralGenerator(integral_ir, backend)
        parts = generator.generate(domain)
        body = RuntimeFormatter(self.options["scalar_type"])(parts)

        used_slots = []
        for integrand_data in integral_ir.expression.integrand.values():
            factorization = integrand_data.get("factorization")
            if factorization is None:
                continue
            for node_data in factorization.nodes.values():
                mt = node_data.get("mt")
                if mt is None or not is_quadrature_function(mt.terminal):
                    continue
                info = self._q_by_terminal_restriction.get(
                    (mt.terminal, mt.restriction)
                )
                if info is not None:
                    used_slots.append(info.slot)

        return RuntimeGeneratedKernel(
            body=body,
            runtime_tables=table_registry.references,
            quadrature_function_slots=sorted(set(used_slots)),
            helpers=generator.helpers,
        )
