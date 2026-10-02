# SPDX-FileCopyrightText: 2025 ONERA
# SPDX-License-Identifier: MIT
#
# This file builds on the FFCx LNodes representation. See
# THIRD_PARTY_NOTICES.md for dependency and provenance notes.

"""Quadrature-chunked tensor contraction for runtime kernels.

FFCx emits, inside the quadrature loop, element tensor updates

    A[lhs(i, j)] += fw(q) * P(q, i) * Q(q, j)

for every point q. Each point then sweeps the whole element tensor, whose
entries are strided for blocked (vector) elements, so the updates neither stay
in registers nor vectorise. Here the points are processed in chunks: the test
factors P(q, i) and the trial-side products are stored for the points of a
chunk, and a register-blocked contraction adds them to contiguous per-block
buffers, which are added to A once per entity.

A term whose weight factor is fw(q) = f * w(q) with f constant on the entity,
e.g. a constant coefficient on an affine cell, only needs the point sums

    R[i, j] = sum_q w(q) P(q, i) Q(q, j)

of its (P, Q) pair; f multiplies R when it is added to A, and the pair (Q, P)
reuses R transposed. This saves work when a pair feeds many terms, as in
vector-valued forms such as linear elasticity, where each pair of reference
derivatives feeds every pair of components. Scalar forms such as the Laplacian
have about as many pairs as terms and are cheaper without the point sums, so
of the two plans the one with fewer operations is generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import ffcx.codegeneration.lnodes as L

from .optimizer import depends_on, expression_key, optimize

# Points per chunk, largest first; see _chunk_size.
_CHUNK_SIZES = (32, 16, 8, 4)
# Budget for the accumulators and staged rows of a kernel call. They live on
# the stack of the calling thread, which can be as small as 512 KiB (secondary
# threads on macOS).
_STACK_BYTES = 192 * 1024
# Points per entity assumed when weighing per-entity against per-point work.
_TYPICAL_POINTS = 16
# Register tiles (rows x columns of a block) of the contraction helper, chosen
# when the helper is compiled: for targets with at least 64 doubles of vector
# registers (AVX: 16 x 4, AArch64: 32 x 2), and for others such as the x86-64
# baseline SSE2 (16 x 2 doubles). Measured on Kaby Lake with clang -O2 for
# blocks of 4x4 to 125x125 dofs, these are within 5% (SSE2) and 30% (AVX2) of
# the best tile at each size, whereas a 3x9 tile was 40% slower with SSE2.
_WIDE_TARGETS = "defined(__AVX__) || defined(__aarch64__)"
_WIDE_TILE = (4, 8)
_NARROW_TILE = (4, 4)

_C_TYPE_TAGS = {
    "double": "d",
    "float": "f",
    "double _Complex": "z",
    "float _Complex": "c",
}


class Verbatim(L.Statement):
    """A C statement given as text."""

    def __init__(self, text: str) -> None:
        """Initialise."""
        self.text = text

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        return isinstance(other, Verbatim) and self.text == other.text

    def __hash__(self) -> int:
        """Hash."""
        return hash(self.text)


def _product(factors: list[L.LExpr] | tuple[L.LExpr, ...]) -> L.LExpr:
    """Return the product of ``factors`` (one for an empty list)."""
    if not factors:
        return L.LiteralFloat(1.0)
    if len(factors) == 1:
        return factors[0]
    return L.Product(list(factors))


def _sum(terms: list[L.LExpr]) -> L.LExpr:
    """Return the sum of ``terms``."""
    return terms[0] if len(terms) == 1 else L.Sum(terms)


def _is_real(factors: list[L.LExpr] | tuple[L.LExpr, ...]) -> bool:
    """Return whether all factors are real (FE tables, weights, literals)."""
    return all(f.dtype in (L.DataType.REAL, L.DataType.INT) for f in factors)


@dataclass(frozen=True)
class _Block:
    """Dof loops of one element tensor block: A[lhs(i, j)], 0 <= i < ni etc."""

    i: L.Symbol
    j: L.Symbol
    ni: int
    nj: int


@dataclass
class _Term:
    """One update A[lhs] += (test factors) * (trial factors) * (rest)."""

    block: _Block
    lhs: L.LExpr
    test: tuple[L.LExpr, ...]
    trial: tuple[L.LExpr, ...]
    rest: tuple[L.LExpr, ...]
    # f when rest is a weight factor f * w(q) with f constant on the entity
    entity_factor: L.LExpr | None


def _statements(body: L.StatementList) -> list[L.LNode]:
    out = []
    for statement in body.statements:
        if isinstance(statement, L.StatementList):
            out += [s.expr for s in statement.statements]
        else:
            out.append(statement.expr)
    return out


def _parse_section(
    section: L.Section,
    weight: L.LExpr,
    weight_factors: dict[str, tuple[L.LExpr, bool]],
) -> list[_Term] | None:
    """Split a rank-2 tensor section into terms, or return None."""
    if len(section.statements) != 1 or L.depth(section.statements[0]) != 2:
        return None
    outer = section.statements[0]
    if not isinstance(outer, L.ForRange) or len(outer.body.statements) != 1:
        return None
    inner = outer.body.statements[0]
    if not isinstance(inner, L.ForRange):
        return None
    for loop in (outer, inner):
        if not (
            isinstance(loop.index, L.Symbol)
            and isinstance(loop.begin, L.LiteralInt)
            and isinstance(loop.end, L.LiteralInt)
            and loop.begin.value == 0
        ):
            return None

    block = _Block(inner.index, outer.index, inner.end.value, outer.end.value)
    terms = []
    for assignment in _statements(inner.body):
        if not isinstance(assignment, L.AssignAdd):
            return None
        rhs = assignment.rhs
        factors = list(rhs.args) if isinstance(rhs, L.Product) else [rhs]
        test, trial, rest = [], [], []
        for f in factors:
            on_i, on_j = depends_on(f, block.i), depends_on(f, block.j)
            if on_i and on_j:
                return None
            (test if on_i else trial if on_j else rest).append(f)

        entity_factor = None
        if len(rest) == 1:
            if rest[0] == weight:
                entity_factor = L.LiteralFloat(1.0)
            elif isinstance(rest[0], L.Symbol) and rest[0].name in weight_factors:
                f, constant = weight_factors[rest[0].name]
                entity_factor = f if constant else None
        if entity_factor is not None and not (_is_real(test) and _is_real(trial)):
            entity_factor = None
        terms.append(
            _Term(
                block,
                assignment.lhs,
                tuple(test),
                tuple(trial),
                tuple(rest),
                entity_factor,
            )
        )
    return terms


def _type_tag(c_type: str) -> str:
    return _C_TYPE_TAGS.get(c_type, "".join(ch for ch in c_type if ch.isalnum()))


def _tiled_contraction(
    ni: int, nj: int, acc_c: str, x_c: str, y_c: str, tile: tuple[int, int]
) -> str:
    """Return C statements adding X[k][:]^T Y[k][:] to acc in register tiles."""
    ti, tj = min(tile[0], ni), min(tile[1], nj)

    def tile_code(i0: str, mi: int, j0: str, mj: int) -> str:
        return (
            f"{{\n"
            f"  {acc_c} t[{mi}][{mj}];\n"
            f"  for (int a = 0; a < {mi}; ++a)\n"
            f"    for (int b = 0; b < {mj}; ++b)\n"
            f"      t[a][b] = acc[({i0} + a) * {nj} + {j0} + b];\n"
            f"  for (int k = 0; k < nk; ++k)\n"
            f"  {{\n"
            f"    const {x_c}* restrict x = X + k * {ni} + {i0};\n"
            f"    const {y_c}* restrict y = Y + k * {nj} + {j0};\n"
            f"    for (int a = 0; a < {mi}; ++a)\n"
            f"      for (int b = 0; b < {mj}; ++b)\n"
            f"        t[a][b] += x[a] * y[b];\n"
            f"  }}\n"
            f"  for (int a = 0; a < {mi}; ++a)\n"
            f"    for (int b = 0; b < {mj}; ++b)\n"
            f"      acc[({i0} + a) * {nj} + {j0} + b] = t[a][b];\n"
            f"}}\n"
        )

    def columns(i0: str, mi: int) -> str:
        full = (nj // tj) * tj
        code = ""
        if full:
            code += (
                f"for (int j0 = 0; j0 < {full}; j0 += {tj})\n"
                f"{tile_code(i0, mi, 'j0', tj)}"
            )
        if nj > full:
            code += tile_code(i0, mi, str(full), nj - full)
        return code

    full = (ni // ti) * ti
    body = ""
    if full:
        body += (
            f"for (int i0 = 0; i0 < {full}; i0 += {ti})\n{{\n{columns('i0', ti)}}}\n"
        )
    if ni > full:
        body += f"{{\n{columns(str(full), ni - full)}}}\n"
    return body


def _contraction_helper(
    ni: int, nj: int, acc_c: str, x_c: str, y_c: str
) -> tuple[str, str]:
    """Return name and C source of acc[ni][nj] += X[k][:]^T Y[k][:] over k."""
    tag = _type_tag(acc_c) + _type_tag(x_c) + _type_tag(y_c)
    name = f"runintgen_contract_{ni}x{nj}_{tag}"
    wide = _tiled_contraction(ni, nj, acc_c, x_c, y_c, _WIDE_TILE)
    narrow = _tiled_contraction(ni, nj, acc_c, x_c, y_c, _NARROW_TILE)
    if wide == narrow:
        body = wide
    else:
        body = f"#if {_WIDE_TARGETS}\n{wide}#else\n{narrow}#endif\n"
    guard = name.upper()
    source = (
        f"#ifndef {guard}\n"
        f"#define {guard}\n"
        f"static inline void {name}({acc_c}* restrict acc, const int nk,\n"
        f"    const {x_c}* restrict X, const {y_c}* restrict Y)\n"
        f"{{\n{body}}}\n"
        f"#endif\n"
    )
    return name, source


@dataclass
class _Row:
    """Values of ``value`` for 0 <= ``index`` < ``size`` at each chunk point."""

    symbol: L.Symbol
    size: int
    value: L.LExpr
    index: L.Symbol
    # Summands of value, for the operation count
    terms: int = 1


@dataclass
class _Source:
    """An accumulator added, times ``factor``, to an element tensor block."""

    factor: L.LExpr | None
    accumulator: tuple
    transposed: bool = False


@dataclass
class _Plan:
    """Buffers and contractions of one quadrature loop."""

    rows: dict[tuple, _Row] = field(default_factory=dict)
    # per-entity accumulators: key -> (symbol, block of its layout)
    accumulators: dict[tuple, tuple[L.Symbol, _Block]] = field(default_factory=dict)
    # contractions: (accumulator key, test row key, trial row key)
    contractions: list[tuple[tuple, tuple, tuple]] = field(default_factory=list)
    # epilogue: lhs key -> (lhs, block, sources)
    updates: dict[tuple, tuple[L.LExpr, _Block, list[_Source]]] = field(
        default_factory=dict
    )

    def row(self, key, prefix, dtype, size, value, index, terms=1) -> tuple:
        if key not in self.rows:
            symbol = L.Symbol(f"{prefix}{len(self.rows)}", dtype=dtype)
            self.rows[key] = _Row(symbol, size, value, index, terms)
        return key

    def accumulator(self, key, dtype, block) -> tuple:
        if key not in self.accumulators:
            symbol = L.Symbol(f"rt_acc{len(self.accumulators)}", dtype=dtype)
            self.accumulators[key] = (symbol, block)
        return key

    def update(self, lhs: L.LExpr, block: _Block, source: _Source) -> None:
        self.updates.setdefault(expression_key(lhs), (lhs, block, []))[2].append(source)

    def operations(self) -> float:
        """Multiply-adds per point, with per-entity work spread over points."""
        per_point = sum(
            self.accumulators[acc][1].ni * self.accumulators[acc][1].nj
            for acc, _, _ in self.contractions
        )
        per_point += sum(row.size * row.terms for row in self.rows.values())
        per_entity = sum(
            block.ni * block.nj * len(sources)
            for _, block, sources in self.updates.values()
        )
        return per_point + per_entity / _TYPICAL_POINTS


def _plan(terms: list[_Term], weight: L.LExpr, pairs: bool) -> _Plan:
    """Assign staged rows, accumulators and contractions to the terms.

    With ``pairs``, terms with an entity factor use the point sums of their
    (test, trial) pair; the other terms are summed per (lhs, test factors).
    """
    plan = _Plan()
    real, scalar = L.DataType.REAL, L.DataType.SCALAR

    def test_row(term: _Term, key: tuple) -> tuple:
        b = term.block
        dtype = real if _is_real(term.test) else scalar
        return plan.row(("x", key), "rt_x", dtype, b.ni, _product(term.test), b.i)

    groups: dict[tuple, tuple[L.LExpr, list[_Term]]] = {}
    for term in terms:
        b = term.block
        # The dof index is anonymised, so test and trial factors compare equal
        # when they are the same function of their index
        test = (b.ni, expression_key(_product(term.test), b.i))
        trial = (b.nj, expression_key(_product(term.trial), b.j))
        if pairs and term.entity_factor is not None:
            key, mirror = ("sum", test, trial), ("sum", trial, test)
            if key not in plan.accumulators and mirror in plan.accumulators:
                plan.update(term.lhs, b, _Source(term.entity_factor, mirror, True))
                continue
            if key not in plan.accumulators:
                x = test_row(term, test)
                y = plan.row(
                    ("wy", trial),
                    "rt_y",
                    real,
                    b.nj,
                    _product([weight, *term.trial]),
                    b.j,
                )
                plan.accumulator(key, real, b)
                plan.contractions.append((key, x, y))
            plan.update(term.lhs, b, _Source(term.entity_factor, key))
        else:
            x = test_row(term, test)
            groups.setdefault((expression_key(term.lhs), x), (term.lhs, []))[1].append(
                term
            )

    for (lhs_key, x), (lhs, group) in groups.items():
        b = group[0].block
        hoisted = [_product([*t.trial, *t.rest]) for t in group]
        c = plan.row(
            ("c", lhs_key, x), "rt_c", scalar, b.nj, _sum(hoisted), b.j, len(hoisted)
        )
        acc = ("block", lhs_key)
        if acc not in plan.accumulators:
            plan.accumulator(acc, scalar, b)
            plan.update(lhs, b, _Source(None, acc))
        plan.contractions.append((acc, x, c))
    return plan


def _chunk_size(plan: _Plan, sizes: dict[L.DataType, int]) -> int | None:
    """Return the points per chunk, or None if the buffers do not fit."""
    entity = sum(b.ni * b.nj * sizes[s.dtype] for s, b in plan.accumulators.values())
    per_point = sum(r.size * sizes[r.symbol.dtype] for r in plan.rows.values())
    for size in _CHUNK_SIZES:
        if entity + size * per_point <= _STACK_BYTES:
            return size
    return None


def chunked_contraction(
    *,
    tensor_sections: list[L.LNode],
    definitions: list[L.LNode],
    intermediates: list[L.LNode],
    weight_declarations: list[L.VariableDecl],
    weight_factors: dict[str, tuple[L.LExpr, bool]],
    weight: L.LExpr,
    point: L.Symbol,
    num_points: L.Symbol,
    c_types: dict[L.DataType, str],
    sizes: dict[L.DataType, int],
) -> tuple[list[L.LNode], list[str]] | None:
    """Return the chunked quadrature loop and its C helpers, or None.

    Args:
        tensor_sections: FFCx rank-2 tensor computation sections.
        definitions: Per-point definitions (coefficients, geometry).
        intermediates: Per-point intermediate assignments.
        weight_declarations: Declarations ``fw = f * weight`` of weight factors.
        weight_factors: Weight factor name -> (f, f constant on the entity).
        weight: Quadrature weight at the current point.
        point: Quadrature point index symbol used by the expressions.
        num_points: Number of points of the entity's rule.
        c_types: C type of real and scalar data.
        sizes: Size in bytes of real and scalar data.
    """
    if not tensor_sections:
        return None
    terms: list[_Term] = []
    for section in tensor_sections:
        if not isinstance(section, L.Section):
            return None
        parsed = _parse_section(section, weight, weight_factors)
        if parsed is None:
            return None
        terms += parsed
    if not terms:
        return None

    plans = [_plan(terms, weight, pairs=False)]
    if any(t.entity_factor is not None for t in terms):
        plans.append(_plan(terms, weight, pairs=True))
    candidates = [
        (plan.operations(), chunk, plan)
        for plan in plans
        if (chunk := _chunk_size(plan, sizes)) is not None
    ]
    if not candidates:
        return None
    _, chunk, plan = min(candidates, key=lambda c: c[0])

    # Weight factors that the staged rows still use
    used = set()
    for row in plan.rows.values():
        for decl in weight_declarations:
            if depends_on(row.value, decl.symbol):
                used.add(decl.symbol.name)
    point_statements = [
        s
        for s in intermediates
        if not (
            isinstance(s, L.Assign)
            and isinstance(s.lhs, L.Symbol)
            and s.lhs.name in weight_factors
            and s.lhs.name not in used
        )
    ]
    point_statements += [
        L.Assign(d.symbol, d.value)
        for d in weight_declarations
        if d.symbol.name in used
    ]
    declarations = [
        L.VariableDecl(d.symbol, 0)
        for d in weight_declarations
        if d.symbol.name in used
    ]
    intermediate_section = L.Section(
        "Intermediates", point_statements, declarations, [], []
    )

    # Staged rows for the points of one chunk
    first = L.Symbol("rt_q0", dtype=L.DataType.INT)
    count = L.Symbol("rt_nqc", dtype=L.DataType.INT)
    local = L.Sub(point, first)
    staging = [
        L.ForRange(
            row.index,
            0,
            row.size,
            [
                L.Assign(
                    row.symbol[L.Sum([L.Product([local, row.size]), row.index])],
                    row.value,
                )
            ],
        )
        for row in plan.rows.values()
    ]
    point_loop = L.ForRange(
        point,
        first,
        L.Sum([first, count]),
        optimize([*definitions, intermediate_section])
        + [L.Section("Staging", staging, [], [], [])],
    )

    helpers: dict[str, str] = {}
    calls = []
    for acc_key, test_key, trial_key in plan.contractions:
        acc, block = plan.accumulators[acc_key]
        x = plan.rows[test_key].symbol
        y = plan.rows[trial_key].symbol
        name, source = _contraction_helper(
            block.ni, block.nj, c_types[acc.dtype], c_types[x.dtype], c_types[y.dtype]
        )
        helpers[name] = source
        calls.append(Verbatim(f"{name}({acc.name}, rt_nqc, {x.name}, {y.name});"))

    chunk_index = L.Symbol("rt_chunk", dtype=L.DataType.INT)
    remaining = L.Sub(num_points, first)
    chunk_loop = L.ForRange(
        chunk_index,
        0,
        L.Div(L.Sum([num_points, L.LiteralInt(chunk - 1)]), L.LiteralInt(chunk)),
        [
            L.VariableDecl(first, L.Product([L.LiteralInt(chunk), chunk_index])),
            L.VariableDecl(
                count,
                L.Conditional(
                    L.LT(remaining, L.LiteralInt(chunk)), remaining, L.LiteralInt(chunk)
                ),
            ),
            *[L.ArrayDecl(r.symbol, [chunk * r.size]) for r in plan.rows.values()],
            point_loop,
            *calls,
        ],
    )

    accumulators = [
        L.ArrayDecl(symbol, [block.ni * block.nj], [0])
        for symbol, block in plan.accumulators.values()
    ]
    epilogue = []
    for lhs, block, sources in plan.updates.values():
        values = []
        for source in sources:
            symbol, layout = plan.accumulators[source.accumulator]
            # A transposed accumulator has the layout of the block (j, i)
            row, column = (
                (block.j, block.i) if source.transposed else (block.i, block.j)
            )
            value = symbol[L.Sum([L.Product([row, layout.nj]), column])]
            values.append(
                value if source.factor is None else L.Product([source.factor, value])
            )
        epilogue.append(
            L.ForRange(
                block.i,
                0,
                block.ni,
                [L.ForRange(block.j, 0, block.nj, [L.AssignAdd(lhs, _sum(values))])],
            )
        )

    # Section statements form one scoped block, so the loops of several
    # quadrature rules in one kernel each have their own accumulators
    code = [
        L.Section(
            "Tensor Computation", [*accumulators, chunk_loop, *epilogue], [], [], []
        )
    ]
    return code, list(helpers.values())
