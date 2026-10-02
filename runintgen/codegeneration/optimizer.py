# SPDX-FileCopyrightText: 2025 ONERA
# SPDX-License-Identifier: MIT
#
# This file extends the FFCx LNodes optimizer. See THIRD_PARTY_NOTICES.md for
# dependency and provenance notes.

"""Loop optimizer for generated element tensor code.

FFCx's ``licm`` hoists the factors of ``A[i, j] += f(i) * g(j) * w`` that do not
depend on the inner dof index into one temporary per statement. Bilinear forms
typically produce many statements per element-tensor block that share the inner
factor ``f(i)`` and differ only in the hoisted part, e.g. the 81 products of
reference derivatives in non-affine elasticity, or the 16 products per derivative
table of a hexahedral Hessian. :func:`grouped_licm` sums the hoisted parts of such
statements into one temporary,

    t[j] = sum_k g_k(j) * w_k,      A[i, j] += f(i) * t[j],

so the inner loop runs once per distinct inner factor instead of once per term.

The dependency analysis inspects index expressions recursively, so the optimizer
is also valid for runtime kernels, whose FE tables are flat arrays addressed with
expressions such as ``table[((d * nq + iq) * ndofs + i) * ncomp]``.
"""

from __future__ import annotations

from collections import defaultdict

import ffcx.codegeneration.lnodes as L
from ffcx.codegeneration.optimizer import fuse_loops, fuse_sections


def depends_on(expr: L.LExpr, index: L.Symbol) -> bool:
    """Return whether ``expr`` references the symbol ``index``."""
    if isinstance(expr, L.Symbol):
        return expr == index
    if isinstance(expr, (L.LiteralFloat, L.LiteralInt)):
        return False
    if isinstance(expr, L.ArrayAccess):
        return any(depends_on(i, index) for i in expr.indices)
    if isinstance(expr, (L.NaryOp, L.MathFunction)):
        return any(depends_on(arg, index) for arg in expr.args)
    if isinstance(expr, L.BinOp):
        return depends_on(expr.lhs, index) or depends_on(expr.rhs, index)
    if isinstance(expr, L.PrefixUnaryOp):
        return depends_on(expr.arg, index)
    if isinstance(expr, L.Conditional):
        return any(
            depends_on(e, index) for e in (expr.condition, expr.true, expr.false)
        )
    if isinstance(expr, L.MultiIndex):
        return any(depends_on(s, index) for s in expr.symbols)
    raise NotImplementedError(
        f"Cannot analyse loop dependencies of {type(expr).__name__}."
    )


def _assignments(statement: L.LNode) -> list[L.LNode]:
    """Return the expressions of a statement or statement list."""
    if isinstance(statement, L.StatementList):
        return [s.expr for s in statement.statements]
    return [statement.expr]


def _product(factors: list[L.LExpr]) -> L.LExpr:
    """Return the product of ``factors`` (a single factor is returned as is)."""
    if not factors:
        return L.LiteralFloat(1.0)
    if len(factors) == 1:
        return factors[0]
    return L.Product(factors)


def grouped_licm(section: L.Section, temp_prefix: str = "gt") -> L.Section:
    """Hoist inner-loop invariant factors and sum terms sharing an inner factor.

    The section must hold one doubly nested loop whose inner body consists of
    ``A[...] += product`` statements, as produced by FFCx for rank-2 element
    tensor blocks. Other sections are returned unchanged.
    """
    if not section.statements or L.depth(section.statements[0]) != 2:
        return section
    outer = section.statements[0]
    if not isinstance(outer, L.ForRange) or len(outer.body.statements) != 1:
        return section
    inner = outer.body.statements[0]
    if not isinstance(inner, L.ForRange):
        return section
    if not all(isinstance(b, L.LiteralInt) for b in (outer.begin, outer.end)):
        return section

    # (lhs, inner-dependent factors) -> hoisted factors of each term
    groups: dict[tuple[L.LExpr, tuple[L.LExpr, ...]], list[list[L.LExpr]]] = (
        defaultdict(list)
    )
    for statement in inner.body.statements:
        for assignment in _assignments(statement):
            if not isinstance(assignment, L.AssignAdd):
                return section
            rhs = assignment.rhs
            factors = list(rhs.args) if isinstance(rhs, L.Product) else [rhs]
            varying = tuple(f for f in factors if depends_on(f, inner.index))
            hoisted = [f for f in factors if not depends_on(f, inner.index)]
            groups[(assignment.lhs, varying)].append(hoisted)

    size = outer.end.value - outer.begin.value
    pre_loop: list[L.LNode] = []
    body: list[L.LNode] = []
    for count, ((lhs, varying), terms) in enumerate(groups.items()):
        if len(terms) == 1 and len(terms[0]) <= 1:
            body.append(L.AssignAdd(lhs, _product([*varying, *terms[0]])))
            continue
        temp = L.Symbol(f"{temp_prefix}_{count}", dtype=L.DataType.SCALAR)
        summands = [_product(term) for term in terms]
        value = summands[0] if len(summands) == 1 else L.Sum(summands)
        pre_loop.append(L.ArrayDecl(temp, size, [0]))
        pre_loop.append(
            L.ForRange(
                outer.index,
                outer.begin,
                outer.end,
                [L.Assign(L.ArrayAccess(temp, [outer.index]), value)],
            )
        )
        body.append(
            L.AssignAdd(lhs, _product([*varying, L.ArrayAccess(temp, [outer.index])]))
        )

    loop = L.ForRange(
        outer.index,
        outer.begin,
        outer.end,
        [L.ForRange(inner.index, inner.begin, inner.end, body)],
    )
    annotations = [a for a in section.annotations if a != L.Annotation.licm]
    return L.Section(
        section.name,
        [*pre_loop, loop, *section.statements[1:]],
        section.declarations,
        section.input,
        section.output,
        annotations,
    )


def optimize(code: list[L.LNode]) -> list[L.LNode]:
    """Optimize the sections of one quadrature loop body.

    Same passes as FFCx's ``optimize``, with :func:`grouped_licm` in place of
    FFCx's ``licm``. Temporaries are named per section so that sections never
    redeclare each other's arrays.
    """
    code = fuse_sections(code, "Coefficient")
    code = fuse_sections(code, "Jacobian")
    out: list[L.LNode] = []
    num_licm = 0
    for section in code:
        if isinstance(section, L.Section):
            if L.Annotation.fuse in section.annotations:
                section = fuse_loops(section)
            if L.Annotation.licm in section.annotations:
                section = grouped_licm(section, temp_prefix=f"gt{num_licm}")
                num_licm += 1
        out.append(section)
    return out
