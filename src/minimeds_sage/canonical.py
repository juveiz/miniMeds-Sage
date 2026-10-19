from __future__ import annotations

from typing import TYPE_CHECKING

from sage.all import diagonal_matrix, matrix, vector

if TYPE_CHECKING:
    from sage.structure.element import FieldElement, Matrix

    from .tensor import Tensor


def corank1_point(T: Tensor) -> list[FieldElement]:
    """Rejection-sample u until T(u, -, -) has rank n - 1."""
    while True:
        u = [T.field.random_element() for _ in range(T.n)]
        if T.contract_axis(0, u).rank() == T.n - 1:
            return u


def _in_span(F, L: list, x) -> bool:
    if not L:
        return False
    return matrix(F, L + [x]).rank() < len(L) + 1


def corank1_walk(
    T: Tensor,
    u: list[FieldElement],
    steps: int | None = None,
    check_span: bool = True,
) -> tuple[Matrix, Matrix, Matrix] | None:
    """Walk u -> v -> w -> u' via 1-dim kernels; return (U, V, W) column matrices or None."""
    F = T.field.F
    n = T.n if steps is None else steps
    u_i = vector(F, u)

    LU: list = [u_i]
    LV: list = []
    LW: list = []

    for i in range(1, n + 1):
        K = T.contract_axis(0, u_i).left_kernel()
        if K.dimension() != 1:
            return None
        v_i = K.basis()[0]
        if check_span and _in_span(F, LV, v_i):
            return None
        LV.append(v_i)

        K = T.contract_axis(1, v_i).right_kernel()
        if K.dimension() != 1:
            return None
        w_i = K.basis()[0]
        if check_span and _in_span(F, LW, w_i):
            return None
        LW.append(w_i)

        K = T.contract_axis(2, w_i).left_kernel()
        if K.dimension() != 1:
            return None
        if i == n:
            break
        u_i = K.basis()[0]
        if check_span and _in_span(F, LU, u_i):
            return None
        LU.append(u_i)

    return (
        matrix(F, LU).transpose(),
        matrix(F, LV).transpose(),
        matrix(F, LW).transpose(),
    )


def scale(T: Tensor) -> tuple[Matrix, Matrix, Matrix] | None:
    """Diagonal scaling (f, g, h) normalising fixed entries of T; None if a pivot is 0 or n < 5."""
    F = T.field.F
    l = T.n
    if l < 5:
        return None

    e = T.entry
    d1 = e(0, 1, 0)
    d2 = e(1, 2, 4)
    d3 = e(0, 2, 1)
    d4 = e(1, 0, 1)
    b3 = e(0, 2, 0)
    c5 = e(0, 1, 4)
    if 0 in (d1, d2, d3, d4, b3, c5):
        return None

    f = [F(0)] * l
    g = [F(0)] * l
    h = [F(0)] * l

    h[0] = F(1)
    g[1] = F(1)
    f[0] = 1 / d1
    f[1] = (b3 * c5) / (d1**2 * d2)
    h[1] = b3 / d3
    g[0] = 1 / (f[1] * h[1] * d4)

    for i in range(2, l):
        a_i = e(i, 1, 0)
        b_i = e(0, i, 0)
        c_i = e(0, 1, i)
        if 0 in (a_i, b_i, c_i):
            return None
        f[i] = d1 * f[0] / a_i
        g[i] = d1 * g[1] / b_i
        h[i] = d1 * h[0] / c_i

    return (
        diagonal_matrix(F, f),
        diagonal_matrix(F, g),
        diagonal_matrix(F, h),
    )
