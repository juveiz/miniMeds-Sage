from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from sage.all import matrix, vector

if TYPE_CHECKING:
    from sage.structure.element import FieldElement, Matrix

    from .field import FiniteField


class Tensor:
    """3-tensor over GF(q) stored as n slices T(e_a, -, -), each an n x n matrix."""

    def __init__(self, field: FiniteField, n: int, slices: list[Matrix]) -> None:
        self.field = field
        self.n = n
        self.slices = slices

    def __repr__(self) -> str:
        return f"Tensor(n={self.n}, over GF({self.field.q}))"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.n == other.n and self.slices == other.slices

    def slice(self, a: int) -> Matrix:
        return self.slices[a]

    def entry(self, a: int, b: int, c: int) -> FieldElement:
        return self.slices[a][b, c]

    def contract_axis(self, axis: int, v: Sequence[FieldElement]) -> Matrix:
        """Contract v along `axis`; result is the n x n matrix of remaining axes."""
        F = self.field.F
        v = vector(F, v)
        if axis == 0:
            result = self.field.zero_matrix(self.n, self.n)
            for a in range(self.n):
                result += v[a] * self.slice(a)
            return result
        if axis == 1:
            # result[a, k] = sum_j v[j] * T[a][j, k]
            return matrix(F, [v * S for S in self.slices])
        if axis == 2:
            # result[a, j] = sum_k v[k] * T[a][j, k]
            return matrix(F, [S * v for S in self.slices])
        raise ValueError("axis must be 0, 1, or 2")

    def transform(self, A: Matrix, B: Matrix, C: Matrix) -> Tensor:
        """Apply (A, B, C): new_i = sum_p A[p, i] * B^T T_p C."""
        m = A.ncols()
        S = [B.transpose() * self.slice(p) * C for p in range(self.n)]

        new_slices: list[Matrix] = []
        for i in range(m):
            Ti = self.field.zero_matrix(m, m)
            for p in range(self.n):
                Ti += A[p, i] * S[p]
            new_slices.append(Ti)

        return Tensor(self.field, m, new_slices)

    def display(self) -> None:
        for a in range(self.n):
            print(f"T(e_{a}, -, -) =")
            print(self.slice(a))
            print()
