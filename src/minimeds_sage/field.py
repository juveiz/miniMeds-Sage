from __future__ import annotations

from typing import TYPE_CHECKING

from sage.all import GF, matrix, random_matrix

if TYPE_CHECKING:
    from sage.structure.element import FieldElement, Matrix


class FiniteField:
    def __init__(self, q: int) -> None:
        self.q = q
        self.F = GF(q)

    def __repr__(self) -> str:
        return f"FiniteField(GF({self.q}))"

    def random_element(self) -> FieldElement:
        return self.F.random_element()

    def zero_matrix(self, rows: int, cols: int) -> Matrix:
        return matrix(self.F, rows, cols)

    def random_matrix(self, rows: int, cols: int) -> Matrix:
        return random_matrix(self.F, rows, cols)

    def random_inv_matrix(self, n: int) -> Matrix:
        while True:
            M = self.random_matrix(n, n)
            if M.is_invertible():
                return M
