from __future__ import annotations

from typing import TYPE_CHECKING

from sage.all import random_matrix

if TYPE_CHECKING:
    from sage.rings.finite_rings.finite_field_base import FiniteField
    from sage.structure.element import Matrix

    from .tensor import Tensor


class Isometry:
    def __init__(self, A: Matrix, B: Matrix, C: Matrix) -> None:
        self.A = A
        self.B = B
        self.C = C

    def __repr__(self) -> str:
        return f"Isometry(n={self.A.nrows()}, over GF({self.A.base_ring().order()}))"

    def act(self, T: Tensor) -> Tensor:
        return T.transform(self.A, self.B, self.C)


class IsometryGroup:
    def __init__(self, F: FiniteField, n: int) -> None:
        self.F = F
        self.n = n

    def random_element(self) -> Isometry:
        return Isometry(*(self._random_invertible() for _ in range(3)))

    def _random_invertible(self) -> Matrix:
        while True:
            M = random_matrix(self.F, self.n, self.n)
            if M.is_invertible():
                return M
