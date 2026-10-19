from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sage.structure.element import Matrix

    from .field import FiniteField
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
    def __init__(self, field: FiniteField, n: int) -> None:
        self.field = field
        self.n = n

    def random_element(self) -> Isometry:
        return Isometry(
            self.field.random_inv_matrix(self.n),
            self.field.random_inv_matrix(self.n),
            self.field.random_inv_matrix(self.n),
        )
