from __future__ import annotations

from sage.all import GF, random_matrix

from .isometry import IsometryGroup
from .tensor import Tensor


class TensorSpace:
    """Parameter set (q, n): field GF(q), n x n x n tensors, and their isometry group."""

    def __init__(self, q: int, n: int) -> None:
        self.q = q
        self.n = n
        self.F = GF(q)
        self.group = IsometryGroup(self.F, n)

    def __repr__(self) -> str:
        return f"TensorSpace(q={self.q}, n={self.n})"

    def random_tensor(self) -> Tensor:
        slices = [random_matrix(self.F, self.n, self.n) for _ in range(self.n)]
        return Tensor(self.F, self.n, slices)
