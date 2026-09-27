from sage.all import *

q = Integer(4093)
n = Integer(25)


class finiteField:
    def __init__(self, q: Integer) -> None:
        self.q = q
        self.F = GF(q)

    def random_element(self) -> FiniteRingElement:
        return self.F.random_element()

    def zero_matrix(self, rows: Integer, cols: Integer) -> Matrix:
        return matrix(self.F, rows, cols)

    def random_matrix(self, rows: Integer, cols: Integer) -> Matrix:
        return random_matrix(self.F, rows, cols)

    def random_inv_matrix(self, n: Integer) -> Matrix:
        while True:
            M = self.random_matrix(n, n)
            if M.is_invertible():
                return M


class Isometry:
    def __init__(self, A: Matrix, B: Matrix, C: Matrix) -> None:
        self.A = A
        self.B = B
        self.C = C

    def act(self, T: Tensor) -> Tensor:
        return T.star(self)

    def __repr__(self) -> str:
        return f"Isometry(n={self.A.nrows()}, over GF({self.A.base_ring().order()}))"

class Tensor:
    def __init__(self, field: finiteField, n: Integer, slices: list[Matrix]) -> None:
        self.field = field
        self.n = n
        self.slices = slices

    def __repr__(self) -> str:
        return f"Tensor(n={self.n}, over GF({self.field.q}))"
    def slice(self, a: Integer) -> Matrix:
        return self.slices[a]

    def contract_axis(self, axis: Integer, v: list[FiniteRingElement]) -> Matrix:
        if axis == 0:
            return self._contract_axis0(v)
        elif axis == 1:
            return self._contract_axis1(v)
        elif axis == 2:
            return self._contract_axis2(v)
        else:
            raise ValueError("axis must be 0, 1, or 2")

    def _contract_axis0(self, v: list[FiniteRingElement]) -> Matrix:
        result = self.field.zero_matrix(self.n, self.n)
        for a in range(self.n):
            result += v[a] * self.slice(a)
        return result

    def _contract_axis1(self, v: list[FiniteRingElement]) -> Matrix:
        result = self.field.zero_matrix(self.n, self.n)
        for a in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    result[a, k] += v[j] * self.slice(a)[j, k]
        return result

    def _contract_axis2(self, v: list[FiniteRingElement]) -> Matrix:
        result = self.field.zero_matrix(self.n, self.n)
        for a in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    result[a, j] += v[k] * self.slice(a)[j, k]
        return result

    def star(self, phi: Isometry) -> "Tensor":
        A, B, C = phi.A, phi.B, phi.C
        S: list[Matrix] = []
        for p in range(self.n):
            S.append(B.transpose() * self.slice(p) * C)

        new_slices: list[Matrix] = []
        for i in range(self.n):
            Ti = self.field.zero_matrix(self.n, self.n)
            for p in range(self.n):
                Ti += A[p, i] * S[p]
            new_slices.append(Ti)

        return Tensor(self.field, self.n, new_slices)

    def corank_1_point_rejection(self) -> list:
        while True:
            u = [self.field.random_element() for _ in range(self.n)]
            Au = self.contract_axis(0, u)
            if Au.rank() == self.n - 1:
                return u

    def display(self) -> None:
        for a in range(self.n):
            print(f"T(e_{a}, -, -) =")
            print(self.slice(a))
            print()

class IsometryGroup:
    def __init__(self, field: finiteField, n: Integer) -> None:
        self.field = field
        self.n = n

    def random_element(self) -> Isometry:
        A = self.field.random_inv_matrix(self.n)
        B = self.field.random_inv_matrix(self.n)
        C = self.field.random_inv_matrix(self.n)
        return Isometry(A, B, C)


class finiteFieldTensor:
    def __init__(self, q: Integer, n: Integer) -> None:
        self.q = q
        self.n = n
        self.F = finiteField(q)
        self.I = IsometryGroup(self.F, n)

    def random_tensor(self) -> Tensor:
        slices = [self.F.random_matrix(self.n, self.n) for _ in range(self.n)]
        return Tensor(self.F, self.n, slices)


O = finiteFieldTensor(q, n)

T0 = O.random_tensor()
phi = O.I.random_element()
T1 = phi.act(T0)

sk = phi
pk = (T0, T1)
print(sk)
print(pk)
