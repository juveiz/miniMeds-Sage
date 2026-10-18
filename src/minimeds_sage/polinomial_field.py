from sage.all import *



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

    def transform(self, A: Matrix, B: Matrix, C: Matrix) -> "Tensor":
        m = A.ncols()
        S: list[Matrix] = []
        for p in range(self.n):
            S.append(B.transpose() * self.slice(p) * C)

        new_slices: list[Matrix] = []
        for i in range(m):
            Ti = self.field.zero_matrix(m, m)
            for p in range(self.n):
                Ti += A[p, i] * S[p]
            new_slices.append(Ti)

        return Tensor(self.field, m, new_slices)

    def star(self, phi: Isometry) -> "Tensor":
        return self.transform(phi.A, phi.B, phi.C)

    def corank_1_point_rejection(self) -> list:
        while True:
            u = [self.field.random_element() for _ in range(self.n)]
            Au = self.contract_axis(0, u)
            if Au.rank() == self.n - 1:
                return u

    def corank1_walk(
        self, u: list[FiniteRingElement], n: Integer, check_span: bool = True
    ) -> "tuple[Matrix, Matrix, Matrix] | None":
        F = self.field.F
        u_i = vector(F, u)

        LU: list = [u_i]
        LV: list = []
        LW: list = []

        for i in range(1, n + 1):
            K = self.contract_axis(0, u_i).left_kernel()
            if K.dimension() != 1:
                return None
            v_i = K.basis()[0]
            if check_span and self._in_span(LV, v_i):
                return None
            LV.append(v_i)

            K = self.contract_axis(1, v_i).right_kernel()
            if K.dimension() != 1:
                return None
            w_i = K.basis()[0]
            if check_span and self._in_span(LW, w_i):
                return None
            LW.append(w_i)

            K = self.contract_axis(2, w_i).left_kernel()
            if K.dimension() != 1:
                return None
            if i == n:
                break
            u_i = K.basis()[0]
            if check_span and self._in_span(LU, u_i):
                return None
            LU.append(u_i)

        return (
            matrix(F, LU).transpose(),
            matrix(F, LV).transpose(),
            matrix(F, LW).transpose(),
        )

    def _in_span(self, L: list, x) -> bool:
        if not L:
            return False
        return matrix(self.field.F, L + [x]).rank() < len(L) + 1

    def scale(self) -> "tuple[Matrix, Matrix, Matrix] | None":
        F = self.field.F
        l = self.n
        if l < 5:
            return None

        def e(a: int, b: int, c: int) -> FiniteRingElement:
            return self.slice(a)[b, c]

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


if __name__ == "__main__":
    q = Integer(4093)
    n = Integer(25)
    O = finiteFieldTensor(q, n)

    T0 = O.random_tensor()
    phi = O.I.random_element()
    T1 = phi.act(T0)

    sk = phi
    pk = (T0, T1)
    print(sk)
    print(pk)
