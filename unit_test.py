from sage.all import (
    Integer,
    diagonal_matrix,
    identity_matrix,
    matrix,
    set_random_seed,
    vector,
)

import pytest

from minimeds_sage.polinomial_field import (
    Isometry,
    Tensor,
    finiteField,
    finiteFieldTensor,
)

# Toy (insecure) parameters
Q = Integer(101)
N = Integer(6)
L = Integer(5)
MAX_TRIES = 50


@pytest.fixture(autouse=True)
def seed():
    set_random_seed(0)


@pytest.fixture
def O() -> finiteFieldTensor:
    return finiteFieldTensor(Q, N)


@pytest.fixture
def T(O: finiteFieldTensor) -> Tensor:
    return O.random_tensor()


@pytest.fixture
def phi(O: finiteFieldTensor) -> Isometry:
    return O.I.random_element()


def entry(T: Tensor, a: int, b: int, c: int):
    return T.slice(a)[b, c]


def eval_form(T: Tensor, x, y, z):
    # T(x, y, z) = sum_{a,b,c} x_a y_b z_c T[a,b,c]
    return sum(
        x[a] * y[b] * z[c] * entry(T, a, b, c)
        for a in range(T.n)
        for b in range(T.n)
        for c in range(T.n)
    )


def walk_with_retry(T: Tensor, l: Integer) -> tuple:
    for _ in range(MAX_TRIES):
        u = T.corank_1_point_rejection()
        walk = T.corank1_walk(u, l)
        if walk is not None:
            return u, walk
    pytest.fail("corank1_walk failed MAX_TRIES times")


def walk_and_scale_with_retry(T: Tensor, l: Integer) -> tuple:
    # Alg 8 Commit "Repeat ... Until": resample u until walk and Scale succeed
    for _ in range(MAX_TRIES):
        u = T.corank_1_point_rejection()
        walk = T.corank1_walk(u, l)
        if walk is None:
            continue
        R = T.transform(*walk)
        fgh = R.scale()
        if fgh is not None:
            return u, walk, R, fgh
    pytest.fail("corank1_walk + scale failed MAX_TRIES times")


def e_canon(T: Tensor, u, l: Integer) -> "Tensor | None":
    # Alg 5 E_canon(T, u)
    walk = T.corank1_walk(u, l)
    if walk is None:
        return None
    R = T.transform(*walk)
    fgh = R.scale()
    if fgh is None:
        return None
    return R.transform(*fgh)


def random_inv_diagonal(O: finiteFieldTensor, m: Integer) -> "Matrix":
    F = O.F.F
    return diagonal_matrix(F, [F.random_element() or F(1) for _ in range(m)])


def proportional(x, y) -> bool:
    # same projective point: x, y non-zero and rank [x y] == 1
    return x != 0 and y != 0 and matrix([x, y]).rank() == 1


# ============================================================
# Phase 2 — finite field and matrices
# ============================================================
class TestFiniteField:
    def test_field_order(self):
        F = finiteField(Q)
        assert F.F.order() == Q
        assert F.random_element().parent() is F.F

    def test_zero_matrix(self):
        F = finiteField(Q)
        Z = F.zero_matrix(3, 4)
        assert Z.dimensions() == (3, 4)
        assert Z.is_zero()

    def test_random_matrix(self):
        F = finiteField(Q)
        M = F.random_matrix(3, 4)
        assert M.dimensions() == (3, 4)
        assert M.base_ring() is F.F

    def test_random_inv_matrix(self):
        F = finiteField(Q)
        for _ in range(10):
            assert F.random_inv_matrix(N).is_invertible()


# ============================================================
# Phase 2 — tensor representation and contractions
# ============================================================
class TestTensor:
    def test_random_tensor_shape(self, T: Tensor):
        assert T.n == N
        assert len(T.slices) == N
        for S in T.slices:
            assert S.dimensions() == (N, N)
            assert S.base_ring().order() == Q

    def test_contract_axis0(self, O: finiteFieldTensor, T: Tensor):
        # M[b,c] = sum_a u_a T[a,b,c] = T(u, b, c)
        u = vector(O.F.F, [O.F.random_element() for _ in range(N)])
        M = T.contract_axis(0, u)
        for b in range(N):
            for c in range(N):
                assert M[b, c] == sum(u[a] * entry(T, a, b, c) for a in range(N))

    def test_contract_axis1(self, O: finiteFieldTensor, T: Tensor):
        # M[a,c] = sum_b v_b T[a,b,c] = T(a, v, c)
        v = vector(O.F.F, [O.F.random_element() for _ in range(N)])
        M = T.contract_axis(1, v)
        for a in range(N):
            for c in range(N):
                assert M[a, c] == sum(v[b] * entry(T, a, b, c) for b in range(N))

    def test_contract_axis2(self, O: finiteFieldTensor, T: Tensor):
        # M[a,b] = sum_c w_c T[a,b,c] = T(a, b, w)
        w = vector(O.F.F, [O.F.random_element() for _ in range(N)])
        M = T.contract_axis(2, w)
        for a in range(N):
            for b in range(N):
                assert M[a, b] == sum(w[c] * entry(T, a, b, c) for c in range(N))

    def test_contract_invalid_axis(self, O: finiteFieldTensor, T: Tensor):
        u = [O.F.random_element() for _ in range(N)]
        with pytest.raises(ValueError):
            T.contract_axis(3, u)

    def test_contract_basis_vector_is_slice(self, O: finiteFieldTensor, T: Tensor):
        e = vector(O.F.F, [0] * N)
        e[2] = 1
        assert T.contract_axis(0, e) == T.slice(2)


# ============================================================
# Phase 2 — group action (A, B, C) ⋆ T  (paper Eq. (1))
# ============================================================
class TestStar:
    def test_star_entrywise(self, T: Tensor, phi: Isometry):
        # ((A,B,C) ⋆ T)[i,j,k] = sum_{p,r,s} A[p,i] B[r,j] C[s,k] T[p,r,s]
        A, B, C = phi.A, phi.B, phi.C
        T2 = T.star(phi)
        for i, j, k in [(0, 0, 0), (1, 2, 3), (N - 1, 0, N - 1), (3, 5, 1)]:
            expected = sum(
                A[p, i] * B[r, j] * C[s, k] * entry(T, p, r, s)
                for p in range(N)
                for r in range(N)
                for s in range(N)
            )
            assert entry(T2, i, j, k) == expected

    def test_star_as_form(self, O: finiteFieldTensor, T: Tensor, phi: Isometry):
        # ((A,B,C) ⋆ T)(x, y, z) = T(Ax, By, Cz)
        F = O.F.F
        x = vector(F, [O.F.random_element() for _ in range(N)])
        y = vector(F, [O.F.random_element() for _ in range(N)])
        z = vector(F, [O.F.random_element() for _ in range(N)])
        T2 = T.star(phi)
        assert eval_form(T2, x, y, z) == eval_form(
            T, phi.A * x, phi.B * y, phi.C * z
        )

    def test_act_equals_star(self, T: Tensor, phi: Isometry):
        assert phi.act(T).slices == T.star(phi).slices

    def test_identity_action(self, O: finiteFieldTensor, T: Tensor):
        I = identity_matrix(O.F.F, N)
        assert T.star(Isometry(I, I, I)).slices == T.slices

    def test_composition(self, O: finiteFieldTensor, T: Tensor):
        # phi2 ⋆ (phi1 ⋆ T) = (A1 A2, B1 B2, C1 C2) ⋆ T
        phi1 = O.I.random_element()
        phi2 = O.I.random_element()
        lhs = T.star(phi1).star(phi2)
        rhs = T.star(Isometry(phi1.A * phi2.A, phi1.B * phi2.B, phi1.C * phi2.C))
        assert lhs.slices == rhs.slices

    def test_inverse(self, T: Tensor, phi: Isometry):
        inv = Isometry(phi.A.inverse(), phi.B.inverse(), phi.C.inverse())
        assert T.star(phi).star(inv).slices == T.slices

    def test_keygen_relation(self, O: finiteFieldTensor):
        # Alg 8 Keygen: T1 = (A1, B1, C1) ⋆ T0
        T0 = O.random_tensor()
        sk = O.I.random_element()
        T1 = sk.act(T0)
        assert T1.n == T0.n
        assert T1.slices != T0.slices


# ============================================================
# Phase 2 — generalized action (A, B, C) ⋆_{n->m} T  (A, B, C: n×m)
# ============================================================
class TestTransform:
    def test_square_equals_star(self, T: Tensor, phi: Isometry):
        assert T.transform(phi.A, phi.B, phi.C).slices == T.star(phi).slices

    def test_rectangular_shape(self, O: finiteFieldTensor, T: Tensor):
        A, B, C = (O.F.random_matrix(N, L) for _ in range(3))
        R = T.transform(A, B, C)
        assert R.n == L
        assert len(R.slices) == L
        for S in R.slices:
            assert S.dimensions() == (L, L)

    def test_rectangular_entrywise(self, O: finiteFieldTensor, T: Tensor):
        # R[i,j,k] = T(A e_i, B e_j, C e_k)
        A, B, C = (O.F.random_matrix(N, L) for _ in range(3))
        R = T.transform(A, B, C)
        for i, j, k in [(0, 0, 0), (1, 2, 3), (L - 1, 0, L - 1), (4, 1, 2)]:
            assert entry(R, i, j, k) == eval_form(
                T, A.column(i), B.column(j), C.column(k)
            )

    def test_composition_rectangular(self, O: finiteFieldTensor, T: Tensor):
        # (A2,B2,C2) ⋆ ((A1,B1,C1) ⋆ T) = (A1 A2, B1 B2, C1 C2) ⋆ T
        A1, B1, C1 = (O.F.random_matrix(N, L) for _ in range(3))
        A2, B2, C2 = (O.F.random_matrix(L, L) for _ in range(3))
        lhs = T.transform(A1, B1, C1).transform(A2, B2, C2)
        rhs = T.transform(A1 * A2, B1 * B2, C1 * C2)
        assert lhs.slices == rhs.slices

    def test_identity_columns_give_subtensor(self, O: finiteFieldTensor, T: Tensor):
        # first L columns of I_n pick out T[:L, :L, :L]
        E = identity_matrix(O.F.F, N)[:, :L]
        R = T.transform(E, E, E)
        for a in range(L):
            for b in range(L):
                for c in range(L):
                    assert entry(R, a, b, c) == entry(T, a, b, c)


# ============================================================
# Phase 2 — naive corank-1 rejection sampler
# ============================================================
class TestCorank1Sampler:
    def test_is_corank1(self, T: Tensor):
        for _ in range(5):
            u = T.corank_1_point_rejection()
            assert len(u) == N
            assert T.contract_axis(0, u).rank() == N - 1

    def test_hint_transport(self, T: Tensor, phi: Isometry):
        # u corank-1 for T  =>  A^{-1} u corank-1 for (A,B,C) ⋆ T  (proof of Thm 3)
        u = vector(T.corank_1_point_rejection())
        u2 = phi.A.inverse() * u
        assert T.star(phi).contract_axis(0, u2).rank() == N - 1


# ============================================================
# Phase 3 — Corank1Walk (Alg 6, shortened per Def 7)
# ============================================================
class TestCorank1Walk:
    @pytest.mark.parametrize("l", [Integer(1), L, N])
    def test_shape(self, T: Tensor, l: Integer):
        _, (LU, LV, LW) = walk_with_retry(T, l)
        assert LU.dimensions() == (N, l)
        assert LV.dimensions() == (N, l)
        assert LW.dimensions() == (N, l)

    def test_first_column_is_u(self, T: Tensor):
        u, (LU, _, _) = walk_with_retry(T, L)
        assert LU.column(0) == vector(u)

    @pytest.mark.parametrize("l", [L, N])
    def test_kernel_relations(self, T: Tensor, l: Integer):
        _, (LU, LV, LW) = walk_with_retry(T, l)
        for i in range(l):
            u_i, v_i, w_i = LU.column(i), LV.column(i), LW.column(i)
            # Alg 6 l.4: T(u_i, v_i, -) = 0
            assert (v_i * T.contract_axis(0, u_i)).is_zero()
            # Alg 6 l.7: T(-, v_i, w_i) = 0
            assert (T.contract_axis(1, v_i) * w_i).is_zero()
            # Alg 6 l.12: T(u_{i+1}, -, w_i) = 0
            if i + 1 < l:
                assert (LU.column(i + 1) * T.contract_axis(2, w_i)).is_zero()

    @pytest.mark.parametrize("l", [L, N])
    def test_corank1_along_walk(self, T: Tensor, l: Integer):
        _, (LU, LV, LW) = walk_with_retry(T, l)
        for i in range(l):
            assert T.contract_axis(0, LU.column(i)).rank() == N - 1
            assert T.contract_axis(1, LV.column(i)).rank() == N - 1
            assert T.contract_axis(2, LW.column(i)).rank() == N - 1

    @pytest.mark.parametrize("l", [L, N])
    def test_linear_independence(self, T: Tensor, l: Integer):
        # Alg 6 l.5/8/13: no vector in span of previous ones
        _, (LU, LV, LW) = walk_with_retry(T, l)
        assert LU.rank() == l
        assert LV.rank() == l
        assert LW.rank() == l

    def test_deterministic(self, T: Tensor):
        u, walk = walk_with_retry(T, L)
        assert T.corank1_walk(u, L) == walk

    def test_fails_on_zero_point(self, O: finiteFieldTensor, T: Tensor):
        assert T.corank1_walk([O.F.F(0)] * N, L) is None

    def test_fails_on_non_corank1_point(self, O: finiteFieldTensor):
        # T with all slices zero except T[0] = I: T(u,-,-) = u_0 I has rank n or 0
        F = O.F.F
        slices = [identity_matrix(F, N)] + [O.F.zero_matrix(N, N) for _ in range(N - 1)]
        T = Tensor(O.F, N, slices)
        u = [F(1)] + [F(0)] * (N - 1)
        assert T.contract_axis(0, u).rank() == N
        assert T.corank1_walk(u, L) is None

    @pytest.mark.parametrize("l", [L, N])
    def test_isometry_invariance(self, T: Tensor, phi: Isometry, l: Integer):
        # Proof of Thm 3, Eq. (3): walk on (phi ⋆ T, A^{-1} u) gives
        # L'_U = A^{-1} L_U, L'_V = B^{-1} L_V, L'_W = C^{-1} L_W (projectively)
        u, (LU, LV, LW) = walk_with_retry(T, l)
        T2 = T.star(phi)
        u2 = phi.A.inverse() * vector(u)
        walk2 = T2.corank1_walk(list(u2), l)
        assert walk2 is not None
        LU2, LV2, LW2 = walk2

        Ai, Bi, Ci = phi.A.inverse(), phi.B.inverse(), phi.C.inverse()
        for i in range(l):
            assert proportional(LU2.column(i), Ai * LU.column(i))
            assert proportional(LV2.column(i), Bi * LV.column(i))
            assert proportional(LW2.column(i), Ci * LW.column(i))


# ============================================================
# Phase 3 — Scale (Alg 7)
# ============================================================
class TestScale:
    def test_output_shape(self, T: Tensor):
        _, _, R, (F, G, H) = walk_and_scale_with_retry(T, L)
        for D in (F, G, H):
            assert D.dimensions() == (L, L)
            assert D.is_diagonal()
            assert D.is_invertible()

    def test_fixed_degrees_of_freedom(self, T: Tensor):
        # Alg 7 l.2: h1 = 1, g2 = 1
        _, _, R, (F, G, H) = walk_and_scale_with_retry(T, L)
        assert H[0, 0] == 1
        assert G[1, 1] == 1

    def test_normalized_entries(self, T: Tensor):
        # after (F,G,H) ⋆ R all entries used by Scale become 1
        _, _, R, fgh = walk_and_scale_with_retry(T, L)
        Rs = R.transform(*fgh)
        assert entry(Rs, 0, 1, 0) == 1           # d1
        assert entry(Rs, 1, 2, 4) == 1           # d2
        assert entry(Rs, 0, 2, 1) == 1           # d3
        assert entry(Rs, 1, 0, 1) == 1           # d4
        for i in range(2, L):
            assert entry(Rs, i, 1, 0) == 1       # a_i
            assert entry(Rs, 0, i, 0) == 1       # b_j
            assert entry(Rs, 0, 1, i) == 1       # c_k

    def test_invariant_under_diagonal_rescaling(self, O: finiteFieldTensor, T: Tensor):
        # Scale removes the choice of representative of each walk vector
        _, _, R, fgh = walk_and_scale_with_retry(T, L)
        D1, D2, D3 = (random_inv_diagonal(O, L) for _ in range(3))
        R2 = R.transform(D1, D2, D3)
        fgh2 = R2.scale()
        assert fgh2 is not None
        assert R2.transform(*fgh2).slices == R.transform(*fgh).slices

    def test_fails_when_too_small(self):
        O4 = finiteFieldTensor(Q, Integer(4))
        assert O4.random_tensor().scale() is None

    @pytest.mark.parametrize("pos", [(0, 1, 0), (1, 2, 4), (0, 2, 1), (1, 0, 1), (3, 1, 0)])
    def test_fails_on_zero_entry(self, pos: tuple):
        R = finiteFieldTensor(Q, L).random_tensor()
        a, b, c = pos
        R.slice(a)[b, c] = 0
        assert R.scale() is None


# ============================================================
# Phase 3 — E_canon (Alg 5) invariance, Theorem 3
# ============================================================
class TestECanon:
    @pytest.mark.parametrize("l", [L, N])
    def test_canonical_form_invariant(self, T: Tensor, phi: Isometry, l: Integer):
        # E_canon(T, u) == E_canon((A,B,C) ⋆ T, A^{-1} u)
        u, _, _, _ = walk_and_scale_with_retry(T, l)
        T_c = e_canon(T, u, l)
        T2 = T.star(phi)
        u2 = list(phi.A.inverse() * vector(u))
        T2_c = e_canon(T2, u2, l)
        assert T_c is not None
        assert T2_c is not None
        assert T_c.n == l
        assert T2_c.slices == T_c.slices
