from sage.all import identity_matrix, vector


def _naive_contract(T, axis, v):
    n = T.n
    R = T.field.zero_matrix(n, n)
    for a in range(n):
        for j in range(n):
            for k in range(n):
                if axis == 0:
                    R[j, k] += v[a] * T.entry(a, j, k)
                elif axis == 1:
                    R[a, k] += v[j] * T.entry(a, j, k)
                else:
                    R[a, j] += v[k] * T.entry(a, j, k)
    return R


def test_contract_matches_definition(space):
    T = space.random_tensor()
    v = [space.field.random_element() for _ in range(space.n)]
    for axis in range(3):
        assert T.contract_axis(axis, v) == _naive_contract(T, axis, v)


def test_contract_accepts_vector(space):
    T = space.random_tensor()
    v = [space.field.random_element() for _ in range(space.n)]
    assert T.contract_axis(1, vector(space.field.F, v)) == T.contract_axis(1, v)


def test_transform_identity(space):
    T = space.random_tensor()
    I = identity_matrix(space.field.F, space.n)
    assert T.transform(I, I, I) == T
