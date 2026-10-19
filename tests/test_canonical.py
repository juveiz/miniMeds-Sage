from minimeds_sage import corank1_point, corank1_walk, scale


def test_corank1_point_rank(space):
    T = space.random_tensor()
    u = corank1_point(T)
    assert T.contract_axis(0, u).rank() == space.n - 1


def test_corank1_walk_shape(space):
    T = space.random_tensor()
    for _ in range(20):
        res = corank1_walk(T, corank1_point(T))
        if res is not None:
            for M in res:
                assert M.nrows() == space.n
            return


def test_scale_small_n_returns_none():
    from minimeds_sage import TensorSpace

    assert scale(TensorSpace(q=101, n=4).random_tensor()) is None
