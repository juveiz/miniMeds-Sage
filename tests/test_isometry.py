from minimeds_sage import Isometry


def test_action_composes(space):
    T = space.random_tensor()
    phi = space.group.random_element()
    psi = space.group.random_element()
    composed = Isometry(phi.A * psi.A, phi.B * psi.B, phi.C * psi.C)
    assert psi.act(phi.act(T)) == composed.act(T)


def test_random_element_invertible(space):
    phi = space.group.random_element()
    assert phi.A.is_invertible() and phi.B.is_invertible() and phi.C.is_invertible()
