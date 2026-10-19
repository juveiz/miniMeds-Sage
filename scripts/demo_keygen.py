from minimeds_sage import TensorSpace

if __name__ == "__main__":
    space = TensorSpace(q=4093, n=25)

    T0 = space.random_tensor()
    phi = space.group.random_element()
    T1 = phi.act(T0)

    sk = phi
    pk = (T0, T1)
    print(sk)
    print(pk)
