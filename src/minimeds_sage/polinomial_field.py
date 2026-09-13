from sage.all import *

q = Integer(4093)

n = Integer(25)

class polinomialField:

    def __init__(self, q: Integer, n: Integer):
        self.q = q
        self.n = n
        self.F = GF(q)
        self.M = FiniteRankFreeModule(self.F, self.n)
        self.e = self.M.basis('e')

    def random_tensor(self):
        T = self.M.tensor((0,3))
        comp = T.set_comp(self.e)

        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    comp[i,j,k] = self.F.random_element()
        return T

    def corank_1_point_rejection(self,T):
        while True:
            u = self.M([self.F.random_element() for _ in range(self.n)])
            A = T.contract(0,u)
            Acomp = A.comp(self.e)
            Amat = matrix(self.F, self.n, self.n, lambda j,k: Acomp[j, k])
            if Amat.rank() == n - 1:
                return u

pf = polinomialField(q,n)

print(pf.corank_1_point_rejection(pf.random_tensor()))
