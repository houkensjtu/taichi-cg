import taichi as ti
from math import sqrt

ti_dtype = ti.f32

@ti.data_oriented
class LinearOperator:
    def __init__(self, matvec):
        self.matvec = matvec

    def _matvec(self, x, Ax):
        self.matvec(x, Ax)


def cg(A, b, x, tol=1e-6, maxiter=5000):
    bshape = b.shape
    xshape = x.shape
    if bshape != xshape:
        print(">>> Dimension not matched.")
    else:
        size = bshape
    p = ti.field(dtype=ti_dtype, shape=size)
    r = ti.field(dtype=ti_dtype, shape=size)
    Ap = ti.field(dtype=ti_dtype, shape=size)
    alpha = ti.field(dtype=ti_dtype, shape=())
    beta = ti.field(dtype=ti_dtype, shape=())

    @ti.kernel
    def init():
        for I in ti.grouped(x):
            r[I] = b[I]
            p[I] = 0.0
            Ap[I] = 0.0
            
    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> ti_dtype:
        sum = 0.0
        for I in ti.grouped(p):
            sum += p[I] * q[I]
        return sum

    @ti.kernel
    def update_x():
        for I in ti.grouped(x):
            x[I] += alpha[None] * p[I]
            
    @ti.kernel
    def update_r():
        for I in ti.grouped(r):
            r[I] -= alpha[None] * Ap[I]

    @ti.kernel
    def update_p():
        for I in ti.grouped(p):
            p[I] = r[I] + beta[None] * p[I]

    def solve():
        init()
        initial_rTr = reduce(r, r)
        print(">>> Initial residual =", (initial_rTr))
        old_rTr = initial_rTr
        update_p()
        # -- Main loop --
        for i in range(maxiter):
            A._matvec(p, Ap)  # compute Ap = A x p
            pAp = reduce(p, Ap)
            alpha[None] = old_rTr / pAp
            update_x()
            update_r()
            new_rTr = reduce(r, r)
            if sqrt(new_rTr) < tol:
                print('>>> Conjugate Gradient method converged.')
                print('>>> #iterations', i)
                break
            beta[None] = new_rTr / old_rTr
            update_p()
            old_rTr = new_rTr
            print(f'>>> Iter = {i+1:4}, Residual = {(new_rTr):e}')
        
    solve()
    
