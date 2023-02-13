import taichi as ti
from linalg import LinearOperator, cg

ti.init(arch=ti.cpu)

x = ti.field(dtype=float, shape=(10, 10))
b = ti.field(dtype=float, shape=(10, 10))

# User should define the following kernels:
# to utilize cg()
@ti.kernel
def init():
    for I in ti.grouped(x):
        x[I] = 2.0
        b[I] = 1.0

@ti.kernel
def compute_Ax(v:ti.template(), mv:ti.template()):
    for i, j in v:
        mv[i, j] = 4 * v[i, j] - v[i - 1, j] \
                 - v[i + 1, j] - v[i, j - 1] - v[i, j + 1]


A = LinearOperator(compute_Ax)
init()
cg(A, b, x)
print(x)
