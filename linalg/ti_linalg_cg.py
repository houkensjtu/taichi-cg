import taichi as ti
import math
from linalg import LinearOperator, cg
import time
ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=True, offline_cache=False)

GRID = 128

x = ti.field(dtype=ti.f32, shape=(GRID, GRID))
b = ti.field(dtype=ti.f32, shape=(GRID, GRID))

# User should define the following kernels:
# to utilize cg()
@ti.kernel
def init():
    for i, j in ti.ndrange(GRID, GRID):
        xl = i / (GRID - 1)
        yl = j / (GRID - 1)
        b[i, j] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)
        x[i, j] = 0.0

@ti.kernel
def compute_Ax(v:ti.template(), mv:ti.template()):
    for i, j in v:
        l = v[i - 1, j] if i - 1 >= 0 else 0.0
        r = v[i + 1, j] if i + 1 <= GRID - 1 else 0.0
        t = v[i, j + 1] if j + 1 <= GRID - 1 else 0.0
        b = v[i, j - 1] if j - 1 >= 0 else 0.0
        mv[i, j] = 4 * v[i, j] - l - r - t - b

A = LinearOperator(compute_Ax)
init()
start = time.time()
cg(A, b, x, maxiter=50000, tol=1e-6)
print(f">>> Time collapsed: {(time.time() - start):e} sec.")
ti.profiler.print_kernel_profiler_info() 
