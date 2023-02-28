import taichi as ti
import numpy as np
import math
import time

ti.init(arch=ti.cuda, kernel_profiler=True, offline_cache=True)

ti_dtype = ti.f32

GRID = 128
SIZE = GRID ** 2

Abuilder = ti.linalg.SparseMatrixBuilder(SIZE, SIZE, max_num_triplets= SIZE*5, dtype=ti_dtype)
b = ti.ndarray(dtype=ti_dtype, shape=SIZE)
x0 = ti.ndarray(dtype=ti_dtype, shape=SIZE)
@ti.kernel
def fill(K: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), x0: ti.types.ndarray()):
    for i,j in ti.ndrange(GRID, GRID):
        row = i * GRID + j
        if j != 0:
            K[row, row - 1] += -1.0
        if j != GRID - 1:
            K[row, row + 1] += -1.0
        if i != 0:
            K[row, row - GRID] += -1.0
        if i != GRID - 1:
            K[row, row + GRID] += -1.0
        K[row, row] += 4.0

    for i, j in ti.ndrange(GRID, GRID):
        idx = i * GRID + j
        xl = i / (GRID - 1)
        yl = j / (GRID - 1)
        b[idx] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)
        x0[idx] = 0.0

fill(Abuilder, b, x0)
A = Abuilder.build(dtype=ti_dtype)
print("Build success ...")

start = time.time()
cg = ti.linalg.CG(A, b, x0, max_iter=5000, atol=1e-6)
x, exit_code = cg.solve()
print(f">>> Time collapsed: {(time.time() - start):e} sec.")
print(">>> Test successfully finished.")
