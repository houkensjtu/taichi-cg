from cgsolver import CGPoissonSolver
from bicgsolver import BICGPoissonSolver

import taichi as ti
import numpy as np
import time

ti.init(arch=ti.cpu, default_fp=ti.f64)

psize = 512
offset = 0.1 # Offset the diagnoal of A matrix with amount = offset

# Solve in Taichi using custom CG
# A is implicitly represented in compute_Ap()
now = time.time()
print('>>> Solving using CGPoissonSolver...')
cgsolver = CGPoissonSolver(psize, 1e-16, offset, quiet=True) # quiet=False to print residual
cgsolver.solve()
print(f'>>> Time spent using CGPoissonSolver: {time.time() - now:4.2f} sec')

# Solve in Taichi using custom BICG
now = time.time()
print('>>> Solving using BICGPoissonSolver...')
bicgsolver = BICGPoissonSolver(psize, 1e-16, offset, quiet=True)
bicgsolver.solve()
print(f'>>> Time spent using BICGPoissonSolver: {time.time() - now:4.2f} sec')

# Compare the residuals: norm(r) where r = Ax - b
residual_cg = cgsolver.check_solution()
residual_bicg = bicgsolver.check_solution()
print('>>> Comparing the residual norm(Ax-b)...')
print(f'>>> Residual CGPoissonSolver: {residual_cg:4.2e}')
print(f'>>> Residual BICGPoissonSolver: {residual_bicg:4.2e}')
