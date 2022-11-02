# taichi-cg
A poisson solver implementing the conjugate-gradient method written in Taichi.

## Introduction
We included several iterative method for solving linear systems, all written in the Taichi language.
`cgsolver.py`: the conjugate-gradient (CG) method solver class.
`bicgsolver.py`: the biconjugate-gradient (BiCG) method solver class.

The solvers are all written to solve the same equation:

$$ \nabla ^2 f = \sin(2\pi x) \sin(2\pi y) $$

## Implementation details
We use a matrix-free approach to represent `A` matrix in the `Ax = b` equation.