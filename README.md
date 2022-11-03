# taichi-cg
A poisson solver implementing the conjugate-gradient method written in Taichi.

## Introduction
We included several iterative method for solving linear systems, all written in the Taichi language.
- `cgsolver.py`: the conjugate-gradient (CG) method solver class.
- `bicgsolver.py`: the biconjugate-gradient (BiCG) method solver class.

The solvers are all written to solve the same equation:

$$ \nabla ^2 f = \sin(2\pi x) \sin(2\pi y) $$

## Implementation details
We use a matrix-free approach to represent `A` matrix in the `Ax = b` equation. Namely, there is no `A` matrix stored in the
program. Instead, `A` is implicitly represented by defining a linear operator. For example, in the conjugate-gradient method,
we define a Taichi `kernel` to return the result of `A` multiplying a vector `x`:
```python
@ti.kernel
def compute_Ax(self):
    for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                           (self.N_ext, self.N_tot - self.N_ext)):
        self.Ax[i, j] = (4.0 + self.offset) * self.x[i, j] - self.x[
            i + 1, j] - self.x[i - 1, j] - self.x[i, j + 1] - self.x[i, j - 1]
```
This is equivalent to a matrix with diagnonal elements = `4.0 + offset` and neighbor elements = `-1.0`.

## References
- Conjugate-gradient method: https://en.wikipedia.org/wiki/Conjugate_gradient_method
- BiConjugate-gradient method: https://en.wikipedia.org/wiki/Biconjugate_gradient_method
- An Introduction to the Conjugate Gradient Method Without the Agonizing Pain: https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf