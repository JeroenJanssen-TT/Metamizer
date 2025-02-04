import numpy as np
import scipy.sparse as sp
import pyamgcl

# Example sparse matrix A (symmetric positive-definite)
n = 10
A = sp.diags([2, -1, -1], [0, -1, 1], shape=(n, n), format="csr")

# Right-hand side vector b
b = np.ones(n)

# Use Conjugate Gradient (CG) solver with Smoothed Aggregation Preconditioner
amg_solver = pyamgcl.solver(
    A,
    b,
    iterative_solver="cg",                  # Conjugate Gradient
    #precond="smoothed_aggregation" # Smoothed Aggregation Preconditioner
)

# Solve the system
x = amg_solver.solve(b)

# Display the results
print("Solution x:", x)
