from get_param import params,toCuda,toCpu,get_hyperparam,toDType
import matplotlib.pyplot as plt
from dataset_poisson import DatasetPoisson
#from setups_multistep_1_channel import Dataset
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
import time
import os
from derivatives import laplace,laplace_detach
import scipy.sparse as sp
import scipy.sparse.linalg as la
import scipy
import time
from get_param import params,device

from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import cg

def laplace_to_symmetric_matrix(input_bc, mask_bc):
	N = input_bc.shape[0]

	A = sp.diags(-12*np.ones(N*N),0)
	left = sp.diags(2*np.ones(N*N-1), -1)
	right = sp.diags(2*np.ones(N*N-1), 1)
	top = sp.diags(2*np.ones(N*(N-1)), -N)
	bottom = sp.diags(2*np.ones(N*(N-1)), N)
	topleft = sp.diags(1*np.ones(N*(N-1)-1), -N-1)
	topright = sp.diags(1*np.ones(N*(N-1)+1), -N+1)
	botleft = sp.diags(1*np.ones(N*(N-1)+1), N-1)
	botright = sp.diags(1*np.ones(N*(N-1)-1), N+1)
	
	A += left + right + top + bottom + topleft + topright + botleft + botright
	A *= -0.25#0.25

	b = input_bc.reshape(N*N)

	x,y = (mask_bc > 0).nonzero()

	A = sp.csr_matrix(A)
	
	# set rows to 0
	for row in list(x*N+y):
		start = A.indptr[row]
		end = A.indptr[row + 1]
		# Set all entries in this row to zero
		A.data[start:end] = 0
		
	A.eliminate_zeros()
			
	# set columns to 0
	A = A.tocsc()
	for col in list(x*N+y):
		start = A.indptr[col]
		end = A.indptr[col + 1]
		A.data[start:end] = 0

	# Step 4: Eliminate explicit zeros again
	A.eliminate_zeros()

	# Step 5: Convert back to CSR and set diagonal elements to 1
	A = A.tocsr()
		
	A[x*N+y,x*N+y] = 1
	
	A = sp.csc_matrix(A)
	return A, b



dataset = DatasetPoisson(params.height,params.width,1,1,params.average_sequence_length,tell_loss=False)
dataset.reset0_env(0)
#dataset.reset1_env(0)

x_gt = (dataset.x_mesh/dataset.w)*(dataset.y_mesh/dataset.h)

bc_values, bc_mask = dataset.bc_values[0,0].cpu().numpy(), dataset.bc_mask[0,0].cpu().numpy()
bc_values = bc_values*bc_mask


# apply laplace to bc_values to achieve symmetric values
cuda_bc_values = toCuda(torch.tensor(bc_values).unsqueeze(0).unsqueeze(0))
laplace_cuda_bc_values = -laplace(cuda_bc_values).detach()
laplace_bc_values = -laplace_cuda_bc_values[0,0].cpu().numpy()
laplace_bc_values = bc_mask*bc_values+(1-bc_mask)*laplace_bc_values

A,b = laplace_to_symmetric_matrix(laplace_bc_values, bc_mask)


# Build a multigrid preconditioner using PyAMG
ml = smoothed_aggregation_solver(A)
M = ml.aspreconditioner()

# Solve the system using Conjugate Gradient (CG) with the multigrid preconditioner
x, info = cg(A, b, rtol=1e-8, maxiter=100, M=M)

if info == 0:
    print("Conjugate Gradient converged!")
elif info > 0:
    print(f"Conjugate Gradient did not converge within the maximum number of iterations: {info}")
else:
    print("Conjugate Gradient failed.")

# Output the solution
print("Solution x:", x)
