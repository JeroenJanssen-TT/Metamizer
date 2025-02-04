from dataset_diffusion import DatasetDiffusion
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_param import params,device
from derivatives import dx
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta
from derivatives import dx,dy,laplace

h,w = params.height,params.width
params.D_range = 5#0.1
params.iterations_per_timestep = 1000
dt = params.dt

dataset = DatasetDiffusion(h,w,dataset_size=1,batch_size=1,average_sequence_length=99999999999,iterations_per_timestep=params.iterations_per_timestep,D_range=params.D_range)
dataset.reset0_env(0)

x = torch.zeros(1,1,h,w,requires_grad = True,device=device)

#optim = SGD([x],lr=0.01)
#optim = Adam([x],lr=0.1) # faster but worse minimum
optim = Adam([x],lr=0.0001) # slower but slightly better minimum
#optim = AdamW([x],lr=0.01)
#optim = RMSprop([x],lr=0.01)
#optim = Adagrad([x],lr=1)
#optim = Adadelta([x],lr=0.5)

for i in range(100000):
	
	grads,_ = dataset.ask()
	
	if (i+1)%100 == 0:
		plt.clf()
		
		# plt u / f / bc_mask / bc_values
		plt.subplot(2,3,1)
		plt.imshow(dataset.c_new[0,0].detach().cpu())
		plt.colorbar()
		plt.title("c")
		
		plt.subplot(2,3,2)
		plt.imshow(grads[0,0].detach().cpu())
		plt.colorbar()
		plt.title("grads")
		
		plt.subplot(2,3,3)
		plt.imshow(dataset.R[0,0].detach().cpu())
		plt.colorbar()
		plt.title("f")
		
		plt.subplot(2,3,6)
		plt.imshow(dx(dataset.c_new)[0,0].detach().cpu())
		plt.colorbar()
		plt.title("dx(c)")
		
				
		plt.subplot(2,3,4)
		c_new = dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*dataset.c_new
		advection_term = dataset.v[:,1:2]*dx(c_new)+dataset.v[:,0:1]*dy(c_new) + (dx(dataset.v[:,1:2])+dy(dataset.v[:,0:1]))*c_new
		residuals = (c_new-dataset.c_old)/dt - dataset.D*laplace(dataset.c_new) + advection_term - dataset.R
		plt.imshow(((1-dataset.bc_mask)*residuals)[0,0].detach().cpu())
		plt.colorbar()
		plt.title("residuals")
		
		"""
		plt.subplot(2,3,4)
		plt.imshow(dataset.bc_mask[0,0].detach().cpu())
		plt.colorbar()
		plt.title("bc_mask")
		"""
		plt.subplot(2,3,5)
		plt.imshow((dataset.bc_values[0,0]*dataset.bc_mask[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("bc_values")
		
		#plt.show()
		plt.draw()
		plt.pause(0.001)
	
	x_old = x.data.clone()
	x.grad = grads
	optim.step()
	loss = dataset.tell(x.data-x_old)
	
	print(f"{i} : {dataset.T[0]} / {loss}")
