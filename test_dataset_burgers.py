from dataset_burgers import DatasetBurgers
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_param import params,device
from derivatives import dx,dy,laplace
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta

h,w = params.height,params.width
params.D_range = 5#0.1
params.iterations_per_timestep = 100
dt = params.dt

dataset = DatasetBurgers(h,w,dataset_size=1,batch_size=1,average_sequence_length=99999999999,iterations_per_timestep=params.iterations_per_timestep)
#dataset.reset0_env(0)


x = torch.zeros(1,2,h,w,requires_grad = True,device=device)

#optim = SGD([x],lr=0.01)
#optim = Adam([x],lr=0.1) # faster but worse minimum
optim = Adam([x],lr=0.01) # slower but slightly better minimum
#optim = AdamW([x],lr=0.01)
#optim = RMSprop([x],lr=0.01)
#optim = Adagrad([x],lr=1)
#optim = Adadelta([x],lr=0.5)


for i in range(100000):
	
	grads,_ = dataset.ask()
	
	if i%20 == 0:
		plt.clf()

		# plt u / f / bc_mask / bc_values
		plt.subplot(2,4,1)
		plt.imshow(dataset.v_new[0,0].detach().cpu())
		plt.colorbar()
		plt.title("u_new")
		
		plt.subplot(2,4,5)
		plt.imshow(dataset.v_new[0,1].detach().cpu())
		plt.colorbar()
		plt.title("v_new")
		
		plt.subplot(2,4,2)
		plt.imshow(grads[0,0].detach().cpu())
		plt.colorbar()
		plt.title("u grads")
		
		plt.subplot(2,4,6)
		plt.imshow(grads[0,1].detach().cpu())
		plt.colorbar()
		plt.title("v grads")
		
		plt.subplot(2,4,4)
		plt.imshow(dataset.bc_mask[0,0].detach().cpu())
		plt.colorbar()
		plt.title("bc_mask")
		"""
		plt.subplot(2,4,3)
		plt.imshow((dataset.bc_values[0,0]*dataset.bc_mask[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("u bc_values")
		
		plt.subplot(2,4,7)
		plt.imshow((dataset.bc_values[0,1]*dataset.bc_mask[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("v bc_values")
		"""
		v_new = dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*dataset.v_new
		residuals_u = (v_new[:,0:1]-dataset.v_old[:,0:1])/dt + v_new[:,0:1]*dx(v_new[:,0:1]) + v_new[:,1:2]*dy(v_new[:,0:1]) - dataset.mu*laplace(v_new[:,0:1])
		residuals_v = (v_new[:,1:2]-dataset.v_old[:,1:2])/dt + v_new[:,0:1]*dx(v_new[:,1:2]) + v_new[:,1:2]*dy(v_new[:,1:2]) - dataset.mu*laplace(v_new[:,1:2])
		plt.subplot(2,4,3)
		plt.imshow((residuals_u[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("residuals u")
		
		plt.subplot(2,4,7)
		plt.imshow((residuals_v[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("residuals v")
		
		plt.draw()
		plt.pause(0.001)
	
	x_old = x.data.clone()
	x.grad = grads
	optim.step()
	loss = dataset.tell(x.data-x_old)
	
	print(f"{i} : {dataset.T[0]} / {loss}")
