from dataset_wave import DatasetWave
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_param import params,device
from derivatives import laplace
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta

h,w = params.height,params.width
params.iterations_per_timestep = 20
dt = params.dt

dataset = DatasetWave(h,w,dataset_size=1,batch_size=1,average_sequence_length=99999999999,iterations_per_timestep=params.iterations_per_timestep,c_range=params.c)

x = torch.zeros(1,1,h,w,requires_grad = True,device=device)

#optim = SGD([x],lr=0.01)
optim = Adam([x],lr=0.05) # faster but worse minimum
#optim = Adam([x],lr=0.01) # slower but slightly better minimum
#optim = AdamW([x],lr=0.01)
#optim = RMSprop([x],lr=0.01)
#optim = Adagrad([x],lr=1)
#optim = Adadelta([x],lr=0.5)

for i in range(100000):
	
	grads,_ = dataset.ask()
	
	
	if i%10 == 0:
		plt.clf()
		
		# plt x / v / a / bc_mask / bc_values
		plt.subplot(2,3,1)
		plt.imshow(dataset.x_old[0,0].detach().cpu())
		plt.colorbar()
		plt.title("x")
		
		plt.subplot(2,3,2)
		plt.imshow(dataset.v_old[0,0].detach().cpu())
		plt.colorbar()
		plt.title("v")
		
		plt.subplot(2,3,3)
		plt.imshow(dataset.a[0,0].detach().cpu())
		plt.colorbar()
		plt.title("a")
		
		plt.subplot(2,3,4)
		plt.imshow(grads[0,0].detach().cpu())
		plt.colorbar()
		plt.title("grads")
		
		"""
		plt.subplot(2,3,5)
		plt.imshow(dataset.bc_mask[0,0].detach().cpu())
		plt.colorbar()
		plt.title("bc_mask")
		"""
		
		plt.subplot(2,3,5)
		v_new = dataset.v_old + dt*dataset.a
		x_new = dataset.x_old + dt*v_new
		v_new = (1-dataset.bc_mask)*v_new # velocity = 0 at boundaries
		x_new = dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*x_new
		residuals = (v_new-dataset.v_old)/dt - dataset.c**2*laplace(0.5*(x_new+dataset.x_old))
		plt.imshow(((1-dataset.bc_mask)*residuals)[0,0].detach().cpu())
		plt.colorbar()
		plt.title("residuals")
		
		plt.subplot(2,3,6)
		plt.imshow((dataset.bc_values[0,0]*dataset.bc_mask[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("bc_values")
		
		plt.draw()
		plt.pause(0.001)
	
	x_old = x.data.clone()
	x.grad = grads
	optim.step()
	loss = dataset.tell(x.data-x_old)
	
	print(f"{i} : {dataset.T[0]} / {loss}")
