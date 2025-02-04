from dataset_poisson import DatasetPoisson
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_param import params,device
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta
from derivatives import laplace

h,w = 64,64#100,100

# TODO: use cuda...

dataset = DatasetPoisson(h,w,dataset_size=1,batch_size=1,average_sequence_length=99999999999)

x = torch.zeros(1,1,h,w,requires_grad = True,device=device)

#optim = SGD([x],lr=0.01)
#optim = Adam([x],lr=0.1) # faster but worse minimum
optim = Adam([x],lr=0.001) # slower but slightly better minimum
#optim = AdamW([x],lr=0.01)
#optim = RMSprop([x],lr=0.01)
#optim = Adagrad([x],lr=1)
#optim = Adadelta([x],lr=0.5)


for i in range(100000):
	
	grads,_ = dataset.ask()
	
	if i%20 == 0:
		plt.clf()
		
		# plt u / f / bc_mask / bc_values
		plt.subplot(2,3,1)
		plt.imshow(dataset.u[0,0].detach().cpu())
		plt.colorbar()
		plt.title("u")
		
		plt.subplot(2,3,2)
		plt.imshow(dataset.f[0,0].detach().cpu())
		plt.colorbar()
		plt.title("f")
		
		plt.subplot(2,3,3)
		plt.imshow(grads[0,0].detach().cpu())
		plt.colorbar()
		plt.title("grads")
		
		plt.subplot(2,3,4)
		plt.imshow((dataset.bc_values[0,0]*dataset.bc_mask[0,0]).detach().cpu())
		plt.colorbar()
		plt.title("bc_values*bc_mask")
				
		plt.subplot(2,3,6)
		u = dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*dataset.u
		residuals = laplace(u)+dataset.f
		plt.imshow((residuals[0,0]*(1-dataset.bc_mask[0,0])).detach().cpu())
		plt.colorbar()
		plt.title("residuals")
				
		plt.suptitle(f"i: {i}")
		
		plt.draw()
		plt.pause(0.001)
	
	x_old = x.data.clone()
	x.grad = grads
	optim.step()
	dataset.tell(x.data-x_old)
