from dataset_cloth import DatasetCloth
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_param import params,device
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta

h,w = 64,64

params.dt = 1

dataset = DatasetCloth(h,w,dataset_size=1,batch_size=1,iterations_per_timestep=100,average_sequence_length=99999999999)

dataset.reset0_env(0)


x_grad = torch.zeros(1,3,h,w,requires_grad = True,device=device)

#optim = SGD([x_grad],lr=0.01)
#optim = Adam([x_grad],lr=0.1) # faster but worse minimum
optim = Adam([x_grad],lr=0.01) # slower but slightly better minimum
#optim = AdamW([x_grad],lr=0.01)
#optim = RMSprop([x_grad],lr=0.01)
#optim = Adagrad([x_grad],lr=1)
#optim = Adadelta([x_grad],lr=0.5)



for i in range(100000):
	
	
	grads,_ = dataset.ask()
	
	if i%100 == 0:
		x = dataset.x[0].cpu()
		a = dataset.a[0].cpu()
		
		plt.clf()
		fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
		surf = ax.plot_surface(x[0], x[1], x[2], linewidth=0.1, antialiased=False,edgecolors='k') # cloth surface
		ax.scatter(x[0,[0,-1],0],x[1,[0,-1],0],x[2,[0,-1],0],marker='o',color='g',depthshade=0) # boundarys conditions
		
		
		q_stride, q_l=8,10 # gradients
		"""
		ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
			q_l*grads[0,0,::q_stride,::q_stride], q_l*grads[0,1,::q_stride,::q_stride], q_l*grads[0,2,::q_stride,::q_stride],color='r')
		"""
		ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
			q_l*a[0,::q_stride,::q_stride], q_l*a[1,::q_stride,::q_stride], q_l*a[2,::q_stride,::q_stride],color='g')
		
		
		ax.set_zlim(-100, 1.01)
		ax.set_xlim(-50, 50)
		ax.set_ylim(-50, 50)
		plt.title(f"i: {dataset.iterations[0]}; T: {dataset.T[0]}")
		
		plt.draw()
		plt.pause(0.01)
	
	x_old = x_grad.data.clone()
	x_grad.grad = grads
	optim.step()
	loss = dataset.tell(x_grad.data-x_old)
	# should we call optim.zero_grad() or is this not needed since we set grad directly anyway? => TODO: test that! => indeed, as expected, doesn't make a difference

