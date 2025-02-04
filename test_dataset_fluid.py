import torch
from torch.optim import Adam
import numpy as np
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from Logger import Logger,t_step
from get_param import params,toCuda,toCpu,device
from dataset_fluid import DatasetFluid
from derivatives import vector2HSV, rot_mac, staggered2normal
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta

#torch.manual_seed(1)
torch.set_num_threads(4)
#np.random.seed(6)

save_movie = False


h,w = 64,64

params.dt = 1

dataset = DatasetFluid(h,w,dataset_size=1,batch_size=1,iterations_per_timestep=100,average_sequence_length=99999999999)


x = torch.zeros(1,2,h,w,requires_grad = True,device=device)

#optim = SGD([x],lr=0.01)
optim = Adam([x],lr=0.1) # faster but worse minimum
#optim = Adam([x],lr=0.01) # slower but slightly better minimum
#optim = AdamW([x],lr=0.01)
#optim = RMSprop([x],lr=0.01)
#optim = Adagrad([x],lr=1)
#optim = Adadelta([x],lr=0.5)



# setup opencv windows:
cv2.namedWindow('legend',cv2.WINDOW_NORMAL) # legend for velocity field
vector = toCuda(torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]))
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('legend',image)


cv2.namedWindow('a',cv2.WINDOW_NORMAL)
cv2.namedWindow('v',cv2.WINDOW_NORMAL)
cv2.namedWindow('p',cv2.WINDOW_NORMAL)


for i in range(100000):
	
	print(f"i: {i} / T: {dataset.T[0]} / mu: {dataset.mus[0]} / rho: {dataset.rhos[0]}")
	
	grads,_ = dataset.ask()
	
	a_new = dataset.a_new[0][0]
	p = dataset.p[0][0]
	
	if i%100 == 0:
		
		# print out a:
		a = a_new.clone().cpu()
		a = a-torch.min(a)
		a = toCpu(a/torch.max(a)).unsqueeze(2).repeat(1,1,3).numpy()
		if save_movie:
			movie_a.write((255*a).astype(np.uint8))
		cv2.imshow('a',a)
		
		# print out v:
		
		#v_new = flow_mask_mac*v_new+cond_mask_mac*v_cond
		v_new = rot_mac(a_new.clone().unsqueeze(0).unsqueeze(1))
		vector = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
		image = vector2HSV(vector)
		image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
		if save_movie:
			movie_v.write((255*image).astype(np.uint8))
		cv2.imshow('v',image)
		
		
		# print out p:
		p = p.clone()
		#p = flow_mask[0,0]*p_new[0,0].clone()
		p = p-torch.min(p)
		p = p/torch.max(p)
		p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy()
		if save_movie:
			movie_p.write((255*p).astype(np.uint8))
		cv2.imshow('p',p)

		# keyboard interactions:
		key = cv2.waitKey(1)
		
		if key==ord('x'): # increase flow speed
			dataset.mousev+=0.1
		if key==ord('y'): # decrease flow speed
			dataset.mousev-=0.1
		
		if key==ord('s'): # increase angular velocity
			dataset.mousew+=0.1
		if key==ord('a'): # decrease angular velocity
			dataset.mousew-=0.1
		
		if key==ord('q'): # quit simulation
			break
	
	# gradient descent with pytorch optimizer
	x_old = x.data.clone()
	x.grad = grads
	optim.step()
	loss = dataset.tell(x.data-x_old)
