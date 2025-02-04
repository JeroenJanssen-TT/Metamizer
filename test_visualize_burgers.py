import matplotlib.pyplot as plt
from dataset_burgers import DatasetBurgers
#from setups_multistep_1_channel import Dataset
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,toDType
import time
import os
from derivatives import vector2HSV, rot_mac, staggered2normal
from derivatives import dx,dy,laplace

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)

use_cv2 = False#True#

save = False#True#
if save:
	path = f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}/burgers/mu_{params.mu}"
	os.makedirs(path,exist_ok=True)
	frame = 0
dpi = 200

metamizer = toDType(toCuda(get_Net(params)))

date_time,index = logger.load_state(metamizer,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, {index}")
metamizer.eval()

if use_cv2:
	import cv2
	cv2.namedWindow('v',cv2.WINDOW_NORMAL)

dt = params.dt

scales = []
losses = []

with torch.no_grad():
	for epoch in range(params.n_epochs):
		original_dataset = DatasetBurgers(params.height,params.width,1,1,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep)
		original_dataset.reset0_env(0)
		original_dataset.mu[:] = params.mu
		dataset = DatasetToSingleChannel(original_dataset)
		FPS=0
		start_time = time.time()

		for i in range(params.average_sequence_length*params.iterations_per_timestep):
			print(f"i: {i}")
			
			grads, hidden_states = dataset.ask()
			
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			
			loss = dataset.tell(update_steps, new_hidden_states)
			print(f"loss: {loss}")
			
			scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
			losses.append(loss.detach().cpu().numpy()) # loss is based on sum
			
			if use_cv2:
				if (i+1)%params.iterations_per_timestep == 0:
					image = np.float32(vector2HSV(original_dataset.v_new[0]))
					image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
					cv2.imshow('v',image)
					
					# keyboard interactions:
					key = cv2.waitKey(1)
					
					if key==ord('n'): # new simulation
						break
					if key==ord('q'): # quit simulation
						exit()
			else:
				if (i+1)%params.iterations_per_timestep==0:
					
					if False: # plot various quantities (u/v/gradients/residuals/...) for debugging
						plt.figure(1,figsize=(20,20),dpi=dpi)
						plt.clf()
				
						# plt u / f / bc_mask / bc_values
						plt.subplot(2,4,1)
						plt.imshow(original_dataset.v_new[0,0].detach().cpu())
						plt.colorbar()
						plt.title("u_new")
						
						plt.subplot(2,4,5)
						plt.imshow(original_dataset.v_new[0,1].detach().cpu())
						plt.colorbar()
						plt.title("v_new")
						
						plt.subplot(2,4,2)
						plt.imshow(grads[0,0].detach().cpu())
						plt.colorbar()
						plt.title("u grads")
						
						plt.subplot(2,4,6)
						plt.imshow(grads[1,0].detach().cpu())
						plt.colorbar()
						plt.title("v grads")
						
						plt.subplot(2,4,4)
						plt.imshow(original_dataset.bc_mask[0,0].detach().cpu())
						plt.colorbar()
						plt.title("bc_mask")
						"""
						plt.subplot(2,4,3)
						plt.imshow((original_dataset.bc_values[0,0]*original_dataset.bc_mask[0,0]).detach().cpu())
						plt.colorbar()
						plt.title("u bc_values")
						
						plt.subplot(2,4,7)
						plt.imshow((original_dataset.bc_values[0,1]*original_dataset.bc_mask[0,0]).detach().cpu())
						plt.colorbar()
						plt.title("v bc_values")
						"""
						v_new = original_dataset.bc_mask * original_dataset.bc_values + (1-original_dataset.bc_mask)*original_dataset.v_new
						residuals_u = (v_new[:,0:1]-original_dataset.v_old[:,0:1])/dt + v_new[:,0:1]*dx(v_new[:,0:1]) + v_new[:,1:2]*dy(v_new[:,0:1]) - original_dataset.mu*laplace(v_new[:,0:1])
						residuals_v = (v_new[:,1:2]-original_dataset.v_old[:,1:2])/dt + v_new[:,0:1]*dx(v_new[:,1:2]) + v_new[:,1:2]*dy(v_new[:,1:2]) - original_dataset.mu*laplace(v_new[:,1:2])
						plt.subplot(2,4,3)
						plt.imshow((residuals_u[0,0]).detach().cpu())
						plt.colorbar()
						plt.title("residuals u")
						
						plt.subplot(2,4,7)
						plt.imshow((residuals_v[0,0]).detach().cpu())
						plt.colorbar()
						plt.title("residuals v")
						
						
						plt.subplot(2,4,8)
						plt.imshow(torch.sqrt(original_dataset.v_new[0,0]**2+original_dataset.v_new[0,1]**2).detach().cpu())
						plt.colorbar()
						plt.title("v norm")
						
						plt.draw()
						plt.pause(0.001)
					
				
					if False: # plot loss over time
						plt.figure(4,dpi=dpi)
						plt.clf()
						plt.semilogy(losses)
						plt.xlabel("iteration")
						plt.ylabel("loss")
					
						plt.draw()
						plt.pause(0.01)
					
					if True: # plot scaling factor over time
						plt.figure(2)
						plt.clf()
						plt.semilogy(scales[-200:])
						plt.xlabel("iteration")
						plt.ylabel("scale")
						
						plt.draw()
						plt.pause(0.001)
						
					if True: # plot velocity norm over time
						plt.figure(3,figsize=(800/dpi,800/dpi),dpi=dpi)
						plt.clf()
						plt.imshow(torch.sqrt(original_dataset.v_new[0,0]**2+original_dataset.v_new[0,1]**2).detach().cpu())
						plt.colorbar()
						plt.axis('off')
						plt.title(f"timestep: {original_dataset.T[0].cpu().numpy()[0]}")
						
						if save:
							plt.savefig(f"{path}/{str(frame).zfill(4)}.png",dpi=dpi)
							frame += 1
						
						plt.draw()
						plt.pause(0.001)
			
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
		
	
