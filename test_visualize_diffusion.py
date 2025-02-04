import matplotlib.pyplot as plt
from dataset_diffusion import DatasetDiffusion
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,toDType
import time
import os
from derivatives import dx,dy,laplace

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)


D = params.D#10#2#0.5#0.3#0.1#
save = False#True#
if save:
	path = f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}/diffusion/D_{D}"
	os.makedirs(path,exist_ok=True)
	frame = 0
dpi = 200

metamizer = toDType(toCuda(get_Net(params)))

date_time,index = logger.load_state(metamizer,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, {index}")
metamizer.eval()

dt = params.dt

scales = []
max_scales = []
losses = []


with torch.no_grad():#enable_grad():#
	for epoch in range(1):
		dataset = DatasetDiffusion(params.height,params.width,1,1,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep)
		
		# warm up NN (first run always takes a bit longer => do not consider for timing
		dataset.reset0_env(0)
		grads, hidden_states = dataset.ask()
		update_steps, new_hidden_states = metamizer(grads, hidden_states)
		
		# reset dataset again and start
		dataset.reset0_env(0)
		dataset.D[:] = D
		FPS=0
		start_time = time.time()

		for t in range(params.average_sequence_length*params.iterations_per_timestep):
			print(f"t: {t}")
			
			grads, hidden_states = dataset.ask()
			
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			
			loss = dataset.tell(update_steps, new_hidden_states)
			
			scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
			losses.append(loss.detach().cpu().numpy())
			
			# TODO: visualize, how gradient scaling changes during update steps
			
			if (t+1)%(params.iterations_per_timestep)==0:
			
				if False:
					plt.figure(1,figsize=(20,20),dpi=dpi)
					plt.clf()
			
					# plt u / f / bc_mask / bc_values
					plt.subplot(2,3,1)
					plt.imshow(dataset.c_new[0,0].detach().cpu())
					plt.colorbar()
					plt.title("c")
					"""
					plt.subplot(2,3,6)
					plt.imshow(dataset.c_old[0,0].detach().cpu())
					plt.colorbar()
					plt.title("c_old")
					"""
					
					plt.subplot(2,3,6)
					plt.imshow(update_steps[0,0].detach().cpu())
					plt.colorbar()
					plt.title("update_steps")
					
					plt.subplot(2,3,2)
					plt.imshow(grads[0,0].detach().cpu())
					plt.colorbar()
					plt.title("grads")
					
					plt.subplot(2,3,3)
					plt.imshow(dataset.R[0,0].detach().cpu())
					plt.colorbar()
					plt.title("R")
					
					
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
					
					plt.draw()
					plt.pause(0.001)
				
				if False:
					plt.figure(4,dpi=dpi)
					plt.clf()
					plt.semilogy(losses)
					plt.xlabel("iteration")
					plt.ylabel("loss")
					plt.xticks([i*10 for i in range(21)])
				
					plt.draw()
					plt.pause(0.01)
				
				if True:
					plt.figure(2,figsize=(1200/dpi,600/dpi),dpi=dpi)
					plt.clf()
					plt.semilogy(scales[-500:])
					plt.xlabel("iteration")
					plt.ylabel("scale")
					plt.legend(["scales"])
					plt.title("Diffusion equation")
					#plt.xticks([i*10 for i in range(21)])
					
					plt.grid(True, which="major", ls="-", color='0.85')
					plt.grid(True, which="minor", ls="--", color='0.95')
					if save and t>=50:
						plt.savefig(f"{path}/diffusion_scaling.pdf",dpi=dpi,bbox_inches="tight")
						exit()
					
					plt.draw()
					plt.pause(0.01)
				
				if True:
					plt.figure(3,figsize=(800/dpi,800/dpi),dpi=dpi)
					plt.clf()
					plt.imshow(dataset.c_new[0,0].detach().cpu())
					plt.colorbar()
					plt.axis('off')
					plt.title(f"timestep: {dataset.T[0].cpu().numpy()[0]}")
					
					if save:
						plt.savefig(f"{path}/{str(frame).zfill(4)}.png",dpi=dpi)
						frame += 1
						
					#plt.show()
					plt.draw()
					plt.pause(0.01)
			
		
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
		
		
