import matplotlib.pyplot as plt
from matplotlib import cm
from dataset_wave import DatasetWave
#from setups_multistep_1_channel import Dataset
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,toDType
from derivatives import laplace
import time
import os
#from moviepy.editor import *

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)


save = False#True#
if save:
	path = f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}/wave/C_{C}"
	os.makedirs(path,exist_ok=True)
	frame = 0

metamizer = toDType(toCuda(get_Net(params)))


date_time,index = logger.load_state(metamizer,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, {index}")
metamizer.eval()


#params.dt=0.1
dt = params.dt

scales = []
losses = []


with torch.no_grad():
	for epoch in range(1):
		dataset = DatasetWave(params.height,params.width,1,1,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep,c_range=1)
		
		dataset.c[:] = params.c
		
		FPS=0
		start_time = time.time()

		for t in range(params.average_sequence_length*params.iterations_per_timestep):
			print(f"t: {t}")
			
			grads, hidden_states = dataset.ask()
			
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			
			loss = dataset.tell(update_steps, new_hidden_states)
			
			scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
			losses.append(loss.detach().cpu().numpy()) # loss verwendet mean
			
			if (t+1)%params.iterations_per_timestep==0:
				
				if False:
					plt.figure(3,figsize=(20,20),dpi=200)
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
				
				if False:
					plt.figure(4,dpi=150)
					plt.clf()
					plt.semilogy(losses[-200:])
					plt.xlabel("iteration")
					plt.ylabel("loss")
				
					plt.draw()
					plt.pause(0.01)
				
				if True:
					plt.figure(2)
					plt.clf()
					plt.semilogy(scales[-200:])
					plt.xlabel("iteration")
					plt.ylabel("scale")
					plt.legend(["scales"])
					plt.draw()
					plt.pause(0.01)
				
				if True:
					plt.figure(num=1,figsize=(20,20),dpi=200)
					plt.clf()
					fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1,computed_zorder=False)
					
					
					bc_masks = dataset.bc_mask[0,0].cpu()
					cond = (bc_masks > 0).nonzero()
					colors =  dataset.x_old[0,0].detach().cpu().unsqueeze(0).repeat(3,1,1)*bc_masks.unsqueeze(0)
					
					u = dataset.x_old[0,0].detach().cpu()
					color_values = 0.5*(u*(1-bc_masks)+1)
					color_values[bc_masks>0.5] = np.nan
					my_col = cm.jet(color_values)
					#u[bc_masks>0.5] = np.nan
					facecolors = cm.viridis((u - u.min()) / (u.max() - u.min()))  # Normalized colors
					facecolors[bc_masks>0.5] = [0, 0, 0, 1]
					
					surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=u, facecolors=facecolors,linewidth=0.1, antialiased=False,shade=True,zorder=4)
					
					
					#surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=u, edgecolors = 'k',linewidth=0.1, antialiased=False,shade=True)
					#surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=u, edgecolors = 'k',linewidth=0.1, antialiased=False,shade=True)#,alpha=0.5) # cloth surface
					#surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=dataset.u[0,0].detach().cpu(), linewidth=0.1, antialiased=False,zorder=4)
					
					#ax.scatter(x[0,[0,-1],0],x[1,[0,-1],0],x[2,[0,-1],0],marker='o',color='g',depthshade=0) # boundarys conditions
					#ax.scatter(dataset.x_mesh.cpu()[cond[:,0],cond[:,1]],dataset.y_mesh.cpu()[cond[:,0],cond[:,1]],dataset.u[0,0].detach().cpu()[cond[:,0],cond[:,1]],marker='o',color='g',depthshade=False) # boundarys conditions
					
					ax.grid(False)
					ax.set_axis_off()  # Completely removes the 3D box

					
					# Remove the axes (pane color)
					ax.xaxis.pane.fill = False
					ax.yaxis.pane.fill = False
					ax.zaxis.pane.fill = False

					# Hide the axes lines
					ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # X-axis
					ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Y-axis
					ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Z-axis

					# Optionally, hide the ticks
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_zticks([])
					
					ax.set_zlim(-3,3)
					ax.set_xlim(-1, params.height+1)
					ax.set_ylim(-1, params.height+1)
					plt.title(f"timestep: {dataset.T[0].cpu().numpy()[0]}")
					
					if save:
						plt.savefig(f"{path}/{str(frame).zfill(4)}.png", bbox_inches='tight')
						frame += 1
					if False:#dataset.T[0].cpu().numpy()[0]%25==0:
						plt.show()
					else:
						plt.draw()
						plt.pause(0.1)
						
					
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
