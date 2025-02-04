import torch
from torch.autograd import Function
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# "pseudo function" that doesn't affect the outputs but only scales the gradients

eps = 1e-12#1e-6#

class ScaleGrads(Function):
	@staticmethod
	def forward(input, scale):
		return input
	
	@staticmethod
	def setup_context(ctx, inputs, output):
		input, scale = inputs
		ctx.save_for_backward(scale)
	
	@staticmethod
	def backward(ctx, grad_output):
		scale = ctx.saved_tensors[0]
		return scale*grad_output, None # no gradients for gradient scaling

scale_grads = ScaleGrads.apply

# "pseudo function" that doesn't affect the outputs but only normalizes the gradients
class NormalizeGrads(Function):
	@staticmethod
	def forward(input):
		return input
	
	@staticmethod
	def setup_context(ctx, inputs, output):
		pass
		
	@staticmethod
	def backward(ctx, grad_output):
		#print(f"{grad_output.shape}: {grad_output.dtype}")
		# achtung, normalization kann sehr hohe / niedrige werte zurückgeben, wenn fast alles = 0 ist => min / max clampen
		return normalize(grad_output,dim=[i+1 for i in range(len(grad_output.shape)-1)],eps=1e-40).clamp(min=-10,max=10)
		
		#std = torch.mean(grad_output**2,dim=[i+1 for i in range(len(grad_output.shape)-1)]).detach().clamp_min(eps).reshape(*([grad_output.shape[0]]+[1 for i in range(len(grad_output.shape)-1)])) # bringt nichts (sollte das selbe wie normalize tun)
		#return grad_output/std

normalize_grads = NormalizeGrads.apply

# "pseudo function" that doesn't affect the outputs but only normalizes the gradients
class NormalizeGradsScale(Function):
	@staticmethod
	def forward(ctx, input, scale):
		ctx.scale = scale
		return input
	
	@staticmethod
	def backward(ctx, grad_output):
		# achtung, normalization kann sehr hohe / niedrige werte zurückgeben, wenn fast alles = 0 ist => min / max clampen
		return ctx.scale*normalize(grad_output,dim=[i+1 for i in range(len(grad_output.shape)-1)]).clamp(min=-10,max=10), None
		
		#std = torch.mean(grad_output**2,dim=[i+1 for i in range(len(grad_output.shape)-1)]).detach().clamp_min(eps).reshape(*([grad_output.shape[0]]+[1 for i in range(len(grad_output.shape)-1)])) # bringt nichts (sollte das selbe wie normalize tun)
		#return grad_output/std

normalize_grads_scale = NormalizeGradsScale.apply


def log_range_params(range_params,default_param=1):# useful to sample parameters from "exponential distribution"
	range_params = default_param if range_params is None else range_params
	range_params = [range_params,range_params] if type(range_params) is not list else range_params
	range_params = np.log(range_params)
	return range_params[0],range_params[1]-range_params[0]

def range_params(r_params,default_param=1):# useful to sample parameters from "exponential distribution"
	r_params = default_param if r_params is None else r_params
	r_params = [r_params,r_params] if type(r_params) is not list else r_params
	return r_params[0],r_params[1]-r_params[0]

def has_nan(x):
	if type(x) is not torch.Tensor:
		return False
	return torch.any(x.isnan())

def has_inf(x):
	if type(x) is not torch.Tensor:
		return False
	return torch.any(x.isinf())

def value_range(x):
	if type(x) is not torch.Tensor:
		return None
	return [torch.min(x).detach().cpu().numpy(), torch.max(x).detach().cpu().numpy()]
