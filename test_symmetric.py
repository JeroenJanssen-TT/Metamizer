from metamizer import SymmetricMixedUnet, SymmetricMetamizer
import torch
from d4torch import C1,C2,C4,D1,D2,D4,cat,Symmetries,scalar, Trafos
from get_param import toCuda
import numpy as np

net = toCuda(SymmetricMixedUnet(in_channels=1, out_channels=1, out_scalar_channels=1, hidden_size=10, bilinear=True, symmetry_group=D4))
net.eval()

for h in range(32,128):
	print(f"h: {h}")
	x_in = toCuda(torch.randn(1,1,h,h))
	
	# TODO: test different trafos!
	np.random.seed(0)
	x_out1_orig, x_out2_orig = net(x_in.clone())
	np.random.seed(0)
	x_out1_neg_orig, x_out2_neg_orig = net(-x_in.clone())
	print(f"antisym: {torch.all(x_out1_orig==-x_out1_neg_orig)}")
	print(f"    sym: {torch.all(x_out2_orig==x_out2_neg_orig)}")
	
	if True:
		
		for trafo in Trafos:
			print(f"trafo: {trafo}")
			np.random.seed(0)
			x_out1, x_out2 = net(trafo.g_mul(x_in.clone()))
			np.random.seed(0)
			x_out1_neg, x_out2_neg = net(-trafo.g_mul(x_in.clone()))
			
			print(f"group-equivariant: {torch.allclose(x_out1,trafo.g_mul(x_out1_orig))}") # warum ist Id nicht immer =? => weil offsets von down/up-sampling randomly bestimmt werden => setze np.random.seed(0)
			"""
			print(f"antisym: {torch.allclose(x_out1,-x_out1_neg)}")
			print(f"antisym: {torch.allclose(x_out1,-trafo.g_mul(x_out1_neg_orig))}")
			#print(f"sym: {x_out2} vs {x_out2_neg} {torch.allclose(x_out2,x_out2_neg)}")
			print(f"    sym: {torch.allclose(x_out2,x_out2_neg)}")
			print(f"    sym: {torch.allclose(x_out2,x_out2_neg_orig)}")
			"""
	
# symmetrisch: 48, 64, 80, 96, 112 => in 16er schritten
