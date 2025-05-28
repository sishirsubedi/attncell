import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import multi_dir_log_likelihood, kl_loss
				
import logging
logger = logging.getLogger(__name__)
import numpy as np
import random 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def attncell_train_common(model,data,
	epochs:int,
	l_rate:float,
	cl_loss_mode:str, 
	min_batchsize:int = 5
	):
	data_size = len(data.dataset)
	opt = torch.optim.Adam(model.parameters(),lr=l_rate,weight_decay=1e-4)
	epoch_losses = []
	for epoch in range(epochs):
		epoch_l = 0 
		for x_c1,y,x_c2,nbr_weight in data:
						
			if x_c1.shape[0] < min_batchsize:
				continue
			
			opt.zero_grad()

			attncell_out = model(x_c1,x_c2)

			z_c1 = attncell_out.z_c1
			bm = attncell_out.etm_out_c1['m']
			bvar = attncell_out.etm_out_c1['var']
			theta = attncell_out.etm_out_c1['theta']
			beta = attncell_out.etm_out_c1['beta']
			alpha = torch.exp(torch.clamp(torch.mm(theta,beta),-10,10))
			loglikloss = multi_dir_log_likelihood(x_c1,alpha)
			klb = kl_loss(bm,bvar)
			train_loss = torch.mean(loglikloss).add(torch.sum(klb)/data_size)
   
			train_loss.backward()

			opt.step()
   
			epoch_l += train_loss.item()
		   
		epoch_losses.append([epoch_l/len(data)])  
		
		if epoch % 10 == 0:
			logger.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1,epoch_l/len(data) ))

		return epoch_losses
