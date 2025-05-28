import sys
import scanpy as sc
import matplotlib.pylab as plt
import seaborn as sns
import anndata as ad
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/attncell/')


############################
sample = 'testds'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/attncell/test'

############ read model results as adata 
attncell_adata = ad.read_h5ad(wdir+'/results/attncell.h5ad')

nn_params = attncell_adata.uns['nn_params']
nn_params['device'] = 'cpu'

from attncell import model,dutil
import torch
import pandas as pd
import numpy as np 


attncell_model = model.ATTNCELLCommonNet(nn_params['input_dim'], nn_params['embedding_dim'],nn_params['attention_dim'], nn_params['latent_dim'], nn_params['encoder_layers'], nn_params['projection_layers']).to(nn_params['device']).to(nn_params['device'])
attncell_model.load_state_dict(torch.load(wdir+'/results/attncell_common.model', map_location=torch.device(nn_params['device'])))


sample_analyzed = []


df = pd.DataFrame()

p1 = 'indrop1'
p2 = 'smartseq2'

adata_p1 = ad.read_h5ad(wdir+'/data/testds_'+p1+'.h5ad')
adata_p2 = ad.read_h5ad(wdir+'/data/testds_'+p2+'.h5ad')

df_nbr = attncell_adata.uns['nbr_map']
df_nbr = df_nbr[df_nbr['batch_pair']==p1+'_'+p2]
nbr_map = {x:(y,z) for x,y,z in zip(df_nbr['key'],df_nbr['neighbor'],df_nbr['score'])}

data_loader = dutil.nn_load_data_pairs(adata_p1, adata_p2, nbr_map,'cpu',batch_size=10)
eval_total_size=1000
main_attn,main_y = model.eval_attention_common(attncell_model,data_loader,eval_total_size)

##############################################

unique_celltypes = adata_p1.obs['celltype'].unique()

num_celltypes = len(unique_celltypes)

df= pd.DataFrame()
for idx, ct in enumerate(unique_celltypes):
	
	ct_ylabel = adata_p1.obs[adata_p1.obs['celltype'] == ct].index.values
	ct_yindxs = np.where(np.isin(main_y, ct_ylabel))[0]

	min_cells = 5
	if len(ct_yindxs) < min_cells:
		continue
	if len(ct_yindxs) > 100:
		ct_yindxs = np.random.choice(ct_yindxs, 100, replace=False)

	df_attn = pd.DataFrame(np.mean(main_attn[ct_yindxs], axis=0),
						index=adata_p1.var.index.values, columns=adata_p1.var.index.values)
	np.fill_diagonal(df_attn.values, 0)
	df_attn.index = [p1+'@'+ct+'_'+x for x in df_attn.index]
	df = pd.concat([df, df_attn], axis=0)
	print(p1,ct,len(ct_yindxs),df.shape)

df.to_hdf(wdir +'/results/attention_scores.h5', key='df', mode='w', format='table', complevel=5, complib='blosc')

