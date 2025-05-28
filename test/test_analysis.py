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
testds_adata = ad.read_h5ad(wdir+'/data/testds_celseq.h5ad')

####################################

sc.pp.neighbors(attncell_adata,use_rep='common')
sc.tl.umap(attncell_adata)
sc.pl.umap(attncell_adata,color=['batch','celltype'])
plt.tight_layout()
plt.savefig(wdir+'/results/attncell_umap_batch.png')

import pandas as pd
stored = attncell_adata.uns["beta"]
df_beta = pd.DataFrame(data=stored["data"], index=stored["index"], columns=stored["columns"])
df_beta.columns = testds_adata.var.index.values

def generate_gene_vals(df,top_n,top_genes,label):

	top_genes_collection = []
	for x in range(df.shape[0]):
		gtab = df.T.iloc[:,x].sort_values(ascending=False)[:top_n].reset_index()
		gtab.columns = ['gene','val']
		genes = gtab['gene'].values
		for g in genes:
			if g not in top_genes_collection:
				top_genes_collection.append(g)

	for g in top_genes_collection:
		for i,x in enumerate(df[g].values):
			top_genes.append(['k'+str(i),label,'g'+str(i+1),g,x])

	return top_genes

top_genes = generate_gene_vals(df_beta,5,[],'top_genes')
df_top = pd.DataFrame(top_genes,columns=['Topic','GeneType','Genes','Gene','Proportion'])
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')

sns.clustermap(df_top,cmap='viridis')
plt.tight_layout()
plt.savefig(wdir+'/results/attncell_beta_hmap.png')

