import picasa 
import anndata as ad
import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


############################

df = pd.read_csv('/results/attention_scores.csv.gz',index_col=0)

unique_celltypes = ['alpha', 'beta', 'delta','acinar','ductal']

##############################################

dfl = pd.DataFrame(df.index.values)
dfl['patient'] = [x.split('@')[0] for x in dfl[0]]
dfl['celltype'] = [x.split('@')[1].split('_')[0] for x in dfl[0]]

dfl = dfl[dfl['celltype'].isin(unique_celltypes)]

dfl['gene'] = [x.split('_')[1] for x in dfl[0]]
dfl.drop(0,axis=1,inplace=True)
dfl.reset_index(inplace=True)


######## fix cell type
marker = [
'PLCE1','LOXL4',#alpha
'NPTX2','DLK1',#beta
'LEPR','RBP4',#delta
'PLA2G1B','CPA1',#acinar
'KRT19','SPP1',#ductal
]

unique_celltypes = dfl['celltype'].unique()

fig, axes = plt.subplots(4, 2, figsize=(10, 20))

for idx, ct in enumerate(unique_celltypes):
    
    row, col = idx // 2, idx % 2
    
    ct_ylabel = dfl[dfl['celltype'] == ct].index.values
    df_attn = df.iloc[ct_ylabel,:].copy()
    print(df_attn.shape)
    
    df_attn[df_attn > .0001] = .0001

    sel_genes = [x for x in marker if x in df_attn.columns]
    df_attn = df_attn.loc[:,sel_genes]
    
    df_attn['gene'] = [x.split('_')[1] for x in df_attn.index.values]
    df_attn = df_attn[df_attn['gene'].isin(sel_genes)]
    df_attn = df_attn.groupby('gene').mean()
    
    df_attn = df_attn.loc[sel_genes,sel_genes]
    
    df_attn.columns = [x.split('-')[0] for x in df_attn.columns]
    df_attn.index = [x.split('-')[0] for x in df_attn.index]
    sns.heatmap(df_attn, ax=axes[row, col],
                yticklabels=df_attn.index,  
                xticklabels=df_attn.columns,  
                cmap='viridis' 
                )
    axes[row, col].set_title(ct)
    
plt.tight_layout()
plt.savefig('results/figure2_attention_celltype_markers_top.pdf')
plt.close()
