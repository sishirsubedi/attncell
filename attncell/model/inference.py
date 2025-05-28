import torch
import anndata as an
import numpy as np
import logging
import gc
logger = logging.getLogger(__name__)

def predict_batch_common(model,x_c1,y,x_c2 ):
    return model(x_c1,x_c2),y

def predict_attention_common(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)

    _,x_c1_attention = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    
    return x_c1_attention

def predict_context_common(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor                
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)

    x_c1_context,_= model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    
    return x_c1_context

def get_latent_common(model,
    x_c1:torch.tensor,
    x_c2:torch.tensor
    ):
    
    x_c1_emb = model.embedding(x_c1)
    x_c2_emb = model.embedding(x_c2)
    x_c1_context,_ = model.attention(x_c1_emb,x_c2_emb,x_c2_emb)
    x_c1_pool_out = model.pooling(x_c1_context)
    h_c1 = model.encoder(x_c1_pool_out)
    return h_c1

def eval_attention_common(model, data_loader,eval_total_size):
    
    model.eval()

    attn_list = []
    ylabel_list = []
    y_count = 0

    for x_c1,y,x_c2,nbr_weight in data_loader:
        x_c1_attn = predict_attention_common(model,x_c1,x_c2)
        attn_list.append(x_c1_attn.cpu().detach().numpy())
        ylabel_list.append(y)
        y_count += len(y)
        
        if y_count>eval_total_size:
            break
        
        del x_c1_attn, y
        gc.collect()
            
    attn_list = np.concatenate(attn_list, axis=0)
    ylabel_list = np.concatenate(ylabel_list, axis=0)

    return attn_list,ylabel_list
