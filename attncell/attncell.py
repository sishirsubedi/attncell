from . import dutil 
from . import model 
from .util import generate_neighbours
import torch
import logging
import gc
import pandas as pd
import anndata as an
import numpy as np
import itertools



class attncell(object):
    """
    Initialize attncell model

    Parameters
    ----------
    data: dict of batch_name:anndata, each anndata is separate batch
    pair_mode: 'seq' for generating pairs in sequential - (1,2),(2,3),(3,4) else use all possible pair combination
    wdir: path to save model outputs            
    sample: name of data sample            
                
    Returns
    -------
    None

    """
    def __init__(self, 
        data: dutil.data.Dataset, 
        sample: str,
        wdir: str,
        pair_mode: str
        ):
     
     
        self.data = data
        self.sample = sample
        self.wdir = wdir
                
        logging.basicConfig(filename=self.wdir+'/results/attncell_model.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
  
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')  
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)
        
        
        self.adata_keys = list(self.data.adata_list.keys())

        
        logging.info('Batch pair mode is - '+pair_mode)
        
        if pair_mode == 'seq':
            
            indices = range(len(self.data.adata_list))
            adata_pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 1)]
            adata_pairs.append((adata_pairs[len(adata_pairs)-1][1],adata_pairs[0][0]))
            self.adata_pairs = adata_pairs
            
        elif pair_mode == 'dist':
            
            from sklearn.decomposition import TruncatedSVD
            from sklearn.metrics.pairwise import euclidean_distances
            
            blist = [self.data.adata_list[x].X for x in self.data.adata_list]
            number_of_pairs = int(pair_mode.split('_')[1])
            blist_keys = self.data.adata_list.keys()

            batch_similarity = euclidean_distances([TruncatedSVD(n_components=10).fit_transform(b).mean(axis=0) for b in blist])

            df_sim = pd.DataFrame(batch_similarity)

            top_n_pairs = pd.DataFrame({
                batch: distances.nsmallest(number_of_pairs + 1).iloc[1:].index.tolist() 
                for batch, distances in df_sim.iterrows()
            }).T

            unique_pairs = set()
            for batch, neighbors in top_n_pairs.iterrows():
                for neighbor in neighbors:
                    unique_pairs.add(tuple(sorted([batch, neighbor])))

            self.adata_pairs = list(unique_pairs)

        elif pair_mode == 'all':

            indices = list(range(len(self.data.adata_list)))
            self.adata_pairs = list(itertools.combinations(indices, 2))
                    
    def estimate_neighbour(self,
        method:str ='approx_50'
        ):
     
        logging.info('Pair search method - '+ method)
        self.nbr_map = {}
  
        if 'approx' in method :

            number_of_trees = int(method.split('_')[1])
   
            for ad_pair in self.adata_pairs:
                p1 = self.adata_keys[ad_pair[0]]
                p2 = self.adata_keys[ad_pair[1]]
                logging.info('Generating neighbour using approximate method - ANNOY...\n'+p1+'_('+str(self.data.adata_list[p1].X.shape)+') >'+p2+'_('+str(self.data.adata_list[p2].X.shape)+')')
                self.nbr_map[p1+'_'+p2] = generate_neighbours(self.data.adata_list[p2],self.data.adata_list[p1],p1+p2,number_of_trees)
                self.nbr_map[p2+'_'+p1] = generate_neighbours(self.data.adata_list[p1],self.data.adata_list[p2],p2+p1,number_of_trees)
    
        elif method == 'exact':
            from scipy.spatial.distance import cdist
            logging.info('Generating neighbour list using exact method - cdist...')
      
            for ad_pair in self.adata_pairs:
                p1 = self.adata_keys[ad_pair[0]]
                p2 = self.adata_keys[ad_pair[1]]

                logging.info(str(self.data.adata_list[p1].X.shape))
                logging.info(str(self.data.adata_list[p2].X.shape))
                distmat =  cdist(self.data.adata_list[p1].X.todense(), self.data.adata_list[p2].X.todense())
                sorted_indices_p1 = np.argsort(distmat, axis=1)
                sorted_indices_p2 = np.argsort(distmat.T, axis=1)
                self.nbr_map[p2+'_'+p1] = {x:y[0] for x,y in enumerate(sorted_indices_p2)}
                self.nbr_map[p1+'_'+p2] = {x:y[0] for x,y in enumerate(sorted_indices_p1)}
                
        logging.info('Pair search estimate is complete.')
                
    def set_nn_params(self,
        params: dict
        ):
        self.nn_params = params
    
    def create_model_adata(self,latent,df_beta):
    
        batches = latent.keys()
        
        frames = []        
        for b in batches:
            c_df = latent[b]
            c_df.index = [x+'@'+b for x in c_df.index.values]
            frames.append(c_df)
        
        df = pd.concat(frames)

        df.columns = ['common_'+str(x) for x in df.columns]
        
        adata = an.AnnData(obs=pd.DataFrame(index=df.index))
        adata.obsm['common'] = df
        
        
        batch_loc = len(df.index.values[0].split('@'))-1
        adata.obs['batch'] = [x.split('@')[batch_loc] for x in df.index.values]

        adata.uns['adata_keys'] = self.adata_keys
        adata.uns['adata_pairs'] = self.adata_pairs
        adata.uns['nn_params'] = self.nn_params
        
        batch_ids = {label: idx for idx, label in enumerate(adata.obs['batch'].unique())}
        adata.obs['batch_id'] = [batch_ids[x] for x in adata.obs['batch']]

        nbr_map_df = pd.DataFrame([
            {'batch_pair': l1_item, 'key': k, 'neighbor': v[0], 'score': v[1]}
            for l1_item, inner_map in self.nbr_map.items()
            for k, v in inner_map.items()
        ])
        adata.uns['nbr_map'] = nbr_map_df  
        adata.uns["beta"] = {
        "data": df_beta.values, 
        "index": list(df_beta.index),        
        "columns": list(df_beta.columns),
    }
        self.result = adata
       
    def set_batch_mapping(self):
        
        batch_mapping = { idx:label for idx, label in zip(self.result.obs.index.values,self.result.obs['batch_id'])}
        self.batch_mapping = batch_mapping
        
    def set_metadata(self):
        
        frames = []
        for ad_name in self.data.adata_list:
            ad = self.data.adata_list[ad_name]
            frames.append(ad.obs)
            
        df_meta = pd.concat(frames)

        df_meta.index = [x+'@'+y for x,y in zip(df_meta.index.values,df_meta['batch'])]
        sel_col = [ x for x in df_meta.columns if x not in ['batch','batch_id']]
        
        self.result.obs = pd.merge(self.result.obs,df_meta[sel_col],left_index=True,right_index=True,how='left')
        
            
    def train(self):

        logging.info('Starting attncell common training...')

        logging.info(self.nn_params)
  
        attncell_model = model.ATTNCELLCommonNet(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers']).to(self.nn_params['device'])
  
        logging.info(attncell_model)
  
        loss = []

        for it in range(self.nn_params['meta_epochs']):
        
            logging.info('meta_epochs : '+ str(it+1)+'/'+str(self.nn_params['meta_epochs']))
  
            for ad_pair in self.adata_pairs:
                p1 = self.adata_keys[ad_pair[0]]
                p2 = self.adata_keys[ad_pair[1]]
    
                logging.info('Training pair - '+p1+'_'+p2)
        
                data = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],self.nn_params['device'],self.nn_params['batch_size'])

                loss_p1_p2 = model.attncell_train_common(attncell_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['cl_loss_mode'])
    
                logging.info('Training pair switch - '+p2+'_'+p1)
    
                data = dutil.nn_load_data_pairs(self.data.adata_list[p2],self.data.adata_list[p1],self.nbr_map[p2+'_'+p1],self.nn_params['device'],self.nn_params['batch_size'])

                loss_p2_p1 = model.attncell_train_common(attncell_model,data,self.nn_params['epochs'],self.nn_params['learning_rate'],self.nn_params['cl_loss_mode'])

                loss_p1_p2 = np.array(loss_p1_p2)
                loss_p2_p1 = np.array(loss_p2_p1)
                stacked_loss_p = np.vstack((loss_p1_p2, loss_p2_p1))
                loss.append(np.mean(stacked_loss_p, axis=0))

        torch.save(attncell_model.state_dict(),self.wdir+'/results/attncell_common.model')
        pd.DataFrame(loss,columns=['ep_cl']).to_csv(self.wdir+'/results/attncell_common_train_loss.txt.gz',index=False,compression='gzip',header=True)
        logging.info('Completed training...model saved in '+self.wdir+'/results/attncell_common.model')
    
    def eval(self,
        eval_batch_size:int,
        device='cpu'
        ):
     
        attncell_model = model.ATTNCELLCommonNet(self.nn_params['input_dim'], self.nn_params['embedding_dim'],self.nn_params['attention_dim'], self.nn_params['latent_dim'], self.nn_params['encoder_layers'], self.nn_params['projection_layers']).to(self.nn_params['device'])
  
        attncell_model.load_state_dict(torch.load(self.wdir+'/results/attncell_common.model', weights_only=True, map_location=torch.device(device)))
  
        attncell_model.eval()
    
        latent = {}
  
        evaled = []
  
        for ad_pair in list(self.nbr_map.keys()):
            
            p1 = ad_pair.split('_')[0]
            p2 = ad_pair.split('_')[1]

            if p1 not in evaled:
                logging.info('eval :'+p1+'_'+p2)
                data_pred = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],device,eval_batch_size)
        
                df_latent = pd.DataFrame()

                for x_c1,y,x_c2,nbr_weight in data_pred:
                    attncell_out,ylabel = model.predict_batch_common(attncell_model,x_c1,y,x_c2)
                    df_latent = pd.concat([df_latent,pd.DataFrame(attncell_out.z_c1.cpu().detach().numpy(),index=ylabel)],axis=0)

                    del x_c1, y, x_c2, attncell_out, ylabel
                    gc.collect()
          
                latent[p1] = df_latent
                evaled.append(p1)
        
        df_beta = pd.DataFrame()
        ad_pair = list(self.nbr_map.keys())[0]
              
        p1 = ad_pair.split('_')[0]
        p2 = ad_pair.split('_')[1]
        logging.info('eval beta:'+p1+'_'+p2)
        data_pred = dutil.nn_load_data_pairs(self.data.adata_list[p1],self.data.adata_list[p2],self.nbr_map[p1+'_'+p2],device,eval_batch_size)
        
        for x_c1,y,x_c2,nbr_weight in data_pred:
            attncell_out,ylabel = model.predict_batch_common(attncell_model,x_c1,y,x_c2)
            df_beta = pd.DataFrame(attncell_out.etm_out_c1['beta'].cpu().detach().numpy())
            df_beta.index = ['t_'+str(i) for i in df_beta.index.values]
            break
                
        self.create_model_adata(latent,df_beta)
        self.set_metadata()

    def plot_loss(self,
        tag:str
        ):
        from attncell.util.plots import plot_loss
        if tag=='common':
            plot_loss(self.wdir+'/results/attncell_common_train_loss.txt.gz',self.wdir+'/results/attncell_common_train_loss.png')
        elif tag=='base':
            plot_loss(self.wdir+'/results/attncell_base_train_loss.txt.gz',self.wdir+'/results/attncell_base_train_loss.png')
    
    def save_model(self):
        self.result.write(self.wdir+'/results/attncell.h5ad',compression='gzip')
        
def create_attncell_object(
    adata_list:an.AnnData,
    sample:str, 
    wdir:str,
    pair_mode:str = 'seq'
    ):
    return attncell(dutil.data.Dataset(adata_list),sample,wdir,pair_mode)
