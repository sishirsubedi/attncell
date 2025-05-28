import os
import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/attncell/')

import attncell
import anndata as an
import glob

 
sample = 'testds'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/attncell/test'

common_epochs = 1
common_meta_epoch = 15
unique_epoch = 250
base_epoch = 250

ddir = wdir+'/data/'

pattern = sample+'_*.h5ad'

file_paths = glob.glob(os.path.join(ddir, pattern))
file_names = [os.path.basename(file_path) for file_path in file_paths]

batch_map = {}
batch_count = 0
for file_name in file_names:
	print(file_name)
	batch_map[file_name.replace('.h5ad','').replace(sample+'_','')] = an.read_h5ad(ddir+file_name)
	batch_count += 1
	if batch_count >=12:
		break

attncell_object = attncell.create_attncell_object(
	batch_map,
    sample,
	wdir
 	)

params = {'device' : 'cuda',
		'batch_size' : 100,
		'input_dim' : 2000,
		'embedding_dim' : 3000,
		'attention_dim' : 15,
		'latent_dim' : 15,
		'encoder_layers' : [100,15],
		'projection_layers' : [25,25],
		'learning_rate' : 0.001,
		'pair_search_method' : 'approx_50',
        'cl_loss_mode' : 'none', 
		'epochs': common_epochs,
		'meta_epochs': common_meta_epoch
		}    

attncell_object.estimate_neighbour(params['pair_search_method'])

attncell_object.set_nn_params(params)


attncell_object.train()


attncell_object.plot_loss(tag='common')


device = 'cpu'
attncell_object.nn_params['device'] = device
eval_batch_size = 500
attncell_object.eval(eval_batch_size,device)

attncell_object.save_model()


