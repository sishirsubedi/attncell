import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .loss import reparameterize
import random 
import logging
logger = logging.getLogger(__name__)


		  
class Stacklayers(nn.Module):

	def __init__(self,input_size,layers,dropout=0.1):
		super(Stacklayers, self).__init__()
		self.layers = nn.ModuleList()
		self.input_size = input_size
		for next_l in layers:
			self.layers.append(nn.Linear(self.input_size,next_l))
			self.layers.append(nn.BatchNorm1d(next_l))
			self.layers.append(self.get_activation())
			self.layers.append(nn.Dropout(dropout))
			self.input_size = next_l

	def forward(self, input_data):
		for layer in self.layers:
			input_data = layer(input_data)
		return input_data

	def get_activation(self):
		return nn.ReLU()

class MLP(nn.Module):
	def __init__(self,
		input_dims:int,
		layers:list
		):
		super(MLP, self).__init__()
		
		self.fc = Stacklayers(input_dims,layers)

	def forward(self, x:torch.tensor):
		z = self.fc(x)
		return z

###### ATTNCELL COMMON MODEL #######

class ATTNCELLCommonOut:
	def __init__(self,z_c1,attn_c1,etm_out_c1):
		self.z_c1 = z_c1
		self.attn_c1 = attn_c1
		self.etm_out_c1 = etm_out_c1
				   
class GeneEmbedor(nn.Module):
	
	def __init__(self,
		emb_dim:int,
		out_dim:int,
		):
		super(GeneEmbedor, self).__init__()
		
		self.embedding = nn.Embedding(emb_dim,out_dim)
		self.emb_norm = nn.LayerNorm(out_dim)
		self.emb_dim = emb_dim

	def forward(self,
		x:torch.tensor):
		
		row_sums = x.sum(dim=1, keepdim=True)
		x_norm = torch.div(x, row_sums) * (self.emb_dim -1)
		return self.emb_norm(self.embedding(x_norm.int()))

class ScaledDotAttention(nn.Module):
	
	def __init__(self,
		weight_dim:int,
		):
		super(ScaledDotAttention, self).__init__()
		
		self.W_query = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.W_key = nn.Parameter(torch.randn(weight_dim, weight_dim))
		self.model_dim = weight_dim
		
	def forward(self,
		query:torch.tensor, 
		key:torch.tensor, 
		value:torch.tensor
		):

		query_proj = torch.matmul(query, self.W_query)
		key_proj = torch.matmul(key, self.W_key)
		
		scores = torch.matmul(query_proj, key_proj.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.model_dim).float())
						
		attention_weights = torch.softmax(scores, dim=-1)
		output = torch.matmul(attention_weights, value)

		return output, attention_weights

class AttentionPooling(nn.Module):

	def __init__(self, 
		model_dim:int
		):
		super(AttentionPooling, self).__init__()
		
		self.weights = nn.Parameter(torch.randn(model_dim))  
	
	def forward(self, 
		attention_output:torch.tensor
		):
		
		weights_softmax = torch.softmax(self.weights, dim=0)
		weighted_output = attention_output * weights_softmax.unsqueeze(0)
		pooled_output = torch.sum(weighted_output, dim=-1, keepdim=True)
		return pooled_output.squeeze(-1)

class ENCODER(nn.Module):
	def __init__(self,
		input_dims:int,
		layers:list
		):
		super(ENCODER, self).__init__()
		self.fc = Stacklayers(input_dims,layers)
  
	def forward(self, x:torch.tensor):
		return self.fc(x)


class ETMDecoder(nn.Module):
	def __init__(self,latent_dims,out_dims,jitter=.1):
		super(ETMDecoder, self).__init__()
		
		self.beta_bias= nn.Parameter(torch.randn(1,out_dims)*jitter)
		self.beta_mean = nn.Parameter(torch.randn(latent_dims,out_dims)*jitter)
		self.beta_lnvar = nn.Parameter(torch.zeros(latent_dims,out_dims))

		self.lsmax = nn.LogSoftmax(dim=-1)

	def forward(self, zz):
		
		theta = torch.exp(self.lsmax(zz))
		
		z_beta = self.get_beta()
		beta = z_beta.add(self.beta_bias)

		return {'m':self.beta_mean,'var':self.beta_lnvar,'theta':theta,'beta':beta}

	def get_beta(self):
		lv = torch.clamp(self.beta_lnvar,-5.0,5.0)
		z_beta = reparameterize(self.beta_mean,lv) 
		return z_beta


class ATTNCELLCommonNet(nn.Module):
	def __init__(self,
		input_dim:int, 
		embedding_dim:int, 
		attention_dim:int, 
		latent_dim:int,
		encoder_layers:list,
		projection_layers:list
		):
		super(ATTNCELLCommonNet,self).__init__()

		self.embedding = GeneEmbedor(embedding_dim,attention_dim)
		
		self.attention = ScaledDotAttention(attention_dim)
		
		self.pooling = AttentionPooling(attention_dim)

		self.encoder = ENCODER(input_dim,encoder_layers)
		
		self.decoder = ETMDecoder(latent_dim,input_dim)
						
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			init.xavier_uniform_(module.weight)
			if module.bias is not None:
				init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			init.xavier_uniform_(module.weight)
		elif isinstance(module, nn.Parameter):
			init.xavier_uniform_(module)
			
	def forward(self,x_c1,x_c2):
		
		x_c1_emb = self.embedding(x_c1)
		x_c2_emb = self.embedding(x_c2)
  
		x_c1_att_out, x_c1_att_w = self.attention(x_c1_emb,x_c2_emb,x_c2_emb)
		x_c1_pool_out = self.pooling(x_c1_att_out)

		z_c1 = self.encoder(x_c1_pool_out)

		etm_out_c1 = self.decoder(z_c1)

		return ATTNCELLCommonOut(z_c1,x_c1_att_w,etm_out_c1)

	def estimate(self,x_c1):
     
		x_c1_emb = self.embedding(x_c1)
  
		x_c1_att_out, x_c1_att_w = self.attention(x_c1_emb,x_c1_emb,x_c1_emb)
		x_c1_pool_out = self.pooling(x_c1_att_out)
		
		z_c1 = self.encoder(x_c1_pool_out)

		etm_out_c1 = self.decoder(z_c1)
		
		return ATTNCELLCommonOut(z_c1,x_c1_att_w,etm_out_c1)

