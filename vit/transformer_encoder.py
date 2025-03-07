from mlp import MLP
from msa import MSA
import torch.nn as nn 
class TransformerEncoder(nn.Module):
	def __init__(self,dim,num_heads,mlp_ratio=4., qkv_bias=False,drop=0.,attn_drop=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.norm2 = norm_layer(dim)
		self.attn = MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
		self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)

	def forward(self,x):
		x = x + self.attn(self.norm1(x)) # x = x + x -> norm -> MSA 
		x = x + self.mlp(self.norm2(x)) # x = x + x -> norm -> mlp
		return x
	
