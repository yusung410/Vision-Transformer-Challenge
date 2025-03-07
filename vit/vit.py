from transformer_encoder import TransformerEncoder
from patch_embed import EmbeddingLayer
import torch.nn as nn

class ViT(nn.Module):
	def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192,depth=4,num_heads=12,mlp_ratio=2.,qkv_bias=False,drop_rate=0.,attn_drop_rate=0.):
		super().__init__()
		self.num_classes = num_classes
		self.num_features = self.embed_dim = embed_dim
		norm_layer = nn.LayerNorm
		act_layer = nn.GELU
		self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size)
		self.transformer_encoder = nn.Sequential(*[TransformerEncoder(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop=drop_rate,attn_drop=attn_drop_rate,norm_layer=norm_layer,act_layer=act_layer)])
		# final norm
		self.final_norm = norm_layer(embed_dim)
		
		# classification head
		self.classification_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

	def forward(self, x):
		x = self.patch_embed(x)
		x = self.transformer_encoder(x)
		x = self.transformer_encoder(x)
		x = self.transformer_encoder(x)
		x = self.transformer_encoder(x)
		x = self.final_norm(x) 
		x = self.classification_head(x)[:,0]
		return x
