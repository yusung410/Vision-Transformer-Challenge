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
		x = self.patch_embed(x) # [128,65,192]
		x = self.transformer_encoder(x) # [128,65,192] -> 64 + 1(패치 + 클래스 토큰) -> 클래스 토큰과 패치가 정보를 주고받으며, feature를 추출
		x = self.transformer_encoder(x) # [128,65,192]
		x = self.transformer_encoder(x) # [128,65,192]
		x = self.transformer_encoder(x) # [128,65,192]
		x = self.final_norm(x) # [128,65,192]
		# self.classification_head(x) = [128,65,10] -> x[:,0]: class token prediction, x[:,1]: patch embedding prediction
		x = self.classification_head(x)[:,0] # [128,10] -> Transformer에서는 일반적으로 CLS 토큰이 최종적으로 문장이나 이미지의 대표적인 특징을 담는다.
		return x
