import torch
import torch.nn as nn
from patch_embed import EmbeddingLayer

class MSA(nn.Module):
	def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		assert dim % num_heads == 0 # dim은 num_heads로 나누어져야 한다.
		
		self.num_heads = num_heads # multi-head의 개수
		head_dim = dim // num_heads  # 각 head의 dimension
		self.scale = head_dim ** -0.5 # 각 head의 dimension으로 정규화

		self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # Patch embedding을 linear projection하여 q,k,v를 만든다.
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim) # self-attention 이후에 다시 linear projection
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape # [Batch, N(65), Channel(192)]
		# self.qkv(x).shape => [128,65,576]
		# qkv.shape => [3,128,12,65,16]
		# q.shape => [128,12,65,16] 
		# attn.shape => [128,12,65,65] : attn = qk^T => [65,65] => 65 = N+1, 즉 패치 개수, 이미지 크기의 제곱에 비례
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0) # 입력 차원 * 3의 dimension으로 임베딩되지만 같은 입력에서 나오는 q, k, v가 같지는 않다.
		attn = (q @ k.transpose(-2,-1)) # Q, K를 내적하여 attention score 계산, 여기서 @는 행렬곱 연산자를 의미한다.

		attn = attn.softmax(dim=-1) # attn = softmax(QK^T) -> attention score를 softmax로 정규화
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1,2).reshape(B,N,C) # attn = softmax(QK^T)V -> attention score와 V를 곱하고 다시 projection
		x = self.proj(x)
		x = self.proj_drop(x)
		return x # Multi-head Self-attention layer의 결과

if __name__ == "__main__":
	device=torch.device("cuda")
	embed_layer=EmbeddingLayer(in_chans=3, embed_dim=192, img_size=32, patch_size=4).to(device)
	msa_layer=MSA(dim=192,num_heads=12,qkv_bias=False,attn_drop=0.,proj_drop=0.).to(device)
	img=torch.randn(128,3,32,32).to(device)
	embed=embed_layer(img).to(device)
	result=msa_layer(embed)