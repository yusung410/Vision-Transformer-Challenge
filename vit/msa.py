import torch
import torch.nn as nn
from patch_embed import EmbeddingLayer

class MSA(nn.Module):
	def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super().__init__()
		assert dim % num_heads == 0 # dim은 num_heads로 나누어져야 한다.
		
		self.num_heads = num_heads # multi-head의 개수
		head_dim = dim // num_heads  # 각 head의 dimension
		self.scale = head_dim ** -0.5 # head의 dimension으로 정규화

		self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # Self-Attention에서 linear projection으로 qkv 생성하기
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape # [Batch, N, Channel]
		# qkv를 만들고, multi-head로 나눈다.
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0) # 입력 차원 * 3의 dimension으로 임베딩되지만 같은 입력에서 나오는 q, k, v가 같지는 않다. 
		attn = (q @ k.transpose(-2,-1)) # Q, K를 내적하여 attention score 계산

		attn = attn.softmax(dim=-1) # attention score를 softmax로 정규화
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1,2).reshape(B,N,C) # attention score와 V를 곱하고 다시 projection
		x = self.proj(x)
		x = self.proj_drop(x)
		return x
