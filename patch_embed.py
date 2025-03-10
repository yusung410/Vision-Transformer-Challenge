import torch
import math
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_ 

class EmbeddingLayer(nn.Module):
	def __init__(self, in_chans, embed_dim, img_size, patch_size):
		super().__init__()
		# N, 패치 개수
		self.num_tokens = (img_size // patch_size) ** 2 
		# D, 패치 임베딩의 차원
		self.embed_dim = embed_dim 
		# kernel_size=patch_size, stride=patch_size라는 것은 겹치지 않는 패치들로 나눈다는 것이다. 
		# 정확히 패치 간격으로 컨볼루션을 수행하는 것이다. # 패치 하나 하나를 192차원으로 만들어주는 것이라고 보면 된다.
		self.project = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size) 
		# class token
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.num_tokens += 1
		nn.init.normal_(self.cls_token, std=1e-6)

	def get_sinusoidal_pos_embed(self, num_patches, embed_dim):
		# patch embedding dimension과 같다.
		pos_embed = torch.zeros(num_patches, embed_dim)
		# torch.arange(0,num_patches)는 [0 ~ num_patches - 1]까지 각각의 값을 가지는 리스트
		position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1) 
		# sin, cos 함수의 주기를 설정하는 부분, [0 ~ embed_dim - 1]까지 2 간격으로 값을 생성한다.
		div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)) 
		pos_embed[:,0::2]=torch.sin(position*div_term) # 짝수 인덱스는 sin
		pos_embed[:,1::2]=torch.cos(position*div_term) # 홀수 인덱스는 cos
		return pos_embed

	def forward(self, x):
		# Batch, Channel, Height, Width
		B, C, H, W = x.shape
		# patch x를 임베딩 (3차원 -> C차원)
		embedding = self.project(x)
		# z가 patch embedding된 벡터 (B,C,H,W) -> (B,C,N) -> (B,N,C)
		z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1)
		# concat cls token -> 마지막 자리에 추가
		cls_tokens = self.cls_token.expand(B, -1, -1)
		z = torch.cat([cls_tokens, z], dim=1)
		# position embedding 더하기 (Patch embedding한 벡터에 position embedding 벡터 더해주기, 각각의 embedding의 위치를 기억
		device = torch.device("cuda")
		pos_encoding = self.get_sinusoidal_pos_embed(self.num_tokens,self.embed_dim)
		# z(patch embedding)와 pos_encoding을 .to(device)로 보내서 더해주기 -> 똑같은 cpu or cuda 장치에 있어야 서로 연산이 가능하다.
		z = z.to(device)
		pos_encoding = pos_encoding.to(device)
		z = z + pos_encoding
		return z # class token, position encoding을 포함한 patch embedding z를 반환한다. 이것이 Transformer encoder의 입력이다.
