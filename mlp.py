import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
		super().__init__()
		out_features = in_features
		self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
		self.act = act_layer()
		self.drop1 = nn.Dropout(drop)
		self.fc2 = nn.Linear(hidden_features,out_features,bias=bias)
		self.drop2 = nn.Dropout(drop)

	def forward(self,x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop1(x)
		x = self.fc2(x)
		x = self.drop2(x)
		return x
