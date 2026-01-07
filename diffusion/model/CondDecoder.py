import torch
import torch.nn as nn
import torch.nn.functional as F


class PCDecoder(nn.Module):
	def __init__(self, cond_embed_dim, dp_rate):
		super(PCDecoder, self).__init__()
		self.l1 = nn.Linear(cond_embed_dim, 1024)
		self.l2 = nn.Linear(1024, 2048)
		self.l3 = nn.Linear(2048, 6890 * 3)
		self.dropout = nn.Dropout(dp_rate)
	
	def forward(self, z):
		out = F.relu(self.l1(z))
		out = self.dropout(out)
		out = F.relu(self.l2(out))
		out = self.dropout(out)
		out = self.l3(out)
		out = out.view(-1, 6890, 3)
		return out
	
	
