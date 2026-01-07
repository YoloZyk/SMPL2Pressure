from __future__ import print_function
import torch
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

'''
main class: PointNet
@param k: 3 or 4
@param global_feat: True or False     global feature? True
@param tran: True or False          space transformation? False

input:
	pc: B x K x N

output:
	global feature: B x F      F=1024
	point feature: B x N x F   F=1088
'''


class STN3d(nn.Module):
	def __init__(self):
		super(STN3d, self).__init__()
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 1024, 1)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 9)
		self.relu = nn.ReLU()
		
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)
	
	def forward(self, x):
		batchsize = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)
		
		x = F.relu(self.bn4(self.fc1(x)))
		x = F.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)
		
		iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
			batchsize, 1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden
		x = x.view(-1, 3, 3)
		return x


class STNkd(nn.Module):
	def __init__(self, k=64, d_model=1088):
		super(STNkd, self).__init__()
		self.d_model = d_model
		
		self.conv1 = torch.nn.Conv1d(k, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, self.d_model - 64, 1)
		self.fc1 = nn.Linear(self.d_model - 64, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, k * k)
		self.relu = nn.ReLU()
		
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(self.d_model - 64)
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)
		
		self.k = k
	
	def forward(self, x):
		batchsize = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, self.d_model - 64)
		
		x = F.relu(self.bn4(self.fc1(x)))
		x = F.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)
		
		iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
			batchsize, 1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden
		x = x.view(-1, self.k, self.k)
		return x


class PointNetfeat(nn.Module):
	def __init__(self, global_feat=True, feature_transform=False):
		super(PointNetfeat, self).__init__()
		self.stn = STN3d()
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 1024, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.global_feat = global_feat
		self.feature_transform = feature_transform
		if self.feature_transform:
			self.fstn = STNkd(k=64)
	
	def forward(self, x):
		n_pts = x.size()[2]
		trans = self.stn(x)
		x = x.transpose(2, 1)
		x = torch.bmm(x, trans)
		x = x.transpose(2, 1)
		x = F.relu(self.bn1(self.conv1(x)))
		
		if self.feature_transform:
			trans_feat = self.fstn(x)
			x = x.transpose(2, 1)
			x = torch.bmm(x, trans_feat)
			x = x.transpose(2, 1)
		else:
			trans_feat = None
		
		pointfeat = x
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)
		if self.global_feat:
			return x, trans, trans_feat
		else:
			x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
			return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
	def __init__(self, k=3, d_model=1088, global_feat=True, tran=False):
		super(PointNet, self).__init__()
		self.d_model = d_model
		self.stn = STNkd(k=k, d_model=d_model)
		self.conv1 = torch.nn.Conv1d(k, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, self.d_model - 64, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(self.d_model - 64)
		self.global_feat = global_feat
		self.tran = tran
	
	def forward(self, x):
		n_pts = x.size()[2]
		trans = None
		if self.tran:
			trans = self.stn(x)
			x = x.transpose(2, 1)
			x = torch.bmm(x, trans)
			x = x.transpose(2, 1)
		x = F.relu(self.bn1(self.conv1(x)))
		
		pointfeat = x
		
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, self.d_model - 64)
		if self.global_feat:
			# return x, trans
			return x
		else:
			x = x.view(-1, self.d_model - 64, 1).repeat(1, 1, n_pts)
			x = torch.cat([x, pointfeat], 1)  # B x (1024+64) x N
			# 将x转为B x N x (1024+64)
			x = x.transpose(1, 2)
			# return x, trans
			return x


def feature_transform_regularizer(trans):
	d = trans.size()[1]
	batchsize = trans.size()[0]
	I = torch.eye(d)[None, :, :]
	if trans.is_cuda:
		I = I.cuda()
	loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
	return loss


class PCEncoder(nn.Module):
	def __init__(self, cond_embed_dim, cond_type, use_cls):
		super(PCEncoder, self).__init__()
		self.cond_type = cond_type
		self.pointnet = PointNet(k=3, d_model=cond_embed_dim, global_feat=True, tran=False)
	
	def forward(self, x, c3d=None):
		out = self.pointnet(x)
		return out


