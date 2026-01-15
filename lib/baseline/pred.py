import numpy as np
import torch

from lib.utils.static import HD_SMPL_MODEL


class PMR:
	def __init__(self, verts, data):
		if torch.cuda.is_available():
			self.GPU = True
			# Use for self.GPU
			dtype = torch.cuda.FloatTensor
			dtypeInt = torch.cuda.LongTensor
		# print('######################### CUDA is available! #############################')
		else:
			self.GPU = False
			# Use for CPU
			dtype = torch.FloatTensor
			dtypeInt = torch.LongTensor
		# print('############################## USING CPU #################################')
		self.dtype = dtype
		self.dtypeInt = dtypeInt
		
		self.verts = verts
		self.data = data
		if self.data == 'pressurepose':
			self.w, self.h = 64, 27
		elif self.data == 'moyo':
			self.w, self.h = 110, 37
		elif self.data == 'tip':
			self.w, self.h = 56, 40
		else:
			raise ValueError('Invalid dataset type.')
		
		self.filler_taxels = []
		for i in range(self.h + 1):
			for j in range(self.w + 1):
				self.filler_taxels.append([i - 1, j - 1, 20000])
		self.filler_taxels = torch.Tensor(self.filler_taxels).type(self.dtypeInt).unsqueeze(0).repeat(
			self.verts.size()[0], 1, 1)
	
	# PMR - Pressure Map Reconstruction#
	def get_pressure(self):
		cbs = self.verts.size()[0]  # current batch size
		
		# print('Vert: ', self.verts.shape)
		# compute the depth and contact maps from the mesh
		verts_taxel = self.verts.clone()
		
		if self.data == 'pressurepose':
			# pressurepose
			verts_taxel /= 0.0286  # 修改后影响投影区域大小，意义类似分辨率
			verts_taxel[:, :, 0] -= 10  # x方向的平移，数值大小不清楚代表什么，
			verts_taxel[:, :, 1] -= 1  # y方向的平移，数值大小...   两个一起代表垫子空间位置
			verts_taxel[:, :, 2] *= 100  # z轴缩放，不清楚什么含义，但改成10后投影为0
		elif self.data == 'moyo':
			# moyo
			verts_taxel /= 0.014  # 修改后影响投影区域大小，意义类似分辨率
			verts_taxel[:, :, 0] += 18  # x方向的平移，数值大小不清楚代表什么，
			verts_taxel[:, :, 1] += 26  # y方向的平移，数值大小...   两个一起代表垫子空间位置
			verts_taxel[:, :, 2] *= 100  # z轴缩放，不清楚什么含义，但改成10后投影为0
		else:
			# # tip
			# verts_taxel[:, :, 0] /= 0.013  # 修改后影响投影区域大小，意义类似分辨率
			# verts_taxel[:, :, 1] /= 0.02  # 修改后影响投影区域大小，意义类似分辨率
			# verts_taxel[:, :, 0] -= 12  # x方向的平移，数值大小不清楚代表什么，
			# verts_taxel[:, :, 1] -= 22  # y方向的平移，数值大小...   两个一起代表垫子空间位置
			# verts_taxel[:, :, 2] *= 100  # z轴缩放，不清楚什么含义，但改成10后投影为0

			# tip new
			verts_taxel[:, :, 0] /= 0.021  # 修改后影响投影区域大小，意义类似分辨率
			verts_taxel[:, :, 1] /= 0.031  # 修改后影响投影区域大小，意义类似分辨率
			# verts_taxel[:, :, 0] += 4.1  # x方向的平移，数值大小不清楚代表什么，
			verts_taxel[:, :, 0] += 2.0  # x方向的平移，数值大小不清楚代表什么，
			verts_taxel[:, :, 1] -= 2  # y方向的平移，数值大小...   两个一起代表垫子空间位置
			verts_taxel[:, :, 2] *= 100  # z轴缩放，不清楚什么含义，但改成10后投影为0

		verts_taxel_int = (verts_taxel).type(self.dtypeInt)
		
		verts_taxel_int = torch.cat((self.filler_taxels[:, :, :], verts_taxel_int), dim=1)
		
		# import pdb; pdb.set_trace()
		vertice_sorting_method = (verts_taxel_int[:, :, 0:1] + 1) * 100000000 + \
		                         (verts_taxel_int[:, :, 1:2] + 1) * 100000 + \
		                         verts_taxel_int[:, :, 2:3]
		verts_taxel_int = torch.cat((vertice_sorting_method, verts_taxel_int), dim=2)
		for i in range(cbs):
			x = torch.unique(verts_taxel_int[i, :, :], sorted=True, return_inverse=False,
			                 dim=0)  # this takes the most time
			
			x[1:, 0] = torch.abs((x[:-1, 1] - x[1:, 1]) + (x[:-1, 2] - x[1:, 2]))
			x = x[x[:, 0] != 0, :]
			x = x[:, 1:]
			x = x[x[:, 1] < self.w, :]
			x = x[x[:, 1] >= 0, :]
			x = x[x[:, 0] < self.h, :]
			x = x[x[:, 0] >= 0, :]

			mesh_matrix = x[:, 2].view(self.h, self.w)
			
			if i == 0:
				mesh_matrix = mesh_matrix.transpose(0, 1).flip(0).unsqueeze(0)
				mesh_matrix_batch = mesh_matrix.clone()
			else:
				mesh_matrix = mesh_matrix.transpose(0, 1).flip(0).unsqueeze(0)
				mesh_matrix_batch = torch.cat((mesh_matrix_batch, mesh_matrix), dim=0)
		
		contact_matrix_batch = mesh_matrix_batch.clone()
		contact_matrix_batch[contact_matrix_batch >= 0] = 0
		contact_matrix_batch[contact_matrix_batch < 0] = 1
		
		mesh_matrix_batch[mesh_matrix_batch > 0] = 0
		mesh_matrix_batch *= -1
		
		return mesh_matrix_batch.type(self.dtype)


def sparse_batch_mm(m1, m2):
	"""
    https://github.com/pytorch/pytorch/issues/14489

    m1: sparse matrix of size N x M
    m2: dense matrix of size B x M x K
    returns m1@m2 matrix of size B x N x K
    """
	
	batch_size = m2.shape[0]
	# stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
	m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
	result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
		.transpose(1, 0)
	return result


def map_pressure_to_ground(vertices_hd, pressure_weights, grid_size=(64, 27),
                           resolution_y=0.7874, resolution_x=0.7874, m_x=0.0, m_y=0.4):
	device = vertices_hd.device
	B, N, _ = vertices_hd.shape
	ground_pressure_map = torch.zeros((B, grid_size[0], grid_size[1]), dtype=torch.float).to(device)
	ground_pressure_map_max = torch.zeros((B, grid_size[0], grid_size[1]), dtype=torch.float).to(device)
	
	# 将实际空间位置转换为传感器坐标
	sensor_per_cm_x = resolution_x
	sensor_per_cm_y = resolution_y
	grid_height, grid_width = grid_size
	half_width = grid_width / 2
	half_height = grid_height / 2
	
	for b in range(B):
		for i in range(N):
			x, y, _ = vertices_hd[b, i]
			
			# 转换点到传感垫的坐标
			sensor_x = int(100 * (x - m_x) * sensor_per_cm_x + half_width)
			sensor_y = int(100 * (y - m_y) * sensor_per_cm_y + half_height)
			
			if 0 <= sensor_x < grid_width and 0 <= sensor_y < grid_height:
				ground_pressure_map[b, sensor_y, sensor_x] += pressure_weights[b, i]
				ground_pressure_map_max[b, sensor_y, sensor_x] = torch.max(
					ground_pressure_map_max[b, sensor_y, sensor_x], pressure_weights[b, i]
				)
	
	return ground_pressure_map, ground_pressure_map_max


def ipman_pred(vertices, dataset, cop_w=10, cop_k=100):
	if dataset == 'moyo':
		resolution_x = 0.72
		resolution_y = 0.72
		m_x = 0.0
		m_y = 0.4
		grid_size = (110, 37)
	elif dataset == 'pressurepose':
		resolution_x = 0.35
		resolution_y = 0.35
		m_x = 0.65
		m_y = 0.95
		grid_size = (64, 27)
	else:
		resolution_x = 0.48
		resolution_y = 0.33
		m_x = 0.4
		m_y = 0.94
		grid_size = (56, 40)
	
	hd_data = np.load(HD_SMPL_MODEL)
	hd_operator = torch.sparse.FloatTensor(
		torch.tensor(hd_data['index_row_col']),
		torch.tensor(hd_data['values']),
		torch.Size(hd_data['size']))
	
	if hd_operator.device != vertices.device:
		hd_operator = hd_operator.to(vertices.device)
	
	vertices = vertices.to(torch.double)
	vertices_hd = sparse_batch_mm(hd_operator, vertices)
	
	ground_plane_height = 0.0
	vertex_height = (vertices_hd[:, :, 2] - ground_plane_height)
	
	inside_mask = (vertex_height < 0.0).float()
	outside_mask = (vertex_height >= 0.0).float()
	pressure_weights = inside_mask * (1 - cop_k * vertex_height) + outside_mask * torch.exp(-cop_w * vertex_height)
	
	pressure_map, pressure_map_max = map_pressure_to_ground(vertices_hd, pressure_weights, resolution_x=resolution_x,
	                                                        resolution_y=resolution_y, m_x=m_x, m_y=m_y,
	                                                        grid_size=grid_size)
	
	# 将pressure_map旋转180度
	pressure_map = torch.flip(pressure_map, dims=(1,))
	pressure_map_max = torch.flip(pressure_map_max, dims=(1,))
	
	return pressure_map_max


def pmr_pred(vertices, dataset):
	if dataset == 'tip':
		vertices[:, :, 2] -= 0.10
	pmr = PMR(vertices, dataset)
	return pmr.get_pressure()



