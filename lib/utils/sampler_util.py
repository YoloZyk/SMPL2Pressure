import torch
import numpy as np
from scipy.stats import truncnorm


def sample_beta(batch_size=1, sampling_method='normal', range_limit=4.0, device='cpu'):
    """
    对SMPL模型的beta参数进行采样，返回1x10的PyTorch张量。

    参数:
        batch_size (int): 采样样本数量，默认为1。
        sampling_method (str): 采样方法，'uniform'（均匀采样）或'normal'（正态分布采样）。默认为'normal'。
        range_limit (float): beta参数的范围限制，默认为3.0（即[-3, 3]）。
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')

    返回:
        torch.Tensor: 形状为(batch_size, 10)的beta参数张量。

    异常:
        ValueError: 如果sampling_method不是'uniform'或'normal'。
    """
    beta_dim = 10  # SMPL beta参数维度
    
    if sampling_method == 'uniform':
        # 均匀分布采样
        beta = np.random.uniform(low=-range_limit, high=range_limit, size=(batch_size, beta_dim))
    elif sampling_method == 'normal':
        # 正态分布采样
        beta = np.random.normal(loc=0, scale=2, size=(batch_size, beta_dim))
        beta = np.clip(beta, -range_limit, range_limit)  # 限制在[-range_limit, range_limit]
    else:
        raise ValueError("sampling_method must be 'uniform' or 'normal'")
    
    # 转换为PyTorch张量
    beta_tensor = torch.tensor(beta, dtype=torch.float32).to(device)
    
    return beta_tensor


def sample_transl4pp(batch_size, device):
    """
    为 SMPL 模型生成全局平移参数 (transl)，与数据集分布相似。

    参数：
        batch_size (int): 采样数量
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')

    返回：
        transl (torch.Tensor): 形状为 (batch_size, 3) 的全局平移参数 [X, Y, Z]
    """
    # X 坐标：均匀分布在 [0.45, 0.85]
    x_min, x_max = 0.45, 0.85
    x = torch.rand(batch_size, 1) * (x_max - x_min) + x_min  # 均匀分布采样
    
    # Y 坐标：均匀分布在 [1.05, 1.45]
    y_min, y_max = 1.05, 1.45
    y = torch.rand(batch_size, 1) * (y_max - y_min) + y_min  # 均匀分布采样
    
    # 卧姿对应
    # Z 坐标：截断正态分布，均值 0.08，标准差 0.03，范围 [-0.02, 0.24]
    z_mean, z_std = 0.08, 0.03
    z_min, z_max = -0.02, 0.24
    # 计算截断正态分布的标准化边界
    a, b = (z_min - z_mean) / z_std, (z_max - z_mean) / z_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    z = truncnorm.rvs(a, b, loc=z_mean, scale=z_std, size=(batch_size, 1))
    z = torch.tensor(z, dtype=torch.float32)
    
    # # 站姿适应
    # # Z 坐标：均匀分布在 [0.75, 0.85]
    # z_min, z_max = 0.75, 0.85
    # z = torch.rand(batch_size, 1) * (z_max - z_min) + z_min  # 均匀分布采样
    
    # 组合 X, Y, Z
    transl = torch.cat([x, y, z], dim=1).to(device)
    return transl


def sample_transl4m(batch_size, device):
    """
    为 SMPL 模型生成全局平移参数 (transl)，与第二个数据集分布相似。
    
    参数：
        batch_size (int): 采样数量
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')
    
    返回：
        transl (torch.Tensor): 形状为 (batch_size, 3) 的全局平移参数 [X, Y, Z]
    """
    # X 坐标：截断正态分布，均值 0.030337209，标准差 0.059348222，范围 [-0.22423534, 0.3106258]
    x_mean, x_std = 0.030337209, 0.059348222
    # x_min, x_max = -0.22423534, 0.3106258
    x_min, x_max = -0.05, 0.1
    # 计算截断正态分布的标准化边界
    a_x, b_x = (x_min - x_mean) / x_std, (x_max - x_mean) / x_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    x = truncnorm.rvs(a_x, b_x, loc=x_mean, scale=x_std, size=(batch_size, 1))
    x = torch.tensor(x, dtype=torch.float32)

    # Y 坐标：截断正态分布，均值 0.5841795，标准差 0.2390917，范围 [-0.09659827, 1.2293766]
    y_mean, y_std = 0.5841795, 0.2390917
    # y_min, y_max = -0.09659827, 1.2293766
    y_min, y_max = 0.0, 1.2
    # 计算截断正态分布的标准化边界
    a_y, b_y = (y_min - y_mean) / y_std, (y_max - y_mean) / y_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    y = truncnorm.rvs(a_y, b_y, loc=y_mean, scale=y_std, size=(batch_size, 1))
    y = torch.tensor(y, dtype=torch.float32)

    # # Z 坐标：均匀分布在 [0.75, 0.85]
    # z_min, z_max = 0.75, 0.85
    # z = torch.rand(batch_size, 1) * (z_max - z_min) + z_min  # 均匀分布采样
    
    # z全零
    z = torch.zeros((batch_size, 1))
    
    # 组合 X, Y, Z
    transl = torch.cat([x, y, z], dim=1).to(device)
    return transl


def sample_transl4t(batch_size, device):
    """
    为 SMPL 模型生成全局平移参数 (transl)，与第三个数据集分布相似。

    参数：
        batch_size (int): 采样数量
        device (torch.device): 计算设备 (e.g., 'cuda' or 'cpu')

    返回：
        transl (torch.Tensor): 形状为 (batch_size, 3) 的全局平移参数 [X, Y, Z]
    """
    # X 坐标：截断正态分布，均值 0.35497144，标准差 0.08321648，范围 [0.10660601, 0.72766024]
    x_mean, x_std = 0.35497144, 0.08321648
    # x_min, x_max = 0.10660601, 0.72766024
    x_min, x_max = 0.15, 0.55
    # 计算截断正态分布的标准化边界
    a_x, b_x = (x_min - x_mean) / x_std, (x_max - x_mean) / x_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    x = truncnorm.rvs(a_x, b_x, loc=x_mean, scale=x_std, size=(batch_size, 1))
    x = torch.tensor(x, dtype=torch.float32)
    
    # Y 坐标：截断正态分布，均值 0.943629，标准差 0.0685662，范围 [0.7616181, 1.4328215]
    y_mean, y_std = 0.943629, 0.0685662
    # y_min, y_max = 0.7616181, 1.4328215
    y_min, y_max = 0.8, 1.1
    # 计算截断正态分布的标准化边界
    a_y, b_y = (y_min - y_mean) / y_std, (y_max - y_mean) / y_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    y = truncnorm.rvs(a_y, b_y, loc=y_mean, scale=y_std, size=(batch_size, 1))
    y = torch.tensor(y, dtype=torch.float32)
    
    # Z 坐标：截断正态分布，均值 -0.15257776，标准差 0.055761524，范围 [-0.44515115, 0.021567477]
    z_mean, z_std = -0.15257776, 0.055761524
    # z_min, z_max = -0.44515115, 0.021567477
    z_min, z_max = -0.18, -0.1
    # 计算截断正态分布的标准化边界
    a_z, b_z = (z_min - z_mean) / z_std, (z_max - z_mean) / z_std
    # 使用 scipy 的 truncnorm 生成截断正态分布采样
    z = truncnorm.rvs(a_z, b_z, loc=z_mean, scale=z_std, size=(batch_size, 1))
    z = torch.tensor(z, dtype=torch.float32)
    
    # 组合 X, Y, Z
    transl = torch.cat([x, y, z], dim=1).to(device)
    return transl




