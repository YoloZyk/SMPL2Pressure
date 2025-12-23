import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(pred, target):
    # pred, target: (B, 1, H, W)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    batch_ssim = []
    for i in range(pred_np.shape[0]):
        # 指定 data_range=1.0 如果是归一化后的，或根据实际值指定
        s = ssim(target_np[i, 0], pred_np[i, 0], data_range=target_np[i, 0].max() - target_np[i, 0].min() + 1e-8)
        batch_ssim.append(s)
    return np.mean(batch_ssim)

def compute_cop(pressure_map):
    """
    计算压力中心 (Center of Pressure)
    pressure_map: (B, 1, H, W)
    """
    B, C, H, W = pressure_map.shape
    device = pressure_map.device
    
    # 创建坐标网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # 计算总压力
    total_pressure = torch.sum(pressure_map, dim=(2, 3)) + 1e-8 # 避免除零
    
    # 计算重心坐标
    cop_x = torch.sum(pressure_map * grid_x, dim=(2, 3)) / total_pressure
    cop_y = torch.sum(pressure_map * grid_y, dim=(2, 3)) / total_pressure
    
    return torch.cat([cop_x, cop_y], dim=1) # (B, 2)

def compute_metrics(pred, target, is_normalized=False, max_val=100.0):
    """
    计算所有指标。如果输入是归一化的，MAE/MSE 会还原到原始量级。
    """
    pred_raw, target_raw = pred, target
    pred = pred / max_val
    target = target / max_val
    
    mre = F.l1_loss(pred, target).item()
    ssim_val = compute_ssim(pred, target)
    mae = F.l1_loss(pred_raw, target_raw).item()
    mse = F.mse_loss(pred_raw, target_raw).item()
    
    # 3. CoP Distance
    cop_pred = compute_cop(pred_raw)
    cop_target = compute_cop(target_raw)
    cop_dist = torch.norm(cop_pred - cop_target, dim=1).mean().item()
    
    return {
        'SSIM': ssim_val,
        'MAE': mae,
        'MSE': mse,
        'MRE': mre,
        'CoP_Dist': cop_dist
    }

