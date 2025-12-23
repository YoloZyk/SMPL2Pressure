import torch
import torch.nn as nn
import torch.nn.functional as F

class SMPL2PressureLoss(nn.Module):
    """
    Combined loss for SMPL2Pressure cVAE.
    - Pressure Map Reconstruction (PMR): MSE Loss
    - Point Cloud Reconstruction (PCR): MSE Loss (from condition features)
    - KL Divergence: Regularization of latent space
    """
    def __init__(self, cfg):
        super(SMPL2PressureLoss, self).__init__()
        self.cfg_loss = cfg['training']['loss']
        
        # 权重配置
        self.pmr_weight = self.cfg_loss.get('pmr_weight', 10.0)
        self.pcr_weight = self.cfg_loss.get('pcr_weight', 6.0)
        self.kl_weight = self.cfg_loss.get('kl_weight', 2.0)

    def forward(self, outputs, target_pressure, target_vertices):
        """
        Args:
            outputs: cVAE模型的输出字典
            target_pressure: 真值压力图 (B, H, W) 或 (B, 1, H, W)
            target_vertices: 真值SMPL顶点 (B, 6890, 3)
        """
        # 1. 压力图重建损失 (PMR)
        recon_pressure = outputs['recon_pressure']
        # 统一维度: 确保 target 也是 (B, 1, H, W)
        if recon_pressure.dim() == 3:
            recon_pressure = recon_pressure.unsqueeze(1)

        if target_pressure.dim() == 3:
            target_pressure = target_pressure.unsqueeze(1)
        
        loss_pmr = F.mse_loss(recon_pressure, target_pressure, reduction='mean')

        # 2. 点云重建损失 (PCR)
        # 根据你的提醒，这里的 recon_vertices 应该是从 cond_features 解码出来的
        recon_vertices = outputs['recon_vertices']
        loss_pcr = F.mse_loss(recon_vertices, target_vertices, reduction='mean')

        # 3. KL 散度损失
        mu = outputs['mu']
        log_var = outputs['log_var']
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        loss_kl = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        # 4. 总损失加权
        total_loss = (self.pmr_weight * loss_pmr + 
                      self.pcr_weight * loss_pcr + 
                      self.kl_weight * loss_kl)

        return {
            'loss': total_loss,
            'loss_pmr': loss_pmr,
            'loss_pcr': loss_pcr,
            'loss_kl': loss_kl
        }
    
