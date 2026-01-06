import os
import torch
import smplx
import matplotlib.pyplot as plt
import trimesh
import pyrender

from lib.utils.static import SMPL_MODEL
from lib.utils.pyrender_visualization import PyRenderVisualizer
from lib.utils.mesh_utils import batch_face_normals
from lib.utils.stability import StabilityLossCoP


def cal_cop(pressure):
    B, Y, X = pressure.shape
    
    x_coords = torch.arange(X, device=pressure.device)  # [0, 1, 2, ..., 36]
    y_coords = torch.arange(Y, device=pressure.device)  # [0, 1, 2, ..., 109]
    
    total_pressure = pressure.sum(dim=(1, 2))  # shape: (B,)
    total_pressure = torch.where(total_pressure == 0, torch.tensor(1e-8, device=pressure.device), total_pressure)
    
    x_weighted_sum = (pressure * x_coords).sum(dim=(1, 2))  # shape: (B,)
    y_weighted_sum = (pressure * y_coords.view(-1, 1)).sum(dim=(1, 2))  # shape: (B,)
    
    x_centers = x_weighted_sum / total_pressure  # shape: (B,)
    y_centers = y_weighted_sum / total_pressure  # shape: (B,)
    
    cop_sensor_xy = torch.stack([x_centers, y_centers], dim=1)
    
    return cop_sensor_xy.cpu().detach()

def viz_pressure(pressure, pred, save_path=None):
    # print('Visualizing pressure and predicted pressure...')
    pressure = pressure.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    fig, ax = plt.subplots(2, 4, figsize=(10, 8))
    for i in range(4):
        ax[0, i].imshow(pred[i], cmap='viridis', interpolation='nearest')
        ax[1, i].imshow(pressure[i], cmap='viridis', interpolation='nearest')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
    
    # 为第一行添加标题（预测）
    ax[0, 0].text(-0.1, 0.5, 'Predicted', va='center', ha='right',
                  rotation=90, transform=ax[0, 0].transAxes,
                  fontsize=12, fontweight='bold')
    
    # 为第二行添加标题（实际）
    ax[1, 0].text(-0.1, 0.5, 'GroundTruth', va='center', ha='right',
                  rotation=90, transform=ax[1, 0].transAxes,
                  fontsize=12, fontweight='bold')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def viz_pwm(vertices, gt, pred, save_path=None):
    if gt and pred:
        viz_pressure(gt, pred, save_path=save_path)
    if gt is None or pred is None:
        pmap = pred if pred is not None else gt
        viz_pressure(pmap, pmap, save_path=save_path)

    pv = PyRenderVisualizer(SMPL_MODEL)

    for i in range(4):
        pv.visualize_mesh(vertices=vertices[i])


def viz_compare(
        vertices: torch.Tensor,
        gt: torch.Tensor,
        baseline_a: torch.Tensor,
        baseline_b: torch.Tensor,
        pred: torch.Tensor,
        num_samples: int = 2,
        show_colorbar: bool = False
):
    """
    可视化对比压力图。

    Args:
        vertices (torch.Tensor): 输入点云
        gt (torch.Tensor): 参考真实压力图
        baseline_a (torch.Tensor): 第一个基线方法预测的压力图张量。
        baseline_b (torch.Tensor): 第二个基线方法预测的压力图张量。
        pred (torch.Tensor): 你的方法预测的压力图张量。
        num_samples (int): 要可视化的样本数量，默认为4。
        show_colorbar (bool): 是否显示颜色条，默认为False。
    """
    # 确保所有张量都在 CPU 上以便可视化
    gt = gt.cpu().detach()
    baseline_a = baseline_a.cpu().detach()
    baseline_b = baseline_b.cpu().detach()
    pred = pred.cpu().detach()
    
    B, H, W = gt.shape
    if H == 110 and W == 37:
        dataset = 'moyo'
    elif H == 64 and W == 27:
        dataset = 'pressurepose'
    elif H == 56 and W == 40:
        dataset = 'tip'
    else:
        dataset = None
    
    smpl_model = smplx.create(SMPL_MODEL, model_type='smpl', gender='neutral', ext='pkl').to('cuda:0')
    caler = StabilityLossCoP(smpl_model.faces, device='cuda:1')
    com = caler.get_com_mat(vertices.to('cuda:1'), dataset)
    
    cop_gt = cal_cop(gt)
    cop_ipman = cal_cop(baseline_a)
    cop_pmr = cal_cop(baseline_b)
    cop_ours = cal_cop(pred)

    # import pdb; pdb.set_trace()
    
    # 限制可视化的样本数量，最多为批次大小
    num_samples = min(num_samples, gt.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 4))
    
    # 如果只有一个样本，axes 可能是 1D 数组，需要将其转换为 2D 以便迭代
    if num_samples == 1:
        axes = [axes]
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for i in range(num_samples):
        # GT
        ax_gt = axes[i][0]
        im_gt = ax_gt.imshow(gt[i, :, :].squeeze(), cmap='viridis', interpolation='nearest')
        ax_gt.scatter(com[i, 0], com[i, 1], c='orange', marker='o', s=100, label='COM')
        ax_gt.scatter(cop_gt[i, 0], cop_gt[i, 1], c='magenta', marker='*', s=100, label='CoP_GT')
        ax_gt.set_title(f'GT - Sample {i + 1}')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.axis('off')
        
        # Baselines
        ax_ba = axes[i][1]
        im_ba = ax_ba.imshow(baseline_a[i, :, :].squeeze(), cmap='viridis', interpolation='nearest')
        ax_ba.scatter(com[i, 0], com[i, 1], c='orange', marker='o', s=100, label='COM')
        ax_ba.scatter(cop_ipman[i, 0], cop_ipman[i, 1], c='cyan', marker='*', s=100, label='CoP_IPMAN')
        ax_ba.set_title(f'IPMAN - Sample {i + 1}')
        ax_ba.set_xticks([])
        ax_ba.set_yticks([])
        ax_ba.axis('off')
        
        ax_bb = axes[i][2]
        im_bb = ax_bb.imshow(baseline_b[i, :, :].squeeze(), cmap='viridis', interpolation='nearest')
        ax_bb.scatter(com[i, 0], com[i, 1], c='orange', marker='o', s=100, label='COM')
        ax_bb.scatter(cop_pmr[i, 0], cop_pmr[i, 1], c='#FFFFFF', marker='*', s=100, label='CoP_PMR')
        ax_bb.set_title(f'PMR - Sample {i + 1}')
        ax_bb.set_xticks([])
        ax_bb.set_yticks([])
        ax_bb.axis('off')
        
        # Pred
        ax_pred = axes[i][3]
        im_pred = ax_pred.imshow(pred[i, :, :].squeeze(), cmap='viridis', interpolation='nearest')
        ax_pred.scatter(com[i, 0], com[i, 1], c='orange', marker='o', s=100, label='COM')
        ax_pred.scatter(cop_ours[i, 0], cop_ours[i, 1], c='#ff3e3e', marker='*', s=100, label='CoP_Ours')
        ax_pred.set_title(f'Ours - Sample {i + 1}')
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.axis('off')
        
        if show_colorbar:
            # 在最右侧添加颜色条
            cbar_ax = fig.add_axes([0.92, ax_pred.get_position().y0, 0.02, ax_pred.get_position().height])
            fig.colorbar(im_pred, cax=cbar_ax)
    
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.suptitle('Pressure Map Visualization Comparison', fontsize=18)
    plt.show()

    pv = PyRenderVisualizer(SMPL_MODEL)
    for i in range(num_samples):
        pv.visualize_mesh(vertices=vertices[i])



