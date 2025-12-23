import os
import torch
import matplotlib.pyplot as plt
import trimesh
import pyrender

from lib.utils.static import SMPL_MODEL
from lib.utils.pyrender_visualization import PyRenderVisualizer


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
    viz_pressure(gt, pred, save_path=save_path)
    pv = PyRenderVisualizer(SMPL_MODEL)

    for i in range(4):
        pv.visualize_mesh(vertices=vertices[i])



