import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.dataset import create_dataset
from lib.model.cvae import SMPL2PressureCVAE # 我们直接从完整模型里拿组件

def pretrain():
    # 1. 加载基础配置
    with open('config/config_base.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 2. 准备数据集 (只需要顶点数据)
    dataset_name = cfg['dataset']['name']
    # 注意：这里直接复用你已有的 dataset 创建逻辑
    train_set = create_dataset(dataset_name, split='train', normalize=cfg['dataset']['normal'], device=device)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # 3. 初始化完整模型，但我们只关心 cond_ 部分
    model = SMPL2PressureCVAE(cfg).to(device)
    
    # 定义优化器：只优化条件分支
    optimizer = torch.optim.Adam(
        list(model.cond_encoder.parameters()) + list(model.cond_decoder.parameters()),
        lr=1e-3
    )
    criterion = nn.MSELoss()

    # 4. 训练循环
    epochs = 50 # 预训练通常不需要太久
    print("Starting Pre-training of Condition Branch...")
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch}")
        epoch_loss = 0
        
        for batch in pbar:
            verts = batch['vertices'].to(device) # (B, 6890, 3)
            
            optimizer.zero_grad()
            
            # 模拟 cvae 内部的 cond 数据流
            pts = verts.transpose(2, 1)
            cond_feat = model.cond_encoder(pts)
            recon_verts = model.cond_decoder(cond_feat)
            
            loss = criterion(recon_verts, verts)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(mse=loss.item())

    # 5. 保存预训练权重
    save_path = 'output/pointnet_pretrain.pth'
    torch.save({
        'cond_encoder': model.cond_encoder.state_dict(),
        'cond_decoder': model.cond_decoder.state_dict(),
    }, save_path)
    print(f"Pre-trained weights saved to {save_path}")

if __name__ == '__main__':
    pretrain()

