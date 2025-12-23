import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader

from lib.dataset import create_dataset
from lib.model.cvae import SMPL2PressureCVAE
from lib.utils.trainer import Trainer
from lib.utils.static import TIP_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Train SMPL2Pressure cVAE")
    parser.add_argument('--config', type=str, default='config/config_base.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Override device in config')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # 1. 设备配置
    device = args.device if args.device else cfg.get('device', 'cuda')
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 2. 数据准备
    dataset_name = cfg['dataset']['name']
    
    # 适配不同数据集的参数构造
    if dataset_name == 'tip':
        # TIP 数据集需要特殊的 cfgs 字典
        dataset_kwargs = {
            'cfgs': {
                'dataset_path': TIP_PATH,
                'dataset_mode': cfg['dataset'].get('mode', 'unseen_group'),
                'curr_fold': cfg['dataset'].get('curr_fold', 1),
                'normalize': cfg['dataset'].get('normal', False),
                'device': str(device)
            }
        }
        train_set = create_dataset(dataset_name, mode='train', **dataset_kwargs)
        val_set = create_dataset(dataset_name, mode='eval', **dataset_kwargs)
    else:
        # MoYo 和 PressurePose 的参数相对通用
        dataset_kwargs = {
            'split': 'train',
            'normalize': cfg['dataset'].get('normal', False),
            'device': device
        }
        train_set = create_dataset(dataset_name, **dataset_kwargs)
        
        dataset_kwargs['split'] = 'val'
        val_set = create_dataset(dataset_name, **dataset_kwargs)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=False,
    )

    # 3. 模型初始化
    # 注意：确保 cfg 传递给了模型，以便内部建立不同的 Encoder/Decoder
    model = SMPL2PressureCVAE(cfg)

    # 3.1 加载预训练条件分支
    pretrained_path = cfg['model'].get('pretrained_cond_path', None)
    if pretrained_path:
        model.load_pretrained_cond(pretrained_path)

    # 4. 启动训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device
    )

    # 记录一些基本信息
    print(f"--- Configuration Loaded ---")
    print(f"Dataset: {dataset_name}")
    print(f"Main Arch: {cfg['model']['main_encoder']['type']} + {cfg['model']['main_decoder']['type']}")
    print(f"PCR Arch: {cfg['model']['cond_encoder']['type']} + {cfg['model']['cond_decoder']['type']}")
    print(f"Device: {device}")
    print(f"Output Directory: {trainer.out_dir}")
    print(f"----------------------------")

    trainer.fit()

if __name__ == '__main__':
    main()




