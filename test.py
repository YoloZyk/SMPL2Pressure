import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from lib.model.cvae import SMPL2PressureCVAE
from lib.dataset import create_dataset
from lib.utils.metrics import compute_metrics
from lib.utils.static import TIP_PATH, DATASET_META
from lib.utils.viz_utils import viz_pwm, viz_compare
from lib.baseline import pmr_pred, ipman_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to the experiment directory in output/')
    parser.add_argument('--ckpt', type=str, default='best_model.pth', help='Checkpoint name')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--baseline', action='store_true', default=False, help='Show baseline result or not')
    parser.add_argument('--viz', action='store_true', default=False, help='Visualize predictions')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载训练时的配置
    config_path = os.path.join(args.exp_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 2. 准备测试集
    dataset_name = cfg['dataset']['name']
    MAX_PRESSURE = DATASET_META[dataset_name]['max_p']
    
    if dataset_name == 'tip':
        dataset_kwargs = {
            'cfgs': {
                'dataset_path': TIP_PATH,
                'dataset_mode': cfg['dataset'].get('mode', 'unseen_group'),
                'curr_fold': cfg['dataset'].get('curr_fold', 1),
                'normalize': cfg['dataset'].get('normal', False),
                'device': str(device)
            }
        }
        test_set = create_dataset(dataset_name, mode='test', **dataset_kwargs)
    else:
        test_set = create_dataset(dataset_name, split='test', 
                                 normalize=cfg['dataset'].get('normal', False), 
                                 device=device)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    # 3. 初始化模型并加载权重
    model = SMPL2PressureCVAE(cfg).to(device)
    ckpt_path = os.path.join(args.exp_dir, 'ckpts', args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. 推理循环
    all_metrics = []
    ipman_metrics = []
    pmr_metrics = []
    print(f"Testing on {dataset_name}...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pmaps = batch['pressure'].to(device)
            verts = batch['vertices'].to(device)

            # cVAE 推理通常使用 inference 模式 (仅点云条件)
            pred_pmap = model.inference(verts) 

            if cfg['dataset'].get('normal', False):
                pmaps = pmaps * MAX_PRESSURE
                pred_pmap = pred_pmap * MAX_PRESSURE
            
            if cfg['dataset']['name'] == 'moyo':
                pred_pmap[pred_pmap < 0.05] = 0
            elif cfg['dataset']['name'] == 'tip':
                pred_pmap[pred_pmap < 1.0] = 0
            else:
                pred_pmap[pred_pmap < 0.1] = 0

            # 计算指标
            metrics = compute_metrics(
                pred_pmap.unsqueeze(1), 
                pmaps.unsqueeze(1), 
                is_normalized=False,
                max_val=MAX_PRESSURE
            )
            all_metrics.append(metrics)

            if args.baseline:
                ipman_res = ipman_pred(verts, cfg['dataset']['name'])
                pmr_res = pmr_pred(verts, cfg['dataset']['name'])
                i_metrics = compute_metrics(
                    ipman_res.unsqueeze(1), 
                    pmaps.unsqueeze(1), 
                    is_normalized=False,
                    max_val=MAX_PRESSURE
                )
                ipman_metrics.append(i_metrics)

                p_metrics = compute_metrics(
                    pmr_res.unsqueeze(1), 
                    pmaps.unsqueeze(1), 
                    is_normalized=False,
                    max_val=MAX_PRESSURE
                )
                pmr_metrics.append(p_metrics)

            if args.viz:
                if args.baseline:
                    viz_compare(verts, pmaps, ipman_res, pmr_res, pred_pmap)
                else:
                    viz_pwm(verts, pmaps, pred_pmap, save_path=None)


    # 5. 汇总结果
    avg_metrics = {}
    for k in all_metrics[0].keys():
        avg_metrics[k] = np.mean([m[k] for m in all_metrics])
    
    avg_ipman = {}
    for k in ipman_metrics[0].keys():
        avg_ipman[k] = np.mean([m[k] for m in ipman_metrics])
    
    avg_pmr = {}
    for k in pmr_metrics[0].keys():
        avg_pmr[k] = np.mean([m[k] for m in pmr_metrics])


    # 6. 保存到文件
    result_path = os.path.join(args.exp_dir, 'test_result.txt')
    with open(result_path, 'w') as f:
        f.write(f"Test Results for experiment: {args.exp_dir}\n")
        f.write(f"Checkpoint used: {args.ckpt}\n")
        f.write("-" * 30 + "\n")
        for k, v in avg_metrics.items():
            line = f"{k}: {v:.6f}\n"
            print(line, end='')
            f.write(line)
        f.write("-" * 30 + "\n")
        for k, v in avg_ipman.items():
            line = f"IPMAN {k}: {v:.6f}\n"
            print(line, end='')
            f.write(line)
        f.write("-" * 30 + "\n")
        for k, v in avg_pmr.items():
            line = f"PMR {k}: {v:.6f}\n"
            print(line, end='')
            f.write(line)
            
    print(f"\nResults saved to {result_path}")

if __name__ == '__main__':
    main()

