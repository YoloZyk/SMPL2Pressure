import os
import yaml
import glob
import smplx
import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from lib.model.cvae import SMPL2PressureCVAE
from lib.dataset import create_dataset
from lib.utils.sampler_util import sample_beta, sample_transl4t
from lib.utils.static import TIP_PATH, DATASET_META, SMPL_MODEL
from lib.utils.viz_utils import viz_pwm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='rand or sbjt')
    parser.add_argument('--pose_dir', type=str, default="/workspace/zyk/motion_generation/outputs/fm_tip_20251223_154811/samples/", help='Path to the pose directory')
    parser.add_argument('--output_dir', type=str, default='/workspace/zyk/public_data/SynthTIP/', help='Path to the output directory')
    parser.add_argument('--ckpt_dir', type=str, default='output/tip_20251223_151400', help='Path to the ckpt directory in output/')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--viz', action='store_true', default=False, help='Visualize predictions')
    return parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def main():
    args = parse_args()
    setup_seed(args.seed)
    
    # 1. 加载训练时的配置
    config_path = os.path.join(args.ckpt_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    dataset_name = cfg['dataset']['name']
    MAX_PRESSURE = DATASET_META[dataset_name]['max_p']

    # 2. 初始化模型并加载权重
    model = SMPL2PressureCVAE(cfg).to(device)
    ckpt_path = os.path.join(args.ckpt_dir, 'ckpts', 'best_model.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. 加载SMPL
    smpl_model = smplx.create(SMPL_MODEL, model_type='smpl', gender='neutral', ext='pkl').to(device)

    # 3. 准备待生成数据
    with torch.no_grad():
        if args.mode == 'rand':
            dataset_kwargs = {
                'cfgs': {
                    'dataset_path': TIP_PATH,
                    'dataset_mode': cfg['dataset'].get('mode', 'unseen_subject'),
                    'curr_fold': cfg['dataset'].get('curr_fold', 3),
                    'normalize': cfg['dataset'].get('normal', True),
                    'device': str(device)
                }
            }
            train_data = create_dataset('tip', mode='train', **dataset_kwargs)
            data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

            save_path = os.path.join(args.output_dir, "random", "lying")
            os.makedirs(save_path, exist_ok=True)
            file_cnt = len(os.listdir(save_path))
            print(f"Generating {len(train_data)} Samples")

            for data in tqdm(data_loader, desc=f"Generating R"):
                global_orient = data['smpl'][:, :3].to(device)
                body_pose = data['smpl'][:, 3:72]
                betas = sample_beta(len(body_pose), range_limit=3, device=device)
                transl = sample_transl4t(len(body_pose), device=device)

                output = smpl_model(
                        betas = betas,
                        global_orient = global_orient,
                        body_pose = body_pose,
                        transl = transl,
                    )

                vertices = output.vertices

                # 坐标转换
                vertices[:, :, 1] = 1.80 - vertices[:, :, 1]
                vertices[:, :, 2] = -vertices[:, :, 2]

                pred_pmap = model.inference(vertices)
                if cfg['dataset']['normal']:
                    pred_pmap = pred_pmap * MAX_PRESSURE
                
                pred_pmap[pred_pmap < 1.0] = 0

                if args.viz:
                    viz_pwm(vertices, None, pred_pmap, save_path=None)

                # save
                for i in range(len(betas)):
                    with open(os.path.join(save_path, f'lying_{(i+file_cnt):06d}.pkl'), 'wb') as f:
                        pickle.dump({
                            "betas": betas[i].cpu(),
                            "transl": transl[i].cpu(),
                            "global_orient": global_orient[i].cpu(),
                            "body_pose": body_pose[i].cpu(),
                            "pressure": pred_pmap[i].cpu()
                        }, f)
                
                file_cnt += len(betas)
        elif args.mode == 'cross':
            print("Cross Subject Generation")

        else:
            pose_paths = os.path.join(args.pose_dir, '*.pt')
            pose_files = glob.glob(pose_paths)

            # import pdb; pdb.set_trace()

            all_pose = torch.empty(0)
            for pose_file in pose_files:
                pose = torch.load(pose_file)
                
                all_pose = torch.cat((all_pose, pose), dim=0)
            
            data_loader = DataLoader(all_pose, batch_size=args.batch_size, shuffle=False)
            
            print(f"Generating 9x{len(all_pose)} Samples")

            # for sid in range(9):
            for sid in [1]:
                beta_file = os.path.join(args.output_dir, str(sid), 'config.pkl')
                save_path = os.path.join(args.output_dir, str(sid), 'lying')
                os.makedirs(save_path, exist_ok=True)
                file_cnt = len(os.listdir(save_path))

                with open(beta_file, 'rb') as f:
                    beta = pickle.load(f)['betas']    # 10

                for poses in tqdm(data_loader, desc=f"Generating {sid}"):
                    global_orient = poses[:, :3].to(device)
                    body_pose = poses[:, 3:].to(device)
                    betas = beta.unsqueeze(0).repeat(len(body_pose), 1).to(device)
                    transl = sample_transl4t(len(body_pose), device=device)
                    
                    output = smpl_model(
                        betas = betas,
                        global_orient = global_orient,
                        body_pose = body_pose,
                        transl = transl,
                    )

                    vertices = output.vertices

                    # 坐标转换
                    vertices[:, :, 1] = 1.80 - vertices[:, :, 1]
                    vertices[:, :, 2] = -vertices[:, :, 2]

                    pred_pmap = model.inference(vertices)
                    if cfg['dataset']['normal']:
                        pred_pmap = pred_pmap * MAX_PRESSURE
                    
                    pred_pmap[pred_pmap < 1.0] = 0

                    if args.viz:
                        viz_pwm(vertices, None, pred_pmap, save_path=None)

                    # save
                    for i in range(len(betas)):
                        with open(os.path.join(save_path, f'lying_{(i+file_cnt):06d}.pkl'), 'wb') as f:
                            pickle.dump({
                                "betas": betas[i].cpu(),
                                "transl": transl[i].cpu(),
                                "global_orient": global_orient[i].cpu(),
                                "body_pose": body_pose[i].cpu(),
                                "pressure": pred_pmap[i].cpu()
                            }, f)
                    
                    file_cnt += len(betas)

    print("\nGenerate Completed!\n")

if __name__ == '__main__':
    main()

