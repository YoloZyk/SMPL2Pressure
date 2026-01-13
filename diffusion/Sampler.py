import os
import torch
import time
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .model.PressureDM import DiffModel
from .utils.padding import pad_to_multiple_of_8
from .Trainer import setup_seed, setup_logger, get_timestamp
from lib.dataset import create_dataset
from lib.utils.viz_utils import viz_pwm, viz_pressure
from lib.utils.static import TIP_PATH, DATASET_META
from lib.utils.metrics import compute_metrics


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# DDPM
class DiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super(DiffusionSampler, self).__init__()
        self.model = model
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, condition):
        # ????
        # var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = self.posterior_var
        var = extract(var, t, x_t.shape)
        # print(x_t.shape)
        eps, _ = self.model(x_t, t, condition)
        nonEps, _ = self.model(x_t, t, torch.zeros_like(condition).to(condition.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, condition, eta, num_steps):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, condition=condition)

            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0

        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class DiffusionSamplerDDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0., eta=0.0, num_steps=50):
        super().__init__()
        self.model = model
        self.T = T
        self.w = w
        self.eta = eta
        self.num_steps = num_steps
        
        # 注册基础参数
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册必要的缓冲区变量
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0_pred):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0_pred) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, t, condition):
        # 条件与非条件预测的组合
        eps_cond, _ = self.model(x, t, condition)
        eps_uncond, _ = self.model(x, t, torch.zeros_like(condition).to(condition.device))
        eps = (1. + self.w) * eps_cond - self.w * eps_uncond
        
        x0_pred = self.predict_start_from_noise(x, t, eps)
        return eps, x0_pred
    
    def forward(self, x_T, condition, eta=None, num_steps=None):
        # 参数覆盖逻辑
        eta = eta if eta is not None else self.eta
        step_size = self.T // self.num_steps if num_steps is None else self.T // num_steps
        
        # 创建时间步序列
        times = torch.arange(0, self.T, step_size, device=x_T.device)
        times = list(times.flip(0))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), ...]
        
        x_t = x_T
        for time, time_next in time_pairs:
            # 创建时间张量
            t = x_t.new_ones(x_T.shape[0], dtype=torch.long) * time
            t_next = x_t.new_ones(x_T.shape[0], dtype=torch.long) * time_next
            
            # 模型预测
            eps, x0_pred = self.model_predictions(x_t, t, condition)
            
            # 计算alpha相关参数
            alpha_cumprod = extract(self.alphas_cumprod, t, x_t.shape)
            alpha_cumprod_next = extract(self.alphas_cumprod, t_next, x_t.shape)
            
            # DDIM更新公式
            sigma = eta * ((1 - alpha_cumprod / alpha_cumprod_next) * (1 - alpha_cumprod_next) / (
                        1 - alpha_cumprod)).sqrt()
            c = (1 - alpha_cumprod_next - sigma ** 2).sqrt()
            
            # 更新x_t
            x_t = x0_pred * alpha_cumprod_next.sqrt() + c * eps + sigma * torch.randn_like(x_t)
        
        return torch.clip(x_t, -1, 1)


def sample(args, device):
    setup_seed(47)
    current_time = get_timestamp()
    
    setup_logger('test', os.path.join(args.output_dir, 'logs/test_{}.log'.format(current_time)))
    logger = logging.getLogger('test')
    # 日志记录配置信息
    logger.info(f'####################### Configurations ({device}) #######################')
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))
    
    logger.info('####################### Loading data&model #######################')
    if args.dataset == 'tip':
        dataset_kwargs = {
            'cfgs': {
                'dataset_path': TIP_PATH,
                'dataset_mode': 'unseen_group',
                'curr_fold': 1,
                'normalize': args.normal,
                'device': device
            }
        }
        test_data = create_dataset(args.dataset, mode='test', **dataset_kwargs)
    else:
        test_data = create_dataset(args.dataset, split='test', 
                                 normalize=args.normal, device=device)
        
    max_p = DATASET_META[args.dataset]['max_p']
    # max_p = 128
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    logger.info(f'test: {len(test_data)}  ckpt: {args.ckpt}')
    
    net_model = DiffModel(args).to(device)
    net_model.load_state_dict(torch.load(args.output_dir + args.ckpt), strict=False)
    net_model.eval()
    
    logger.info('####################### Building Sampler #######################')
    # sampler = DiffusionSampler(model=net_model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T, w=args.w).to(device)
    sampler = DiffusionSamplerDDIM(model=net_model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T, w=args.w).to(device)
    
    logger.info('####################### Start testing #######################')

    all_metrics = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc='Test: ')):
            vertices = data['vertices'].to(device)
            
            # import pdb; pdb.set_trace()

            data['pressure'] = pad_to_multiple_of_8(data['pressure'].unsqueeze(1)).squeeze()
            pressure = data['pressure'].to(device)
            
            B, H, W = pressure.shape
            noisyImage = torch.randn(size=[B, 1, H, W], device=device)
            
            sampledImg = sampler(noisyImage, vertices, args.eta, args.num_steps)
            sampledImg = (sampledImg + 1) / 2 * max_p
            
            pred = sampledImg.squeeze()
            
            if args.dataset == 'moyo':
                pred[pred < 0.05] = 0
            elif args.dataset == 'tip':
                pred[pred < 1.0] = 0
            else:
                pred[pred < 0.1] = 0

            metrics = compute_metrics(
                pred.unsqueeze(1), 
                pressure.unsqueeze(1), 
                is_normalized=False,
                max_val=max_p
            )
            all_metrics.append(metrics)

            if args.viz:
                # viz_pressure(pressure, pred, save_path=None)
                viz_pwm(vertices, pressure, pred, save_path=None)
    
    avg_metrics = {}
    for k in all_metrics[0].keys():
        avg_metrics[k] = np.mean([m[k] for m in all_metrics])

    file_name = args.ckpt.split('/')[-1].split('.')[0]
    result_path = os.path.join(args.output_dir, f'results/{file_name}_result.txt')
    
    # import pdb; pdb.set_trace()

    with open(result_path, 'w') as f:
        f.write(f"Test Results for experiment: {args.output_dir}\n")
        f.write(f"Checkpoint used: {args.ckpt}\n")
        f.write("-" * 30 + "\n")
        for k, v in avg_metrics.items():
            line = f"{k}: {v:.6f}\n"
            logger.info(line)
            f.write(line)
    
    logger.info(f"Results saved to: {result_path}")
    
