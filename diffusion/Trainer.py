import os
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model.PressureDM import DiffModel
from .utils.padding import pad_to_multiple_of_8
from .utils.Scheduler import GradualWarmupScheduler
from lib.dataset import create_dataset
from lib.utils.static import TIP_PATH, DATASET_META


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


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


def setup_logger(name, log_dir=None, level=logging.INFO):
    """To setup as many loggers as you want"""
    lg = logging.getLogger(name)
    lg.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if log_dir is not None:
        fh = logging.FileHandler(log_dir, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # lg.addHandler(sh)
    

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        
        self.model = model
        self.T = T
        self.loss = nn.MSELoss()
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    
    def forward(self, x_0, condition):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        # loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        noise_bar, cond_pred = self.model(x_t, t, condition)
        x_loss = 100*self.loss(noise_bar, noise)
        cond_loss = 60*self.loss(cond_pred, condition) if cond_pred is not None else torch.tensor(0.0, device=x_0.device)

        return x_loss + cond_loss, x_loss, cond_loss
    
    
def train(args, device):
    setup_seed(47)
    current_time = get_timestamp()
    name = 'debug' if args.debug else 'train'
    setup_logger('base', os.path.join(args.output_dir, 'logs/{}_{}_{}_{}.log'.format(name, current_time, args.dataset[0], args.note)))
    logger = logging.getLogger('base')
    tb_logger = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb_logs/{}_{}_{}'.format(name, current_time, args.note)))
    # 日志记录配置信息
    logger.info(f'####################### Configurations ({device}) #######################')
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))
    
    logger.info('######################### Loading Data #########################')
    if args.dataset == 'tip':
        # TIP 数据集需要特殊的 cfgs 字典
        dataset_kwargs = {
            'cfgs': {
                'dataset_path': TIP_PATH,
                'dataset_mode': 'unseen_group',
                'curr_fold': 1,
                'normalize': args.normal,
                'device': device
            }
        }
        train_data = create_dataset(args.dataset, mode='train', **dataset_kwargs)
        # val_data = create_dataset(args.dataset, mode='val', **dataset_kwargs)
    else:
        # MoYo 和 PressurePose 的参数相对通用
        dataset_kwargs = {
            'split': 'train',
            'normalize': args.normal,
            'device': device
        }
        train_data = create_dataset(args.dataset, **dataset_kwargs)
        
        # dataset_kwargs['split'] = 'val'
        # val_data = create_dataset(args.dataset, **dataset_kwargs)

    max_P = DATASET_META[args.dataset]['max_p']
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # logger.info('Train: {} | Val: {}'.format(len(train_data), len(val_data)))
    logger.info('Train: {} '.format(len(train_data)))
    
    logger.info('######################### Building Model #########################')
    net_model = DiffModel(args).to(device)
    
    if args.cond_load_weight is not None:
        logger.info('######################### Loading Condition #########################')
        net_model.cond_encoder.load_state_dict(torch.load(args.cond_load_weight)['model_state_dict_en'], strict=False)
        # net_model.cond_decoder.load_state_dict(torch.load(args.cond_load_weight)['model_state_dict_de'], strict=False)
    
    if args.training_load_weight is not None:
        logger.info('######################### Loading Diffusion #########################')
        net_model.load_state_dict(torch.load(args.output_dir + 'checkpoint/' + args.training_load_weight), strict=False)
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier,
                                             warm_epoch=args.epochs // 10, after_scheduler=cosineScheduler)
    
    trainer = DiffusionTrainer(model=net_model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)
    logger.info("######################### Start Training #########################")
    
    # import pdb; pdb.set_trace()
    
    for epoch in range(args.epochs):
        step = 0
        # 记录不同模块耗时
        with tqdm(train_loader, dynamic_ncols=True) as tqdmLoader:
            for batch in tqdmLoader:
                # start_time = time.time()
                step += 1
                optimizer.zero_grad()
                
                vertices = batch['vertices'].to(device)
                pressure = batch['pressure'].to(device)
                smpl = batch['smpl'].to(device)
                
                b = pressure.shape[0]
                pressure = pad_to_multiple_of_8(pressure.unsqueeze(1))
                x_0 = (pressure / max_P) * 2 - 1
                if np.random.rand() < 0.1:
                    vertices = torch.zeros_like(vertices)
                
                # time1
                # data_load_time = time.time()
                # print(f"数据准备耗时：{data_load_time-start_time}s")
                
                loss, x_loss, cond_loss = trainer(x_0, vertices)
                # time2
                # diffusion_time = time.time()
                # print(f"扩散模型耗时：{diffusion_time-data_load_time}s")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), args.grad_clip)
                optimizer.step()
                # time3
                # update_time = time.time()
                # print(f"参数更新耗时：{update_time-diffusion_time}s")
                
                if step % args.print_freq == 0:
                    logger.info(f"Epoch: {epoch}/{args.epochs} Step: {step}/{len(train_loader)} "
                                f"PMRLoss: {x_loss.item():.6f} PCRLoss: {cond_loss.item():.6f} LR: {optimizer.state_dict()['param_groups'][0]['lr']}")
                    tb_logger.add_scalar('train_loss', loss.item(), step + epoch * len(train_loader))
                    tb_logger.add_scalar('train_pmr_loss', x_loss.item(), step + epoch * len(train_loader))
                    tb_logger.add_scalar('train_pcr_loss', cond_loss.item(), step + epoch * len(train_loader))
                tqdmLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "pmr_loss": x_loss.item(),
                    "pcr_loss": cond_loss.item(),
                    # "img shape": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                # time4
                # log_time = time.time()
                # print(f"日志记录耗时：{log_time-update_time}s")
                # break
                
        warmUpScheduler.step()
        
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            torch.save(net_model.state_dict(),
                       os.path.join(args.output_dir, f'ckpt/ckpt_{current_time}_{args.dataset[0]}_{args.note}_epoch_{epoch + 1}.pth'))
                
    logger.info('----------------------- Training Done ------------------------')
    tb_logger.close()
    
