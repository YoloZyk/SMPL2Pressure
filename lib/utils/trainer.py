import os
import time
import yaml
import torch
import logging
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from lib.model.loss import SMPL2PressureLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        
        self.save_freq = cfg['training'].get('save_freq', 999)
        self.log_freq = cfg['training'].get('log_freq', 10)
        self.val_freq = cfg['training'].get('val_freq', 1)

        # 1. 创建输出目录
        self.exp_name = cfg['dataset']['name']
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = os.path.join('output', f"{self.exp_name}_{self.timestamp}")
        
        self.ckpt_dir = os.path.join(self.out_dir, 'ckpts')
        self.log_dir = os.path.join(self.out_dir, 'logs')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 2. 保存原始配置
        with open(os.path.join(self.out_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)

        # 3. 设置日志记录
        self._init_logger()
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        
        # 4. 损失函数、优化器与调度器
        self.criterion = SMPL2PressureLoss(cfg)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _init_logger(self):
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # 控制台输出
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def _get_optimizer(self):
        opt_cfg = self.cfg['training']['optimizer']
        lr = self.cfg['training']['learning_rate']
        if opt_cfg['type'] == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=opt_cfg['weight_decay'])
        elif opt_cfg['type'] == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=opt_cfg['weight_decay'])
        return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def _get_scheduler(self):
        sch_cfg = self.cfg['training']['scheduler']
        if sch_cfg['type'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg['training']['num_epochs'])
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=sch_cfg['step_size'], gamma=sch_cfg['gamma'])

    def train_epoch(self):
        self.model.train()
        total_losses = {k: 0.0 for k in ['loss', 'loss_pmr', 'loss_pcr', 'loss_kl']}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for i, batch in enumerate(pbar):
            # 数据移动到设备
            pmaps = batch['pressure'].unsqueeze(1).to(self.device)   # (B, 1, H, W)
            verts = batch['vertices'].to(self.device)      # (B, 6890, 3)

            self.optimizer.zero_grad()
            outputs = self.model(pmaps, verts)
            
            loss_dict = self.criterion(outputs, pmaps, verts)
            loss_dict['loss'].backward()
            self.optimizer.step()

            self.global_step += 1

            # 统计
            for k in total_losses.keys():
                total_losses[k] += loss_dict[k].item()

            if i % self.log_freq == 0:
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Iter/{k}', v.item(), self.global_step)
            
            pbar.set_postfix(loss=loss_dict['loss'].item(), pmr=loss_dict['loss_pmr'].item())

        # 平均损失记录
        avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        for batch in self.val_loader:
            pmaps = batch['pressure'].unsqueeze(1).to(self.device)
            verts = batch['vertices'].to(self.device)
            outputs = self.model(pmaps, verts)
            loss_dict = self.criterion(outputs, pmaps, verts)
            val_loss += loss_dict['loss'].item()
        
        return val_loss / len(self.val_loader)

    def fit(self):
        self.logger.info(f"Starting training on {self.device}: {self.exp_name}...")
        for epoch in range(self.cfg['training']['num_epochs']):
            self.current_epoch = epoch
            
            # 训练与验证
            train_losses = self.train_epoch()
            self.scheduler.step()

            if epoch % self.val_freq == 0:
                val_loss = self.validate()
                
                # 记录 Epoch 级指标
                for k, v in train_losses.items():
                    self.writer.add_scalar(f'Epoch_Train/{k}', v, epoch)
                self.writer.add_scalar('Epoch_Val/TotalLoss', val_loss, epoch)
                self.writer.add_scalar('Misc/LR', self.optimizer.param_groups[0]['lr'], epoch)

                self.logger.info(f"Epoch {epoch} | Train Loss: {train_losses['loss']:.4f} | Val Loss: {val_loss:.4f}")

                # 3. 保存逻辑
                # 始终保存最新一个 (latest)
                self.save_checkpoint('latest.pth')

                # 保存最佳一个 (best)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                    self.logger.info(f"--> Best model saved at epoch {epoch}")

            # 按频率保存固定 checkpoint
            if epoch != 0 and epoch % self.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')

        self.writer.close()
        self.logger.info("Training complete.")

    def save_checkpoint(self, filename):
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), 
            'best_val_loss': self.best_val_loss,
            'config': self.cfg
        }
        torch.save(state, os.path.join(self.ckpt_dir, filename))


