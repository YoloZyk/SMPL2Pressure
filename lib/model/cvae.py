import os
import torch
import torch.nn as nn
from lib.model.resnet import resnet18, resnet34, resnet50
from lib.model.unet import UNetEncoder, UNetDecoder
from lib.model.pointnet import create_pointnet_encoder
from lib.model.mlp import create_pointcloud_decoder
from lib.utils.static import DATASET_META


class SMPL2PressureCVAE(nn.Module):
    """
    SMPL2Pressure cVAE model with dual branches.
    Main Branch: Pressure map encoding and reconstruction.
    Condition Branch: Point cloud encoding and reconstruction.
    """
    def __init__(self, cfg):
        super(SMPL2PressureCVAE, self).__init__()
        self.cfg = cfg
        self.embed_dim = cfg['model']['embed_dim']
        self.cond_embed_dim = cfg['model']['cond_embed_dim']
        
        # 1. Main Encoder (Pressure Map -> z_params)
        # Supports ResNet or UNet as the visual encoder
        main_enc_type = cfg['model']['main_encoder']['type']
        if "resnet" in main_enc_type:
            # Using the ResNet implementation provided
            model_func = eval(main_enc_type)
            self.main_encoder = model_func(
                embed_dim=self.embed_dim, 
                cond_embed_dim=self.cond_embed_dim,
                dp_rate=cfg['model']['dropout_rate']
            )
        else:
            self.main_encoder = UNetEncoder(
                cond_dim=self.cond_embed_dim,
                embed_dim=self.embed_dim,
                crop=DATASET_META[cfg['dataset']['name']]['crop_size'] # Need to pass this from config
            )

        # 2. Condition Encoder (Point Cloud -> cond_features)
        self.cond_encoder = create_pointnet_encoder(
            input_dim=3,
            feature_dim=self.cond_embed_dim,
            use_spatial_transformer=cfg['model']['cond_encoder']['use_spatial_transformer'],
            return_global_feature=cfg['model']['cond_encoder']['return_global_feature']
        )

        # 3. Main Decoder (z + cond -> Pressure Map)
        self.main_decoder = UNetDecoder(
            cond_dim=self.cond_embed_dim,
            embed_dim=self.embed_dim,
            bilinear=cfg['model']['main_decoder']['bilinear'],
            crop=DATASET_META[cfg['dataset']['name']]['crop_size']
        )

        # 4. Condition Decoder (cond_features -> Point Cloud)
        self.cond_decoder = create_pointcloud_decoder(
            latent_dim=self.embed_dim,
            num_points=6890, # SMPL vertices
            architecture=cfg['model']['cond_decoder']['type']
        )

    def load_pretrained_cond(self, path):
        """
        专门用于加载预训练的点云编解码分支权重
        """
        if not os.path.exists(path):
            print(f"Warning: Pretrained path {path} not found. Training from scratch.")
            return

        print(f"Loading pretrained condition branch from {path}...")
        ckpt = torch.load(path, map_location='cpu')
        
        # 加载 encoder 和 decoder
        self.cond_encoder.load_state_dict(ckpt['cond_encoder'])
        self.cond_decoder.load_state_dict(ckpt['cond_decoder'])
        
        # 可选：如果你希望预训练的组件在主训练初期不被破坏，可以冻结它们
        # for param in self.cond_encoder.parameters(): param.requires_grad = False

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample z from N(mu, var)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pressure_map, vertices):
        """
        Forward pass during training.
        Args:
            pressure_map: (B, 1, H, W)
            vertices: (B, 6890, 3)
        """
        # A. Encode Point Cloud to get condition (Condition branch)
        # PointNet expects (B, 3, N)
        pts = vertices.transpose(2, 1)
        cond_feat = self.cond_encoder(pts) # (B, cond_embed_dim)

        # B. Encode Pressure Map with condition to get latent distribution
        # Note: resnet implementation already handles concatenation inside
        mu, log_var = self.main_encoder(pressure_map, cond_feat)
        
        # C. Sample z
        z = self.reparameterize(mu, log_var)

        # D. Reconstruct Pressure Map
        recon_pressure = self.main_decoder(z, cond_feat)

        # E. Reconstruct Point Cloud (Side task for better latent space)
        recon_vertices = self.cond_decoder(cond_feat)

        return {
            'recon_pressure': recon_pressure,
            'recon_vertices': recon_vertices,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }

    @torch.no_grad()
    def inference(self, vertices):
        """
        Inference: Generate pressure from Point Cloud only.
        """
        self.eval()
        # 1. Encode Point Cloud
        pts = vertices.transpose(2, 1)
        cond_feat = self.cond_encoder(pts)

        # 2. Sample z from prior N(0, 1)
        z = torch.randn(vertices.size(0), self.embed_dim).to(vertices.device)

        # 3. Decode to Pressure Map
        gen_pressure = self.main_decoder(z, cond_feat)
        
        return gen_pressure
    

if __name__ == "__main__":
    import yaml

    cfg_path = 'config/config_base.yaml'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model = SMPL2PressureCVAE(cfg).to(device)

    pressure = torch.randn(8, 1, 56, 40).to(device)
    vertices = torch.randn(8, 6890, 3).to(device)

    res = model(pressure, vertices)

    pred_pressure = model.inference(vertices)

    import pdb; pdb.set_trace()




