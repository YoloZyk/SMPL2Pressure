import torch
import torch.nn as nn

from diffusion.model.CondEncoder import PointNet
from diffusion.model.CondDecoder import PCDecoder
from diffusion.model.unet import UNet


class DiffModel(nn.Module):
    def __init__(self, args):
        super(DiffModel, self).__init__()
        
        self.cond_encoder = PointNet(k=3, d_model=args.cond_dim+64, global_feat=True, tran=False)
        self.cond_decoder = PCDecoder(cond_embed_dim=args.cond_dim, dp_rate=args.dp_rate)
        self.unet = UNet(T=args.T, cond_dim=args.cond_dim, ch=args.channel, ch_mult=args.channel_mult,
                     num_res_blocks=args.num_res_blocks, dropout=args.dp_rate, in_ch=1, f_out_ch=1)

    def forward(self, x, t, cond):
        cond_z = self.cond_encoder(cond.permute(0, 2, 1))
        cond_pred = self.cond_decoder(cond_z)
        x_output = self.unet(x, t, cond_z)
        return x_output, cond_pred
    
    

