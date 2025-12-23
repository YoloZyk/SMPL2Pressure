import torch
import torch.nn as nn
import torch.nn.functional as F


def get_crop_config(crop):
    """
    Get configuration parameters based on crop size.

    Args:
        crop: List of [height, width] or None

    Returns:
        dict: Configuration including alpha, h, w, diffx, diffy
    """
    if crop is None:
        # Default values for TIP dataset (56x40)
        return {
            'alpha': 6,
            'h': 3,
            'w': 2,
            'diffx': [1, 0, 0, 0],
            'diffy': [1, 0, 0, 0]
        }
    elif crop == [64, 27]:  # PressurePose dataset
        return {
            'alpha': 4,
            'h': 4,
            'w': 1,
            'diffx': [1, 0, 1, 1],
            'diffy': [0, 0, 0, 0]
        }
    elif crop == [110, 37]:  # MOYO dataset
        return {
            'alpha': 12,
            'h': 6,
            'w': 2,
            'diffx': [0, 1, 0, 1],
            'diffy': [1, 1, 1, 0]
        }
    else:  # Default for other sizes
        return {
            'alpha': 6,
            'h': 3,
            'w': 2,
            'diffx': [1, 0, 0, 0],
            'diffy': [1, 0, 0, 0]
        }


class DoubleConv(nn.Module):
    """Double Convolutional Block with BN and ReLU"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class ConditionalBatchNorm2d(nn.Module):
    """Conditional BatchNorm2d to modulate output with condition vector"""
    
    def __init__(self, num_features, cond_dim):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)
    
    def forward(self, x, cond):
        gamma = self.gamma(cond).view(-1, self.num_features, 1, 1)
        beta = self.beta(cond).view(-1, self.num_features, 1, 1)
        return self.bn(x) * gamma + beta


class UpWithCondition(nn.Module):
    """Upscaling then double conv with condition modulation"""
    
    def __init__(self, in_channels, out_channels, cond_dim, diffX, diffY, bilinear=True):
        super(UpWithCondition, self).__init__()
        self.diffX = diffX
        self.diffY = diffY
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

        # Conditional batch norm for output modulation
        self.cond_bn = ConditionalBatchNorm2d(out_channels, cond_dim)
    
    def forward(self, x, cond):
        x = self.up(x)
        x = F.pad(x, [self.diffX // 2, self.diffX - self.diffX // 2, self.diffY // 2, self.diffY - self.diffY // 2])
        x = self.conv(x)
        x = self.cond_bn(x, cond)  # Modulate with condition
        return x


class UNetEncoder(nn.Module):
    def __init__(self, cond_dim=256, embed_dim=256, dp_rate=0.0, bilinear=False, crop=None):
        super(UNetEncoder, self).__init__()
        self.cond_dim = cond_dim
        self.crop = crop

        # Get crop-specific configuration
        crop_config = get_crop_config(crop)
        alpha = crop_config['alpha']

        # Encoder
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.dropout = nn.Dropout(dp_rate)

        # VAE latent space parameters
        self.fc_mu = nn.Linear((1024 // factor) * alpha + cond_dim, embed_dim)
        self.fc_log_var = nn.Linear((1024 // factor) * alpha + cond_dim, embed_dim)
    
    def forward(self, x, cond):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # B x (1024 // factor) x H x W

        # Flatten and concatenate with condition vector
        x5_flat = x5.view(x5.size(0), -1)
        x5_cond = torch.cat([x5_flat, cond], dim=1)
        
        # Compute mu and log_var for latent space
        mu = self.fc_mu(x5_cond)
        log_var = self.fc_log_var(x5_cond)
        
        return mu, log_var
        

class UNetDecoder(nn.Module):
    def __init__(self, cond_dim=256, embed_dim=256, dp_rate=0.0, bilinear=False, crop=None):
        super(UNetDecoder, self).__init__()
        self.bilinear = bilinear
        self.cond_dim = cond_dim
        self.crop = crop

        # Get crop-specific configuration
        crop_config = get_crop_config(crop)
        alpha = crop_config['alpha']
        self.h = crop_config['h']
        self.w = crop_config['w']
        diffx = crop_config['diffx']
        diffy = crop_config['diffy']

        factor = 2 if bilinear else 1

        # Map latent vector and condition back to decoder size
        self.fc_z = nn.Linear(embed_dim + cond_dim, (1024 // factor) * alpha)
        self.dropout = nn.Dropout(dp_rate)

        # Decoder with conditional batch norm
        self.up1 = UpWithCondition(1024, 512 // factor, cond_dim, diffX=diffx[0], diffY=diffy[0], bilinear=bilinear)
        self.up2 = UpWithCondition(512, 256 // factor, cond_dim, diffX=diffx[1], diffY=diffy[1], bilinear=bilinear)
        self.up3 = UpWithCondition(256, 128 // factor, cond_dim, diffX=diffx[2], diffY=diffy[2], bilinear=bilinear)
        self.up4 = UpWithCondition(128, 64, cond_dim, diffX=diffx[3], diffY=diffy[3], bilinear=bilinear)
        self.outc = OutConv(64, 1)
    
    def forward(self, z, cond):
        b, _ = z.shape
        factor = 2 if self.bilinear else 1

        # Decode latent vector concatenated with condition
        z_cond = torch.cat([z, cond], dim=1)
        z_decoded = self.fc_z(z_cond).view(b, 1024 // factor, self.h, self.w)
        z_decoded = self.dropout(z_decoded)

        # Decoder path with conditional modulation
        x = self.up1(z_decoded, cond)
        x = self.up2(x, cond)
        x = self.up3(x, cond)
        x = self.up4(x, cond)
        x = self.outc(x)

        return x.view(b, self.crop[0], self.crop[1])


if __name__ == "__main__":
    def reparameterize(mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Test data
    data = torch.randn(4, 1, 56, 40)
    condition = torch.randn(4, 85)

    # Initialize encoder and decoder
    encoder = UNetEncoder(cond_dim=85, embed_dim=256, bilinear=False, crop=[56, 40])
    decoder = UNetDecoder(cond_dim=85, embed_dim=256, bilinear=False, crop=[56, 40])

    # Forward pass through encoder
    mu, logvar = encoder(data, condition)
    print(f"Mu shape: {mu.shape}, Logvar shape: {logvar.shape}")

    # Reparameterization trick
    z = reparameterize(mu, logvar)
    print(f"Latent z shape: {z.shape}")

    # Forward pass through decoder
    reconstructed = decoder(z, condition)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Expected shape: [4, 56, 40]")

    assert reconstructed.shape == (4, 56, 40), f"Shape mismatch: {reconstructed.shape} vs (4, 56, 40)"
    print("Test passed!")

