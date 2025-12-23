"""
MLP-based decoder for reconstructing point clouds from latent codes.

This module provides flexible MLP architectures for decoding latent representations
into point clouds, commonly used in generative models like VAE and cVAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PointCloudDecoder(nn.Module):
    """
    MLP-based decoder for reconstructing point clouds from latent representations.

    This decoder uses a series of fully connected layers with optional dropout
    and normalization to transform a latent code into point cloud coordinates.

    Args:
        latent_dim: Dimensionality of input latent code
        num_points: Number of output points in the point cloud
        point_dim: Dimensionality of each point (default: 3 for XYZ coordinates)
        hidden_dims: List of hidden layer dimensions (default: [1024, 2048])
        dropout_rate: Dropout probability (default: 0.3)
        use_batch_norm: Whether to use batch normalization (default: False)
        activation: Activation function to use (default: 'relu')

    Input:
        Latent code of shape (B, latent_dim)

    Output:
        Point cloud of shape (B, num_points, point_dim)
    """

    def __init__(
        self,
        latent_dim: int,
        num_points: int,
        point_dim: int = 3,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        super(PointCloudDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.point_dim = point_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [1024, 2048]
        self.hidden_dims = hidden_dims

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        output_dim = num_points * point_dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code into point cloud.

        Args:
            z: Latent code of shape (B, latent_dim)

        Returns:
            Reconstructed point cloud of shape (B, num_points, point_dim)
        """
        x = z

        # Process through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)

        # Output layer
        x = self.output_layer(x)

        # Reshape to point cloud
        x = x.view(-1, self.num_points, self.point_dim)

        return x

    def get_output_shape(self) -> tuple:
        """
        Get the output point cloud shape (excluding batch dimension).

        Returns:
            Tuple of (num_points, point_dim)
        """
        return (self.num_points, self.point_dim)


class ResidualMLPDecoder(nn.Module):
    """
    MLP decoder with residual connections for improved gradient flow.

    This decoder uses residual blocks to help with training deep networks.

    Args:
        latent_dim: Dimensionality of input latent code
        num_points: Number of output points in the point cloud
        point_dim: Dimensionality of each point (default: 3 for XYZ coordinates)
        hidden_dim: Hidden layer dimension (default: 1024)
        num_blocks: Number of residual blocks (default: 3)
        dropout_rate: Dropout probability (default: 0.3)
        use_batch_norm: Whether to use batch normalization (default: True)

    Input:
        Latent code of shape (B, latent_dim)

    Output:
        Point cloud of shape (B, num_points, point_dim)
    """

    def __init__(
        self,
        latent_dim: int,
        num_points: int,
        point_dim: int = 3,
        hidden_dim: int = 1024,
        num_blocks: int = 3,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(ResidualMLPDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim

        # Initial projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate, use_batch_norm)
            for _ in range(num_blocks)
        ])

        # Output projection
        output_dim = num_points * point_dim
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code into point cloud.

        Args:
            z: Latent code of shape (B, latent_dim)

        Returns:
            Reconstructed point cloud of shape (B, num_points, point_dim)
        """
        # Initial projection
        x = F.relu(self.input_proj(z))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)
        x = x.view(-1, self.num_points, self.point_dim)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block with optional batch normalization and dropout.

    Args:
        hidden_dim: Dimension of the hidden layer
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn1 = None
            self.bn2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor of shape (B, hidden_dim)

        Returns:
            Output tensor of shape (B, hidden_dim)
        """
        residual = x

        out = self.fc1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        out = out + residual
        out = F.relu(out)

        return out


def create_pointcloud_decoder(
    latent_dim: int,
    num_points: int = 6890,
    point_dim: int = 3,
    architecture: str = 'mlp',
    **kwargs
) -> nn.Module:
    """
    Factory function to create a point cloud decoder with specified architecture.

    Args:
        latent_dim: Dimensionality of input latent code
        num_points: Number of output points (default: 6890 for SMPL mesh)
        point_dim: Dimensionality of each point (default: 3 for XYZ)
        architecture: Decoder architecture type ('mlp' or 'residual')
        **kwargs: Additional arguments passed to the decoder constructor

    Returns:
        Configured decoder instance

    Examples:
        >>> # Create a simple MLP decoder
        >>> decoder = create_pointcloud_decoder(
        ...     latent_dim=256,
        ...     num_points=6890,
        ...     architecture='mlp',
        ...     hidden_dims=[1024, 2048],
        ...     dropout_rate=0.3
        ... )

        >>> # Create a residual decoder
        >>> decoder = create_pointcloud_decoder(
        ...     latent_dim=256,
        ...     num_points=6890,
        ...     architecture='residual',
        ...     hidden_dim=1024,
        ...     num_blocks=3
        ... )
    """
    if architecture == 'mlp':
        return PointCloudDecoder(
            latent_dim=latent_dim,
            num_points=num_points,
            point_dim=point_dim,
            **kwargs
        )
    elif architecture == 'residual':
        return ResidualMLPDecoder(
            latent_dim=latent_dim,
            num_points=num_points,
            point_dim=point_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


if __name__ == '__main__':
    print("Testing Point Cloud Decoders...\n")

    # Test parameters
    batch_size = 32
    latent_dim = 256
    num_points = 6890  # SMPL mesh vertices
    point_dim = 3

    # Test standard MLP decoder
    print("1. Standard MLP Decoder:")
    mlp_decoder = create_pointcloud_decoder(
        latent_dim=latent_dim,
        num_points=num_points,
        point_dim=point_dim,
        architecture='mlp',
        hidden_dims=[1024, 2048],
        dropout_rate=0.3,
        use_batch_norm=False
    )

    z = torch.randn(batch_size, latent_dim)
    output = mlp_decoder(z)
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output expected shape: {mlp_decoder.get_output_shape()}")
    print()

    # Test MLP decoder with batch norm
    print("2. MLP Decoder with Batch Normalization:")
    mlp_bn_decoder = create_pointcloud_decoder(
        latent_dim=latent_dim,
        num_points=num_points,
        architecture='mlp',
        use_batch_norm=True,
        activation='leaky_relu'
    )
    output = mlp_bn_decoder(z)
    print(f"  Output shape: {output.shape}")
    print()

    # Test residual MLP decoder
    print("3. Residual MLP Decoder:")
    residual_decoder = create_pointcloud_decoder(
        latent_dim=latent_dim,
        num_points=num_points,
        architecture='residual',
        hidden_dim=1024,
        num_blocks=3,
        dropout_rate=0.3
    )
    output = residual_decoder(z)
    print(f"  Input shape: {z.shape}")
    print(f"  Output shape: {output.shape}")
    print()

    # Test with different point dimensions
    print("4. Decoder with 6D points (XYZ + RGB):")
    decoder_6d = create_pointcloud_decoder(
        latent_dim=latent_dim,
        num_points=1000,
        point_dim=6,
        architecture='mlp',
        hidden_dims=[512, 1024]
    )
    output = decoder_6d(z)
    print(f"  Output shape: {output.shape}")
    print()
