"""
PointNet implementation for point cloud feature extraction.

This module implements the PointNet architecture for encoding point clouds into
global or per-point features. It can be used as a conditional encoder in cVAE models.

Reference:
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    Charles R. Qi et al., CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialTransformer3D(nn.Module):
    """
    Spatial Transformer Network for 3D point clouds.

    Predicts a 3x3 transformation matrix to canonicalize input point clouds.
    """

    def __init__(self):
        super(SpatialTransformer3D, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial transformer.

        Args:
            x: Input point cloud of shape (B, 3, N)

        Returns:
            Transformation matrix of shape (B, 3, 3)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode point cloud
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)

        # Predict transformation
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix as residual
        identity = torch.eye(3, dtype=x.dtype, device=device).view(1, 9)
        identity = identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(batch_size, 3, 3)

        return x


class SpatialTransformerKD(nn.Module):
    """
    Spatial Transformer Network for K-dimensional features.

    Predicts a KxK transformation matrix for feature space alignment.

    Args:
        input_dim: Dimensionality of input features
        feature_dim: Dimensionality of intermediate features (default: 1024)
    """

    def __init__(self, input_dim: int = 64, feature_dim: int = 1024):
        super(SpatialTransformerKD, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial transformer.

        Args:
            x: Input features of shape (B, K, N)

        Returns:
            Transformation matrix of shape (B, K, K)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)

        # Predict transformation
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix as residual
        identity = torch.eye(self.input_dim, dtype=x.dtype, device=device)
        identity = identity.view(1, self.input_dim * self.input_dim)
        identity = identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(batch_size, self.input_dim, self.input_dim)

        return x


class PointNetEncoder(nn.Module):
    """
    PointNet encoder for extracting features from point clouds.

    This encoder can output either global features (for the entire point cloud)
    or per-point features (combining global and local information).

    Args:
        input_dim: Dimensionality of input point features (default: 3 for XYZ)
        feature_dim: Dimensionality of output global features (default: 256)
        use_spatial_transformer: Whether to use spatial transformer network (default: False)
        return_global_feature: If True, return global feature; if False, return per-point features (default: True)

    Input:
        Point cloud of shape (B, input_dim, N) where:
            B = batch size
            input_dim = feature dimension per point (e.g., 3 for XYZ)
            N = number of points

    Output:
        If return_global_feature=True:
            Global feature of shape (B, feature_dim)
        If return_global_feature=False:
            Per-point features of shape (B, N, feature_dim + 64)
    """

    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 256,
        use_spatial_transformer: bool = False,
        return_global_feature: bool = True
    ):
        super(PointNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.use_spatial_transformer = use_spatial_transformer
        self.return_global_feature = return_global_feature

        # Spatial transformer
        if use_spatial_transformer:
            self.stn = SpatialTransformerKD(input_dim=input_dim, feature_dim=feature_dim)

        # Feature extraction layers
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_transform: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of PointNet encoder.

        Args:
            x: Input point cloud of shape (B, input_dim, N)
            return_transform: If True, also return the transformation matrix (default: False)

        Returns:
            If return_transform=False:
                features: Encoded features
            If return_transform=True:
                (features, transform): Tuple of features and transformation matrix
        """
        num_points = x.size(2)
        transform = None

        # Apply spatial transformation
        if self.use_spatial_transformer:
            transform = self.stn(x)
            x = x.transpose(2, 1)  # (B, N, K)
            x = torch.bmm(x, transform)  # (B, N, K)
            x = x.transpose(2, 1)  # (B, K, N)

        # Extract features
        x = F.relu(self.bn1(self.conv1(x)))
        local_features = x  # Save for per-point features

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Global max pooling
        global_features = torch.max(x, 2, keepdim=True)[0]
        global_features = global_features.view(-1, self.feature_dim)

        # Return based on mode
        if self.return_global_feature:
            features = global_features
        else:
            # Concatenate global and local features for per-point features
            global_features_expanded = global_features.view(-1, self.feature_dim, 1)
            global_features_expanded = global_features_expanded.repeat(1, 1, num_points)
            features = torch.cat([global_features_expanded, local_features], dim=1)
            # Transpose to (B, N, feature_dim + 64)
            features = features.transpose(1, 2)

        if return_transform:
            return features, transform
        return features

    def get_output_dim(self) -> int:
        """
        Get the output feature dimensionality.

        Returns:
            Output dimension based on the configuration
        """
        if self.return_global_feature:
            return self.feature_dim
        else:
            return self.feature_dim + 64


def create_pointnet_encoder(
    input_dim: int = 3,
    feature_dim: int = 256,
    use_spatial_transformer: bool = False,
    return_global_feature: bool = True
) -> PointNetEncoder:
    """
    Factory function to create a PointNet encoder with specified configuration.

    Args:
        input_dim: Dimensionality of input point features
        feature_dim: Dimensionality of output global features
        use_spatial_transformer: Whether to use spatial transformer network
        return_global_feature: Whether to return global or per-point features

    Returns:
        Configured PointNetEncoder instance
    """
    return PointNetEncoder(
        input_dim=input_dim,
        feature_dim=feature_dim,
        use_spatial_transformer=use_spatial_transformer,
        return_global_feature=return_global_feature
    )


if __name__ == '__main__':
    print("Testing PointNetEncoder with different configurations...\n")

    # Test data
    batch_size = 32
    num_points = 6890  # SMPL mesh vertices
    input_dim = 3

    sim_data = torch.randn(batch_size, input_dim, num_points)

    configs = [
        (True, False, "Global features without STN"),
        (True, True, "Global features with STN"),
        (False, False, "Per-point features without STN"),
        (False, True, "Per-point features with STN"),
    ]

    for return_global, use_stn, desc in configs:
        encoder = create_pointnet_encoder(
            input_dim=input_dim,
            feature_dim=256,
            use_spatial_transformer=use_stn,
            return_global_feature=return_global
        )

        output = encoder(sim_data)
        print(f"{desc}:")
        print(f"  Input shape: {sim_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dim: {encoder.get_output_dim()}")
        print()
