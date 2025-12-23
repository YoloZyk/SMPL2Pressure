"""
MoYo Dataset Module

This module provides a PyTorch Dataset implementation for the MoYo (Motion and Yoga) dataset,
which contains pressure map data paired with SMPL body model parameters for yoga poses.
"""

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import smplx
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lib.utils.static import MoYo_PATH, SMPL_MODEL
from lib.dataset.base_dataset import BasePressureDataset


class DataConstants:
    """Constants used throughout the MoYo dataset."""
    PRESSURE_HEIGHT = 110
    PRESSURE_WIDTH = 37
    PRESSURE_MAX_VALUE = 100.0
    BETAS_DIM = 10
    BODY_POSE_DIM = 69
    GLOBAL_ORIENT_DIM = 3
    TRANSL_DIM = 3
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1


class MoYoDataset(BasePressureDataset):
    """
    MoYo Dataset for loading pressure maps and corresponding SMPL body parameters.

    This dataset loads pressure sensor data from yoga poses paired with SMPL body
    model parameters, including body shape (betas), pose parameters, and translation.

    Args:
        split: Dataset split - 'train', 'val', or 'test'
        normalize: Whether to normalize pressure values to [0, 1] range (default: False)
        device: Device to load tensors onto (default: 'cuda')

    Attributes:
        betas: SMPL shape parameters (N x 10)
        body_pose: SMPL body pose parameters (N x 69)
        global_orient: SMPL global orientation (N x 3)
        transl: SMPL translation parameters (N x 3)
        pressures: Pressure map data (N x 110 x 37)
        all_betas: All beta parameters from the full dataset
        smpl_model: SMPL model for generating meshes

    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> dataset = MoYoDataset(split='train', normalize=True, device=device)
        >>> dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        >>> for batch in dataloader:
        ...     vertices = batch['vertices']  # B x 6890 x 3
        ...     pressure = batch['pressure']  # B x 110 x 37
        ...     smpl_params = batch['smpl']   # B x 85
    """

    def __init__(
        self,
        split: str,
        normalize: bool = False,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize the MoYo dataset."""
        # Initialize base class
        super().__init__(split=split, normalize=normalize, device=device)

        # Transformation for converting numpy arrays to tensors
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load data
        self._load_dataset()

        # Initialize SMPL model
        self._initialize_smpl_model()

    def _load_dataset(self) -> None:
        """Load and split the MoYo dataset based on the specified split."""
        # Construct file paths
        smpl_file = os.path.join(MoYo_PATH, 'smpl.pkl')
        pressure_file = os.path.join(MoYo_PATH, 'pressure.npy')

        # Validate file existence
        self._validate_files(smpl_file, pressure_file)

        # Load SMPL parameters
        with open(smpl_file, 'rb') as f:
            smpls = pickle.load(f)

        # Store all betas for reference
        self.all_betas = torch.tensor(smpls['betas'], dtype=torch.float32)

        # Load pressure data
        pressures = np.load(pressure_file)

        # Get split indices
        start_idx, end_idx = self._get_split_indices(len(pressures))

        # Extract data for this split
        self.betas = torch.tensor(
            smpls['betas'][start_idx:end_idx],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.body_pose = torch.tensor(
            smpls['body_pose'][start_idx:end_idx],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.transl = torch.tensor(
            smpls['transl'][start_idx:end_idx],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.global_orient = torch.tensor(
            smpls['global_orient'][start_idx:end_idx],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.pressures = pressures[start_idx:end_idx]

        # Store base index for reference
        self.base_idx = start_idx

        # print(f"Loaded MoYo {self.split} split: {len(self)} samples")

    @staticmethod
    def _validate_files(*file_paths: str) -> None:
        """
        Validate that all required files exist.

        Args:
            *file_paths: Variable number of file paths to validate

        Raises:
            FileNotFoundError: If any file does not exist
        """
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

    def _get_split_indices(self, total_samples: int) -> tuple:
        """
        Calculate start and end indices for the current split.

        Args:
            total_samples: Total number of samples in the dataset

        Returns:
            Tuple of (start_index, end_index)

        Raises:
            ValueError: If split is invalid
        """
        train_end = int(DataConstants.TRAIN_RATIO * total_samples)
        val_end = int((DataConstants.TRAIN_RATIO + DataConstants.VAL_RATIO) * total_samples)

        if self.split == 'train':
            return 0, train_end
        elif self.split == 'val':
            return train_end, val_end
        elif self.split == 'test':
            return val_end, total_samples
        else:
            # This should never happen due to base class validation
            raise ValueError(f"Invalid split: {self.split}")

    def _initialize_smpl_model(self) -> None:
        """Initialize the SMPL model for generating meshes."""
        try:
            self.smpl_model = smplx.create(
                SMPL_MODEL,
                model_type='smpl',
                gender='male',
                ext='pkl'
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SMPL model: {e}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.pressures)

    def get_all_betas(self) -> np.ndarray:
        """
        Return all beta (shape) parameters from the full dataset.

        Returns:
            Numpy array of shape parameters (N x 10)
        """
        return self.all_betas.cpu().numpy()[:, :DataConstants.BETAS_DIM]

    def get_all_poses(self) -> np.ndarray:
        """
        Return all body pose parameters from the full dataset.

        Returns:
            Numpy array of pose parameters (N x 69)
        """
        # Extract from the loaded split data
        return self.body_pose.squeeze(0).cpu().numpy()

    def _process_pressure(self, pressure: np.ndarray) -> torch.Tensor:
        """
        Process pressure data with reshaping, flipping, clamping, and optional normalization.

        Args:
            pressure: Raw pressure data

        Returns:
            Processed pressure tensor (110 x 37)
        """
        # Reshape to pressure map dimensions
        pressure = pressure.reshape(
            DataConstants.PRESSURE_HEIGHT,
            -1,
            1
        )

        # Convert to tensor
        pressure = self.to_tensor(pressure)

        # Flip along width dimension (legacy processing)
        pressure = pressure.flip(2)

        # Clamp to maximum value
        pressure = torch.clamp(pressure, max=DataConstants.PRESSURE_MAX_VALUE)

        # Normalize if requested
        if self.normalize:
            pressure = pressure / DataConstants.PRESSURE_MAX_VALUE

        return pressure.squeeze()

    def _generate_smpl_output(self, idx: int) -> smplx.SMPL:
        """
        Generate SMPL mesh output for a given index.

        Args:
            idx: Index of the sample

        Returns:
            SMPL output containing vertices and other mesh information
        """
        output = self.smpl_model(
            betas=self.betas[:, idx, :DataConstants.BETAS_DIM],
            body_pose=self.body_pose[:, idx, :],
            global_orient=self.global_orient[:, idx, :],
            transl=self.transl[:, idx, :]
        )
        return output

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - 'vertices': SMPL mesh vertices (6890 x 3)
                - 'pressure': Pressure map (110 x 37)
                - 'smpl': Concatenated SMPL parameters (85,)
                  Format: [global_orient(3) + body_pose(69) + betas(10) + transl(3)]

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Generate SMPL mesh
        smpl_output = self._generate_smpl_output(idx)
        vertices = smpl_output.vertices.squeeze().float().to(self.device)

        # Concatenate SMPL parameters
        smpl_params = torch.cat([
            self.global_orient[:, idx, :],
            self.body_pose[:, idx, :],
            self.betas[:, idx, :DataConstants.BETAS_DIM],
            self.transl[:, idx, :]
        ], dim=1).squeeze().float().to(self.device)

        # Process pressure data
        pressure = self._process_pressure(self.pressures[idx])

        result = {
            'vertices': vertices,              # 6890 x 3
            'pressure': pressure.float().to(self.device),  # 110 x 37
            'smpl': smpl_params,               # 85
            'gender': torch.tensor([1])[0]     # B
        }

        # Validate return format
        self._validate_return_dict(result)

        return result

    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get metadata about the dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        info = super().get_dataset_info()
        info.update({
            'pressure_shape': (DataConstants.PRESSURE_HEIGHT, DataConstants.PRESSURE_WIDTH),
            'pressure_max_value': DataConstants.PRESSURE_MAX_VALUE,
            'base_index': self.base_idx,
        })
        return info


def main():
    """Example usage and testing of MoYoDataset."""
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    dataset = MoYoDataset(split='train', normalize=True, device=device)

    # Print dataset info
    print("\nDataset Info:")
    for key, value in dataset.get_dataset_info().items():
        print(f"  {key}: {value}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Test iteration
    print("\nTesting dataloader...")
    for i, batch in enumerate(tqdm(dataloader, desc='Checking')):
        print(f"\nBatch {i}:")
        print(f"  Vertices shape: {batch['vertices'].shape}")
        print(f"  Pressure shape: {batch['pressure'].shape}")
        print(f"  SMPL params shape: {batch['smpl'].shape}")

        # Break after first batch for quick check
        if i == 0:
            break


if __name__ == '__main__':
    main()
