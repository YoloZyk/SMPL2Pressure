"""
Base Dataset Module for SMPL-based Pressure Datasets

This module provides an abstract base class for all pressure-SMPL datasets,
ensuring a consistent interface across different dataset implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset


class BasePressureDataset(Dataset, ABC):
    """
    Abstract base class for pressure-SMPL datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods to ensure a consistent interface.

    Standard Return Format:
        All datasets should return a dictionary with the following keys:
        - 'vertices': SMPL mesh vertices (N x 6890 x 3)
        - 'pressure': Pressure map data (varying dimensions per dataset)
        - 'smpl': Concatenated SMPL parameters (global_orient + body_pose + betas + transl)
        - Additional dataset-specific keys are allowed

    Args:
        split: Dataset split - 'train', 'val', or 'test'
        normalize: Whether to normalize pressure values
        device: Device to load tensors onto
    """

    def __init__(
        self,
        split: str,
        normalize: bool = False,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize the base dataset."""
        self.split = split
        self.normalize = normalize
        self.device = device

        # Validate split
        self._validate_split(split)

    @staticmethod
    def _validate_split(split: str) -> None:
        """
        Validate that the split parameter is valid.

        Args:
            split: Dataset split name

        Raises:
            ValueError: If split is not 'train', 'val', or 'test'
        """
        valid_splits = {'train', 'val', 'test'}
        if split not in valid_splits:
            raise ValueError(f"Split must be one of {valid_splits}, got '{split}'")

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Number of samples
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing at minimum:
                - 'vertices': SMPL mesh vertices
                - 'pressure': Pressure map data
                - 'smpl': SMPL parameters (global_orient + body_pose + betas + transl)
        """
        pass

    def get_all_betas(self) -> torch.Tensor:
        """
        Return all beta (shape) parameters.

        Returns:
            Tensor of shape parameters (N x 10)

        Note:
            Default implementation returns None. Override in subclass if supported.
        """
        return None

    def get_all_poses(self) -> torch.Tensor:
        """
        Return all body pose parameters.

        Returns:
            Tensor of pose parameters (N x 69)

        Note:
            Default implementation returns None. Override in subclass if supported.
        """
        return None

    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get metadata about the dataset.

        Returns:
            Dictionary containing dataset metadata such as:
                - 'name': Dataset name
                - 'split': Current split
                - 'num_samples': Number of samples
                - 'pressure_shape': Shape of pressure maps
                - 'normalize': Whether pressure is normalized
        """
        return {
            'name': self.__class__.__name__,
            'split': self.split,
            'num_samples': len(self),
            'normalize': self.normalize,
            'device': str(self.device)
        }

    def _validate_return_dict(self, data: Dict[str, torch.Tensor]) -> None:
        """
        Validate that the returned dictionary contains required keys.

        Args:
            data: Dictionary to validate

        Raises:
            KeyError: If required keys are missing
        """
        required_keys = {'vertices', 'pressure', 'smpl'}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise KeyError(
                f"Return dictionary missing required keys: {missing_keys}. "
                f"Got keys: {set(data.keys())}"
            )
