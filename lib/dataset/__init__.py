"""
Dataset Module for SMPL2Pressure

This module provides unified access to all pressure-SMPL datasets through a common interface.
"""

from typing import Dict, Any, Union
import torch

from lib.dataset.base_dataset import BasePressureDataset
from lib.dataset.moyo import MoYoDataset
from lib.dataset.pressurepose import PressurePoseDataset
from lib.dataset.tip import InBedPressureDataset


__all__ = [
    'BasePressureDataset',
    'MoYoDataset',
    'PressurePoseDataset',
    'InBedPressureDataset',
    'create_dataset',
    'get_available_datasets'
]


# Dataset registry
DATASET_REGISTRY = {
    'moyo': MoYoDataset,
    'pressurepose': PressurePoseDataset,
    'tip': InBedPressureDataset,
}


def create_dataset(
    dataset_name: str,
    **kwargs
) -> BasePressureDataset:
    """
    Factory function to create dataset instances.

    This function provides a unified interface for creating different dataset types
    with appropriate parameters.

    Args:
        dataset_name: Name of the dataset ('moyo', 'pressurepose', 'tip'/'inbed')
        **kwargs: Dataset-specific arguments

    Common kwargs for 'moyo' and 'pressurepose':
        - split (str): Dataset split - 'train', 'val', or 'test'
        - normalize (bool): Whether to normalize pressure values
        - device (torch.device): Device to load tensors onto

    Special kwargs for 'tip'/'inbed':
        - cfgs (dict): Configuration dictionary containing:
            - dataset_path (str): Path to dataset directory
            - dataset_mode (str): 'unseen_group' or 'unseen_subject'
            - curr_fold (int): Current fold for cross-validation (1, 2, or 3)
            - normalize (bool): Whether to normalize pressure data
            - device (str): Device to load tensors to ('cpu', 'cuda')
        - mode (str): Dataset mode - 'train', 'eval', or 'test'
        - path2 (str, optional): Path to augmented data
        - ratio (float): Ratio of augmented data to mix in
        - train_ratio (float): Train/eval split ratio

    Returns:
        Instance of the requested dataset

    Raises:
        ValueError: If dataset_name is not recognized

    Examples:
        >>> # Create MoYo dataset
        >>> device = torch.device('cuda')
        >>> moyo_data = create_dataset(
        ...     'moyo',
        ...     split='train',
        ...     normalize=True,
        ...     device=device
        ... )

        >>> # Create PressurePose dataset
        >>> pp_data = create_dataset(
        ...     'pressurepose',
        ...     split='train',
        ...     normalize=False,
        ...     device=device
        ... )

        >>> # Create InBed Pressure dataset
        >>> cfgs = {
        ...     'dataset_path': "/path/to/dataset",
        ...     'dataset_mode': 'unseen_group',
        ...     'curr_fold': 1,
        ...     'normalize': True,
        ...     'device': 'cuda'
        ... }
        >>> tip_data = create_dataset(
        ...     'tip',
        ...     cfgs=cfgs,
        ...     mode='train'
        ... )
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class(**kwargs)


def get_available_datasets() -> Dict[str, str]:
    """
    Get information about available datasets.

    Returns:
        Dictionary mapping dataset names to their class names

    Example:
        >>> datasets = get_available_datasets()
        >>> print(datasets)
        {'moyo': 'MoYoDataset', 'pressurepose': 'PressurePoseDataset', ...}
    """
    return {
        name: cls.__name__
        for name, cls in DATASET_REGISTRY.items()
    }


def get_dataset_info(dataset_name: str) -> str:
    """
    Get documentation for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset class documentation string

    Raises:
        ValueError: If dataset_name is not recognized

    Example:
        >>> info = get_dataset_info('moyo')
        >>> print(info)
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_class = DATASET_REGISTRY[dataset_name]
    return dataset_class.__doc__ or "No documentation available."
