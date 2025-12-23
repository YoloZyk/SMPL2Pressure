"""
In-Bed Pressure Dataset for SMPL-based Human Pose Estimation.

This module provides a PyTorch Dataset for loading and processing in-bed pressure
data along with corresponding SMPL parameters.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lib.dataset.base_dataset import BasePressureDataset


# Constants
PRESSURE_MAX_VALUE = 512.0
PRESSURE_NORMALIZE_SCALE = 512.0
PRESSURE_NORMALIZE_OFFSET = 0.5
PRESSURE_NORMALIZE_MULTIPLIER = 2.0

METER_TO_PIXEL_X = 0.0195  # Conversion factor for x-axis
METER_TO_PIXEL_Y = 0.0311  # Conversion factor for y-axis

DEFAULT_TRAIN_RATIO = 0.9

# Dataset index mapping
DATASET_IDX_MAPPING = [
    0, 1, 2, 3, 4, 5,
    1, 2, 3, 4,
    1, 2, 3, 4, 5,
    1, 2, 3, 4, 5, 6,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4
]

# Subject name to dataset index range mapping
NAME_GROUP_MAP = {
    'wq': [1, 6],
    'lgr': [6, 10],
    'wyc': [10, 15],
    'zyk': [15, 21],
    'nmt': [21, 25],
    'wyx': [25, 29],
    'lz': [29, 33],
    'twj': [33, 37],
    'xft': [37, 41],
}

# Three-fold cross-validation split
THREE_FOLD = [
    ['wyc', 'lgr', 'wyx'],
    ['wq', 'nmt', 'twj'],
    ['xft', 'zyk', 'lz'],
]


class InBedPressureDataset(BasePressureDataset):
    """
    PyTorch Dataset for in-bed pressure data with SMPL parameters.

    This dataset supports two modes:
    1. 'unseen_group': Leave-one-group-out evaluation
    2. 'unseen_subject': Leave-one-subject-out evaluation with three-fold cross-validation

    Args:
        cfgs: Configuration dictionary containing:
            - dataset_path (str): Path to dataset directory
            - dataset_mode (str): 'unseen_group' or 'unseen_subject'
            - curr_fold (int): Current fold for cross-validation (1, 2, or 3)
            - normalize (bool): Whether to normalize pressure data
            - device (str, optional): Device to load tensors to ('cpu', 'cuda'). Default: 'cpu'
        mode (str): Dataset mode - 'train', 'eval', or 'test'
        path2 (str, optional): Path to augmented data (if using data augmentation)
        ratio (float): Ratio of augmented data to mix in (0.0 to 1.0)
        train_ratio (float): Train/eval split ratio for 'unseen_subject' mode

    Example:
        >>> cfgs = {
        ...     'dataset_path': "/workspace/zyk/public_data/wzy_opt_dataset_w_feats",
        ...     'dataset_mode': 'unseen_group',
        ...     'curr_fold': 1,
        ...     'normalize': False,
        ...     'device': 'cuda'
        ... }
        >>> train_data = InBedPressureDataset(cfgs, mode='train')
        >>> train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    """

    def __init__(
        self,
        cfgs: Dict[str, Any],
        mode: str = 'train',
        path2: Optional[str] = None,
        ratio: float = 0.0,
        train_ratio: float = DEFAULT_TRAIN_RATIO
    ):
        """Initialize the InBedPressureDataset."""
        # Map 'eval' to 'val' for base class compatibility
        split = 'val' if mode == 'eval' else mode

        # Initialize base class
        super().__init__(
            split=split,
            normalize=cfgs.get('normalize', False),
            device=torch.device(cfgs.get('device', 'cpu'))
        )

        # Configuration
        self.cfgs = cfgs
        self.dataset_path = cfgs['dataset_path']
        self.dataset_mode = cfgs['dataset_mode']
        self.curr_fold = cfgs['curr_fold']
        self.mode = mode  # Keep original mode for internal logic

        # Data storage
        self.segments: List[np.ndarray] = []
        self.db_segments: List[List[int]] = []
        self.data_len = 0
        self.data: Dict[str, np.ndarray] = {}

        # Metadata storage
        self.info: Dict[str, List] = {
            'date': [],
            'name': [],
            'idx': [],
            'corner': [],
            'sensor_position': []
        }

        # Load dataset based on mode
        self._load_dataset()

        # Post-process loaded data
        self._postprocess_data(path2, ratio, train_ratio)

    def _load_dataset(self) -> None:
        """Load dataset based on dataset_mode and mode."""
        if self.dataset_mode == 'unseen_group':
            self._load_unseen_group_data()
        elif self.dataset_mode == 'unseen_subject':
            self._load_unseen_subject_data()
        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")

    def _load_unseen_group_data(self) -> None:
        """Load data for unseen_group mode."""
        if self.mode == 'train':
            self._load_unseen_group_train()
        elif self.mode == 'val':
            self._load_unseen_group_eval()
        elif self.mode == 'test':
            self._load_unseen_group_test()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _load_unseen_group_train(self) -> None:
        """Load training data for unseen_group mode."""
        for name in NAME_GROUP_MAP:
            if name == 'wyc':
                # Special case for 'wyc'
                for idx in [10, 11, 14]:
                    print(f'Loading train dataset: {idx}')
                    self._load_single_db(idx)
            else:
                # Use all but last 2 indices for training
                start_idx, end_idx = NAME_GROUP_MAP[name]
                for idx in range(start_idx, end_idx - 2):
                    print(f'Loading train dataset: {idx}')
                    self._load_single_db(idx)

    def _load_unseen_group_eval(self) -> None:
        """Load evaluation data for unseen_group mode."""
        for name in NAME_GROUP_MAP:
            if name == 'wyc':
                idx = NAME_GROUP_MAP[name][1] - 3
            else:
                idx = NAME_GROUP_MAP[name][1] - 2
            print(f'Loading eval dataset: {idx}')
            self._load_single_db(idx)

    def _load_unseen_group_test(self) -> None:
        """Load test data for unseen_group mode."""
        for name in NAME_GROUP_MAP:
            if name == 'wyc':
                idx = NAME_GROUP_MAP[name][1] - 2
            else:
                idx = NAME_GROUP_MAP[name][1] - 1
            print(f'Loading test dataset: {idx}')
            self._load_single_db(idx)

    def _load_unseen_subject_data(self) -> None:
        """Load data for unseen_subject mode with three-fold cross-validation."""
        if self.mode in ['train', 'val']:
            self._load_unseen_subject_train_eval()
        elif self.mode == 'test':
            self._load_unseen_subject_test()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _load_unseen_subject_train_eval(self) -> None:
        """Load training/evaluation data for unseen_subject mode."""
        for fold, name_list in enumerate(THREE_FOLD):
            # Load all folds except the current one
            if fold != self.curr_fold - 1:
                for name in THREE_FOLD[fold]:
                    start_idx, end_idx = NAME_GROUP_MAP[name]
                    for idx in range(start_idx, end_idx):
                        print(f'Loading train dataset: {idx}')
                        self._load_single_db(idx)

    def _load_unseen_subject_test(self) -> None:
        """Load test data for unseen_subject mode."""
        for name in THREE_FOLD[self.curr_fold - 1]:
            start_idx, end_idx = NAME_GROUP_MAP[name]
            for idx in range(start_idx, end_idx):
                print(f'Loading test dataset: {idx}')
                self._load_single_db(idx)

    def _load_single_db(self, idx: int) -> None:
        """
        Load a single database file.

        Args:
            idx: Dataset index to load
        """
        db_path = os.path.join(self.dataset_path, f'data_{idx}.npz')
        db = dict(np.load(db_path, allow_pickle=True))

        sensor_position = db['infer_sensor_position']
        segments = db['segments']

        # Extract relevant data
        data = {
            'pressure': db['pressure'],
            'keypoints_pi': db['keypoints_meter_smooth'] / np.array([METER_TO_PIXEL_X, METER_TO_PIXEL_Y]),
            'betas': db['label_betas'],
            'pose': db['label_pose'],
            'trans': db['label_trans'],
            'verts': db['label_verts'],
        }

        # Process each segment
        for segment in segments:
            # Store metadata
            self.info['name'].append(db['name'])
            self.info['date'].append(db['date'])
            self.info['sensor_position'].append(db['infer_sensor_position'])
            self.info['corner'].extend(db['bed_corner_shift'][segment[0]: segment[1]])
            self.info['idx'].append(idx)

            # Store data
            segment_start, segment_end = segment[0], segment[1]
            segment_length = segment_end - segment_start

            if not len(self.data):
                # Initialize data storage
                for key in data.keys():
                    self.data[key] = data[key][segment_start:segment_end]
                self.segments.append(np.array(segment) - segment_start + self.data_len)
                self.db_segments.append(segment)
                self.data_len += segment_length
            else:
                # Concatenate with existing data
                for key in data.keys():
                    self.data[key] = np.concatenate(
                        [self.data[key], data[key][segment_start:segment_end]],
                        axis=0
                    )
                self.segments.append(np.array(segment) - segment_start + self.data_len)
                self.db_segments.append(segment)
                self.data_len += segment_length

    def _postprocess_data(
        self,
        path2: Optional[str],
        ratio: float,
        train_ratio: float
    ) -> None:
        """
        Post-process loaded data: normalize, augment, and split.

        Args:
            path2: Path to augmented data
            ratio: Ratio of augmented data to mix in
            train_ratio: Train/eval split ratio
        """
        pressure = self.data['pressure'].astype(np.float32)
        joints = self.data['keypoints_pi']
        shape = self.data['betas']
        verts = self.data['verts']

        # Handle train/eval split for unseen_subject mode
        if self.dataset_mode == 'unseen_subject' and self.mode in ['train', 'val']:
            pressure, joints, shape = self._split_train_eval(
                pressure, joints, shape, train_ratio
            )

        # Mix in augmented data if provided
        if path2 and ratio > 0:
            pressure, joints = self._mix_augmented_data(
                pressure, joints, path2, ratio
            )

        # Clip and normalize pressure values
        pressure = self._normalize_pressure(pressure)

        # Store processed data
        self.data['pressure'] = pressure.copy()
        self.data['keypoints_pi'] = joints.copy()
        self.data['betas'] = shape.copy()

        # print(f'Final data shape - Pressure: {pressure.shape}, Verts: {verts.shape}')

    def _split_train_eval(
        self,
        pressure: np.ndarray,
        joints: np.ndarray,
        shape: np.ndarray,
        train_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and eval sets.

        Args:
            pressure: Pressure data
            joints: Joint keypoints
            shape: SMPL shape parameters
            train_ratio: Ratio for train split

        Returns:
            Tuple of (pressure, joints, shape) after splitting
        """
        # Shuffle data
        shuffle_ix = np.random.permutation(np.arange(pressure.shape[0]))
        pressure = pressure[shuffle_ix]
        joints = joints[shuffle_ix]
        shape = shape[shuffle_ix]

        # Split based on mode
        train_len = int(pressure.shape[0] * train_ratio)
        if self.mode == 'train':
            pressure = pressure[:train_len]
            joints = joints[:train_len]
            shape = shape[:train_len]
        else:  # eval mode
            pressure = pressure[train_len:]
            joints = joints[train_len:]
            shape = shape[train_len:]

        print(f'Split data shape: {pressure.shape}')
        return pressure, joints, shape

    def _mix_augmented_data(
        self,
        pressure: np.ndarray,
        joints: np.ndarray,
        path2: str,
        ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mix in augmented data from external source.

        Args:
            pressure: Original pressure data
            joints: Original joint keypoints
            path2: Path to augmented data file
            ratio: Ratio of augmented data to mix in

        Returns:
            Tuple of (pressure, joints) with augmented data mixed in
        """
        augmented_file = os.path.join(path2, 'pure_data_fromdownstream_train_labels_augmented.h5')
        with h5py.File(augmented_file, 'r') as f:
            data_augmented = f['data_pure']
            print(f'Augmented data shape: {data_augmented.shape}')

            fake_len = int(pressure.shape[0] * ratio)
            data_augmented = data_augmented[:fake_len].reshape(-1, 56, 40)
            pos_info_augmented = f['pos'][:fake_len].astype(np.float32)
            pos_info_augmented = pos_info_augmented.reshape(-1, 15, 2)

        # Concatenate with original data
        pressure = np.vstack((pressure, data_augmented))
        joints = np.vstack((joints, pos_info_augmented))

        return pressure, joints

    def _normalize_pressure(self, pressure: np.ndarray) -> np.ndarray:
        """
        Normalize pressure values.

        Args:
            pressure: Raw pressure data

        Returns:
            Normalized pressure data
        """
        # Clip maximum values
        pressure = np.clip(pressure, None, PRESSURE_MAX_VALUE)

        if self.normalize:
            # Normalize to [-1, 1] range
            pressure = pressure / PRESSURE_NORMALIZE_SCALE
            # pressure = (pressure - PRESSURE_NORMALIZE_OFFSET) * PRESSURE_NORMALIZE_MULTIPLIER

        return pressure

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            index: Sample index

        Returns:
            Dictionary containing:
                - 'pressure': Pressure map tensor [H, W]
                - 'vertices': SMPL vertices tensor
                - 'smpl': Concatenated SMPL parameters (pose + betas + trans)
        """
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])

        # Concatenate SMPL parameters
        smpl_params = torch.cat([
            torch.tensor(self.data['pose'][index], dtype=torch.float32),
            torch.tensor(self.data['betas'][index], dtype=torch.float32),
            torch.tensor(self.data['trans'][index], dtype=torch.float32),
        ], dim=0)

        # import pdb; pdb.set_trace()
        vertices = torch.tensor(self.data['verts'][index], dtype=torch.float32)

        # 坐标变换
        vertices[:, 1] = 1.80 - vertices[:, 1]
        vertices[:, 2] = -vertices[:, 2]
        # vertices[:, 2] -= 0.10

        result = {
            'pressure': transform(self.data['pressure'][index]).type(torch.FloatTensor).squeeze().to(self.device),
            'vertices': vertices.to(self.device),
            'smpl': smpl_params.to(self.device)
        }

        # Validate return format
        self._validate_return_dict(result)

        return result

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.data['pressure'].shape[0]

    def get_segments(self) -> List[np.ndarray]:
        """
        Get segment boundaries.

        Returns:
            List of segment arrays
        """
        return self.segments

    def get_data_len(self) -> int:
        """
        Get total data length.

        Returns:
            Total number of frames across all segments
        """
        return self.data_len

    def load_fake_label_2d(self, path: Optional[str] = None) -> None:
        """
        Load fake 2D keypoint labels for testing.

        Args:
            path: Optional custom path to 2D keypoint predictions
        """
        if path is None:
            path = '/workspace/wzy1999/projects/PIDHMR/results/KP/pimesh_None_None_None_20240909_2147/test.npz'

        kp_db = np.load(path)
        self.test_kp_db = kp_db['kp_2d']
        print(f'Loaded 2D keypoints from: {path}')

    def load_fake_label_3d(self, path: Optional[str] = None) -> None:
        """
        Load fake 3D keypoint labels for testing.

        Args:
            path: Optional custom path to 3D keypoint predictions
        """
        if path is None:
            if self.dataset_mode == 'unseen_subject':
                paths = [
                    '/workspace/wzy1999/projects/PIDHMR/results/KP/unsubject/1/3D/pimesh_None_None_None_20240919_2121/test.npz',
                    '/workspace/wzy1999/projects/PIDHMR/results/KP/unsubject/2/3D/pimesh_None_None_None_20240919_2123/test.npz',
                    '/workspace/wzy1999/projects/PIDHMR/results/KP/unsubject/3/3D/pimesh_None_None_None_20240919_2123/test.npz'
                ]
                path = paths[self.curr_fold - 1]
            else:
                path = '/workspace/wzy1999/projects/PIDHMR/results/KP/3D/pimesh_None_None_None_20240911_1043/test.npz'

        kp_db = np.load(path)
        self.test_3dkp_db = kp_db['kp_3d']
        print(f'Loaded 3D keypoints from: {path}')


def main():
    """Example usage of InBedPressureDataset."""
    # Configuration for unseen_group mode
    cfgs = {
        'dataset_path': "/workspace/zyk/public_data/wzy_opt_dataset_w_feats",
        'dataset_mode': 'unseen_group',  # or 'unseen_subject'
        'curr_fold': 1,  # Used for 'unseen_subject' mode (1, 2, or 3)
        'normalize': False,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),  # or 'cuda' for GPU
    }

    # Create dataset and dataloader
    train_data = InBedPressureDataset(cfgs, mode='train')
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        # pin_memory=True
    )

    # Iterate through batches
    for i, data in enumerate(tqdm(train_loader, desc='Checking dataset')):
        print(f"Batch {i}:")
        print(f"  Pressure shape: {data['pressure'].shape}")
        print(f"  Vertices shape: {data['vertices'].shape}")
        print(f"  SMPL params shape: {data['smpl'].shape}")

        # Example: Break after first batch for quick check
        if i == 0:
            break


if __name__ == '__main__':
    main()
