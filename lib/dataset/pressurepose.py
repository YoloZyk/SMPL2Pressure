"""
PressurePose Dataset Module

This module provides a PyTorch Dataset implementation for the PressurePose dataset,
which contains pressure map data paired with SMPL body model parameters.
"""

from enum import IntEnum
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pickle
import torch
import smplx
from torch.utils.data import Dataset, DataLoader

from lib.utils.static import PressurePose_PATH, SMPL_MODEL, DATASET_META
from lib.dataset.base_dataset import BasePressureDataset


# Constants
class Gender(IntEnum):
    """Gender enumeration for SMPL models."""
    FEMALE = 0
    MALE = 1


class DataConstants:
    """Constants used throughout the dataset."""
    PRESSURE_HEIGHT = 64
    PRESSURE_WIDTH = 27
    PRESSURE_MAX_VALUE = DATASET_META['pressurepose']['max_p']
    GLOBAL_ORIENT_DIM = 3
    BODY_POSE_START_IDX = 3
    FEMALE_MARKER = '_f_'
    TRAIN_FILE_PREFIX = 'train'
    TEST_FILE_PREFIX = 'test'


class PressurePoseDataset(BasePressureDataset):
    """
    PressurePose Dataset for loading pressure maps and corresponding SMPL body parameters.

    This dataset loads pressure sensor data paired with SMPL body model parameters,
    including body shape (betas), pose parameters, and contact information.

    Args:
        split: Dataset split - 'train', 'val', or 'test'
        normalize: Whether to normalize pressure values to [0, 1] range
        device: Device to load tensors onto (default: 'cuda')

    Attributes:
        betas: SMPL shape parameters (N x 10)
        body_pose: SMPL body pose parameters (N x 69)
        global_orient: SMPL global orientation (N x 3)
        transl: SMPL translation parameters (N x 3)
        pressures: Pressure map data (N x 64 x 27)
        contacts: Contact information (N x 64 x 27)
        gender: Gender labels (N,) - 0 for female, 1 for male
    """

    def __init__(
        self,
        split: str,
        normalize: bool = False,
        device: torch.device = torch.device('cuda')
    ):
        """Initialize the PressurePose dataset."""
        # Initialize base class
        super().__init__(split=split, normalize=normalize, device=device)

        # Determine file mode based on split
        self.file_mode = self._get_file_mode(split)

        # Load all data
        self._load_dataset()

        # Initialize SMPL models
        self._initialize_smpl_models()

    @staticmethod
    def _get_file_mode(split: str) -> str:
        """Determine which file prefix to use based on split."""
        return DataConstants.TRAIN_FILE_PREFIX if split == 'train' else DataConstants.TEST_FILE_PREFIX

    def _load_dataset(self) -> None:
        """Load all data files for the specified split."""
        # Initialize data collectors as lists for efficiency
        data_collectors = {
            'betas': [],
            'body_pose': [],
            'transl': [],
            'global_orient': [],
            'pressures': [],
            'contacts': [],
            'gender': []
        }

        # Get all subject directories
        data_dirs = self._get_data_directories()

        # Load data from all files
        for data_dir in data_dirs:
            self._load_from_directory(data_dir, data_collectors)

        # Convert collected lists to tensors and move to device
        self._finalize_data_loading(data_collectors)

    def _get_data_directories(self) -> List[Path]:
        """Get all valid data directories in the dataset path."""
        dataset_path = Path(PressurePose_PATH)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {PressurePose_PATH}")

        data_dirs = [
            dataset_path / p
            for p in dataset_path.iterdir()
            if p.is_dir()
        ]

        if not data_dirs:
            raise ValueError(f"No data directories found in {PressurePose_PATH}")

        return data_dirs

    def _load_from_directory(
        self,
        data_dir: Path,
        data_collectors: Dict[str, List]
    ) -> None:
        """Load data from all matching files in a directory."""
        # import pdb; pdb.set_trace()
        for file_path in data_dir.iterdir():
            if file_path.name.startswith(self.file_mode) and file_path.suffix in {'.pkl', '.pickle', '.p'}:
                self._load_single_file(file_path, data_collectors)

    def _load_single_file(
        self,
        file_path: Path,
        data_collectors: Dict[str, List]
    ) -> None:
        """Load and process a single data file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            # Extract gender from filename
            gender = self._extract_gender_from_filename(file_path.name)

            # Get data slice based on split
            data_slice = self._get_data_slice(data, self.split)

            # Extract and append data
            self._extract_data(data, data_slice, gender, data_collectors)

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    @staticmethod
    def _extract_gender_from_filename(filename: str) -> Gender:
        """Extract gender information from filename."""
        return Gender.FEMALE if DataConstants.FEMALE_MARKER in filename else Gender.MALE

    @staticmethod
    def _get_data_slice(data: Dict, split: str) -> slice:
        """Determine the data slice based on split type."""
        num_samples = len(data['images'])
        split_idx = num_samples // 2

        if split == 'train':
            return slice(None)  # All data
        elif split == 'val':
            return slice(None, split_idx)  # First half
        else:  # test
            return slice(split_idx, None)  # Second half

    def _extract_data(
        self,
        data: Dict,
        data_slice: slice,
        gender: Gender,
        data_collectors: Dict[str, List]
    ) -> None:
        """Extract relevant data from loaded pickle file and append to collectors."""
        # Convert numpy arrays to tensors
        body_shape = torch.tensor(np.array(data['body_shape']), dtype=torch.float32)[data_slice]
        joint_angles = torch.tensor(np.array(data['joint_angles']), dtype=torch.float32)[data_slice]
        root_xyz = torch.tensor(np.array(data['root_xyz_shift']), dtype=torch.float32)[data_slice]
        images = torch.tensor(np.array(data['images']), dtype=torch.float32)[data_slice]
        contacts = torch.tensor(np.array(data['mesh_contact']), dtype=torch.float32)[data_slice]

        # Append to collectors
        data_collectors['betas'].append(body_shape)
        data_collectors['body_pose'].append(joint_angles[:, DataConstants.BODY_POSE_START_IDX:])
        data_collectors['global_orient'].append(joint_angles[:, :DataConstants.GLOBAL_ORIENT_DIM])
        data_collectors['transl'].append(root_xyz)
        data_collectors['pressures'].append(images)
        data_collectors['contacts'].append(contacts)

        # Create gender tensor
        num_samples = len(images)
        gender_tensor = torch.full((num_samples,), gender, dtype=torch.int16)
        data_collectors['gender'].append(gender_tensor)

    def _finalize_data_loading(self, data_collectors: Dict[str, List]) -> None:
        """Concatenate collected data and move to device."""
        if not data_collectors['betas']:
            raise ValueError(f"No data found for split '{self.split}'")

        # Concatenate all collected tensors
        self.betas = torch.cat(data_collectors['betas'], dim=0).to(self.device)
        self.body_pose = torch.cat(data_collectors['body_pose'], dim=0).to(self.device)
        self.global_orient = torch.cat(data_collectors['global_orient'], dim=0).to(self.device)
        self.transl = torch.cat(data_collectors['transl'], dim=0).to(self.device)
        self.pressures = torch.cat(data_collectors['pressures'], dim=0)
        self.contacts = torch.cat(data_collectors['contacts'], dim=0)
        self.gender = torch.cat(data_collectors['gender'], dim=0)

        # print(f"Loaded {len(self)} samples for {self.split} split")

    def _initialize_smpl_models(self) -> None:
        """Initialize SMPL models for both genders."""
        self.male_model = smplx.create(
            SMPL_MODEL,
            model_type='smpl',
            gender='male',
            ext='pkl'
        ).to(self.device)

        self.female_model = smplx.create(
            SMPL_MODEL,
            model_type='smpl',
            gender='female',
            ext='pkl'
        ).to(self.device)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.pressures)

    def get_all_betas(self) -> torch.Tensor:
        """Return all beta (shape) parameters as a CPU tensor."""
        return self.betas.cpu()

    def get_all_poses(self) -> torch.Tensor:
        """Return all body pose parameters as a CPU tensor."""
        return self.body_pose.cpu()

    def _process_pressure(self, pressure: torch.Tensor) -> torch.Tensor:
        """Process pressure data with reshaping, clamping, and optional normalization."""
        # Reshape to pressure map dimensions
        pressure = pressure.reshape(
            -1,
            DataConstants.PRESSURE_HEIGHT,
            DataConstants.PRESSURE_WIDTH
        )

        # Clamp to maximum value
        pressure = torch.clamp(pressure, max=DataConstants.PRESSURE_MAX_VALUE)

        # Normalize if requested
        if self.normalize:
            pressure = pressure / DataConstants.PRESSURE_MAX_VALUE

        return pressure.squeeze()

    def _get_smpl_output(self, idx: int) -> smplx.SMPL:
        """Generate SMPL mesh output for a given index."""
        # Select appropriate model based on gender
        model = self.female_model if self.gender[idx] == Gender.FEMALE else self.male_model

        # Generate SMPL output
        output = model(
            betas=self.betas[idx].unsqueeze(0),
            body_pose=self.body_pose[idx].unsqueeze(0),
            global_orient=self.global_orient[idx].unsqueeze(0),
            transl=self.transl[idx].unsqueeze(0)
        )

        return output

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - vertices: SMPL mesh vertices (6890 x 3)
                - pressure: Pressure map (64 x 27)
                - smpl: Concatenated SMPL parameters (85,)
                - gender: Gender label (0 for female, 1 for male)
        """
        # Generate SMPL mesh
        smpl_output = self._get_smpl_output(idx)
        vertices = smpl_output.vertices.squeeze().float()

        # Concatenate SMPL parameters
        smpl_params = torch.cat([
            self.global_orient[idx],
            self.body_pose[idx],
            self.betas[idx],
            self.transl[idx]
        ], dim=0)

        # Process pressure data
        pressure = self._process_pressure(self.pressures[idx])

        result = {
            'vertices': vertices,       # B x 6890 x 3
            'pressure': pressure.float().to(self.device),  # B x 64 x 27
            'smpl': smpl_params.float(),   # B x 85  global_orient + body_pose + betas + transl
            'gender': self.gender[idx]     # B
        }

        # Validate return format
        self._validate_return_dict(result)

        return result
    

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = PressurePoseDataset(split='train', device=device)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    for i, batch in enumerate(tqdm(train_loader, desc="checking")):
        import pdb; pdb.set_trace()







