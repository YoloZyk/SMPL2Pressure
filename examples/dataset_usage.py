"""
Dataset Usage Examples

This file demonstrates how to use the refactored dataset modules with the unified interface.
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

# Import using the unified interface
from lib.dataset import create_dataset
from lib.utils.static import TIP_PATH


def main():
    """Main function to check dataset."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # # moyo check
    # train_data = create_dataset('moyo', split='train', normalize=False, device=device)
    # val_data = create_dataset('moyo', split='val', normalize=False, device=device)
    # test_data = create_dataset('moyo', split='test', normalize=False, device=device)

    # # pressurepose check
    # train_data = create_dataset('pressurepose', split='train', normalize=False, device=device)
    # val_data = create_dataset('pressurepose', split='val', normalize=False, device=device)
    # test_data = create_dataset('pressurepose', split='test', normalize=False, device=device)

    # tip check
    cfgs = {
        'dataset_path': TIP_PATH,
        'dataset_mode': 'unseen_group',
        'curr_fold': 1,
        'normalize': False,
        'device': device,
        }

    train_data = create_dataset('tip', cfgs=cfgs, mode='train')
    val_data = create_dataset('tip', cfgs=cfgs, mode='val')
    test_data = create_dataset('tip', cfgs=cfgs, mode='test')

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    print(f"train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)}")

    for i, batch in enumerate(tqdm(train_loader, desc="train")):
        for key in batch.keys():
            print(f"{key}: {batch[key].shape}   {batch[key].device}")
        break

    for i, batch in enumerate(tqdm(val_loader, desc="val")):
        for key in batch.keys():
            print(f"{key}: {batch[key].shape}   {batch[key].device}")
        print(batch['smpl'][0])
        break

    for i, batch in enumerate(tqdm(test_loader, desc="test")):
        for key in batch.keys():
            print(f"{key}: {batch[key].shape}   {batch[key].device}")
        print(batch['smpl'][0])
        break


    # pp_dataset = create_dataset('pressurepose', split='train', normalize=False, device=device)


if __name__ == '__main__':
    main()
