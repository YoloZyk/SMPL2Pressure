"""
Test script for visualizing dataset samples using the visualization tools.

This script demonstrates how to visualize mesh data from the three datasets:
1. MoYo
2. PressurePose
3. TIP (InBed Pressure)
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lib.dataset import create_dataset
from lib.utils.pyrender_visualization import PyRenderVisualizer
from lib.utils.static import TIP_PATH


def test_moyo_visualization():
    """Test visualization with MoYo dataset."""
    print("Testing MoYo dataset visualization...")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = create_dataset('moyo', split='train', normalize=False, device=device)
    print(f"Loaded MoYo dataset with {len(dataset)} samples")
    
    # import pdb; pdb.set_trace()

    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        print("Sample keys:", list(sample.keys()))
        
        # Create visualizer
        visualizer = PyRenderVisualizer(device=device)
        
        # Visualize the sample
        print("Visualizing MoYo sample...")
        success = visualizer.visualize_dataset_sample(sample, show_floor=True)
        if success:
            print("✓ MoYo visualization successful")
        else:
            print("✗ MoYo visualization failed")
    else:
        print("⚠ MoYo dataset is empty")


def test_pressurepose_visualization():
    """Test visualization with PressurePose dataset."""
    print("Testing PressurePose dataset visualization...")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = create_dataset('pressurepose', split='train', normalize=False, device=device)
    print(f"Loaded PressurePose dataset with {len(dataset)} samples")
    
    # import pdb; pdb.set_trace()

    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        print("Sample keys:", list(sample.keys()))
        
        # Create visualizer
        visualizer = PyRenderVisualizer(device=device)
        
        # Visualize the sample
        print("Visualizing PressurePose sample...")
        success = visualizer.visualize_dataset_sample(sample, show_floor=True)
        if success:
            print("✓ PressurePose visualization successful")
        else:
            print("✗ PressurePose visualization failed")
    else:
        print("⚠ PressurePose dataset is empty")


def test_tip_visualization():
    """Test visualization with TIP dataset."""
    print("Testing TIP dataset visualization...")
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # Configuration for TIP dataset
    cfgs = {
        'dataset_path': TIP_PATH,
        'dataset_mode': 'unseen_group',
        'curr_fold': 1,
        'normalize': False,
        'device': device,
    }
    
    # Create dataset
    dataset = create_dataset('tip', cfgs=cfgs, mode='train')
    print(f"Loaded TIP dataset with {len(dataset)} samples")
    
    # import pdb; pdb.set_trace()

    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        print("Sample keys:", list(sample.keys()))
        
        # Create visualizer
        visualizer = PyRenderVisualizer(device=device)
        
        # Visualize the sample
        print("Visualizing TIP sample...")
        success = visualizer.visualize_dataset_sample(sample, show_floor=True)
        if success:
            print("✓ TIP visualization successful")
        else:
            print("✗ TIP visualization failed")
    else:
        print("⚠ TIP dataset is empty")


def main():
    """Main function to test visualization on all datasets."""
    print("Starting dataset visualization tests...\n")
    
    # Test each dataset
    test_moyo_visualization()
    print()
    
    test_pressurepose_visualization()
    print()
    
    test_tip_visualization()
    print()
    
    print("All visualization tests completed.")


if __name__ == "__main__":
    main()