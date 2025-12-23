import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys

sys.path.append('/workspace/zyk/SMPL2Pressure/')

from lib.dataset import create_dataset
from lib.utils.static import TIP_PATH

def collect_pressure_data(dataset_name, device):
    """
    加载指定数据集并收集所有压力值。
    """
    # 假设所有 split (train/val/test) 的分布一致，只收集 train/val/test 中的一个或全部
    if dataset_name == "tip":
        cfgs = {
            'dataset_path': TIP_PATH,
            'dataset_mode': 'unseen_group',
            'curr_fold': 1,
            'normalize': False,
            'device': device,
        }
        data = create_dataset(dataset_name, cfgs=cfgs, mode='train')
    else: 
        data = create_dataset(dataset_name, split='train', normalize=False, device=device)
    
    all_pressures = []
    # 使用 DataLoader 批量处理数据以节省内存
    loader = torch.utils.data.DataLoader(data, batch_size=64) 
    
    for batch in tqdm(loader, desc=f"Collecting {dataset_name} pressure"):
        pressure_tensor = batch['pressure'] 
        
        # # 计算每个样本的最大压力值
        # pressures_max = pressure_tensor.max(dim=2).values.max(dim=1).values.cpu().numpy()
        # all_pressures.append(pressures_max)

        pressure = pressure_tensor.cpu().numpy()
        all_pressures.append(pressure)

    # 将所有批次的压力数据合并成一个大的 NumPy 数组
    return np.concatenate(all_pressures)

def plot_persuasive_distribution(datasets_info):
    """
    绘制极具说服力的压力分布图，包含百分比覆盖率和极值标注。
    """
    # 统一 X 轴上限，设为所有数据中 P99.99 的最大值，避免被极个别离群点拉得太长
    global_max = max(np.max(data) for data, _, _ in datasets_info.values())
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for ax, (name, (data, threshold, color)) in zip(axes, datasets_info.items()):
        # 1. 计算统计信息
        max_val = np.max(data)
        coverage = np.mean(data <= threshold) * 100
        p99 = np.percentile(data, 99)
        
        # 2. 绘制直方图 (使用 log=True 可以更好地观察长尾)
        # 如果数据非常集中在0附近，建议开启 log=True
        n, bins, patches = ax.hist(data, bins=100, density=True, color=color, alpha=0.6, label='Data Distribution')
        
        # 3. 绘制截断阈值线
        line = ax.axvline(threshold, color='red', linestyle='--', linewidth=2.5)
        
        # 4. 在图上添加“说服性”标注
        # 标注百分比覆盖率
        ax.text(threshold * 1.05, ax.get_ylim()[1] * 0.8, 
                f'Threshold: {threshold}\nCoverage: {coverage:.2f}%', 
                color='red', fontweight='bold', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
        
        # 标注最大值 (Max)
        ax.annotate(f'Max: {max_val:.1f}', xy=(max_val, 0), xytext=(max_val*0.8, ax.get_ylim()[1]*0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    horizontalalignment='right')

        # 5. 修饰图表
        ax.set_title(f'Dataset: {name.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pressure Value')
        ax.set_ylabel('Density')
        
        # 动态调整 X 轴范围：为了看清主体，范围定在 [0, max_val]
        # 如果想对比不同数据集的缩放，可以使用统一的 global_max
        ax.set_xlim(0, max_val * 1.1) 
        
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    plt.suptitle('Pressure Distribution & Selection Basis of Truncation Thresholds', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# --- 主执行逻辑 (伪代码) ---
if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 1. 数据收集
    tip_data = collect_pressure_data('tip', device)
    pp_data = collect_pressure_data('pressurepose', device)
    moyo_data = collect_pressure_data('moyo', device)
    
    # 2. 绘图信息准备
    datasets_info = {
        'tip': (tip_data, 512, 'blue'),
        'pressurepose': (pp_data, 100, 'green'),
        'moyo': (moyo_data, 100, 'orange'),
    }
    
    # 3. 绘制并显示图表
    plot_persuasive_distribution(datasets_info)

    # 4. 打印统计信息（可选但推荐）
    print("\n--- Summary Statistics ---")
    for name, (data, threshold, _) in datasets_info.items():
        # 找到低于截断值的数据百分比
        percentile_below_threshold = np.mean(data < threshold) * 100
        print(f"Dataset: {name}")
        print(f"Max Pressure: {np.max(data):.2f}")
        print(f"Mean Pressure: {np.mean(data):.2f}")
        print(f"P99: {np.percentile(data, 99):.2f}")
        print(f"P99.9: {np.percentile(data, 99.9):.2f}")
        print(f"Selected Threshold ({threshold}): Covers {percentile_below_threshold:.4f}% of data.")
        print("-" * 20)

