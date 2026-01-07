import torch

import torch


def pad_to_multiple_of_8(x):
	"""
    将输入张量填充到 H 和 W 是 8 的倍数。
    输入形状: B x 1 x H x W
    输出形状: B x 1 x H_pad x W_pad
    """
	B, C, H, W = x.shape
	assert C == 1, "输入张量的通道数必须为 1"
	
	# 计算需要填充的高度和宽度
	H_pad = (H + 7) // 8 * 8  # 向上取整到最近的 8 的倍数
	W_pad = (W + 7) // 8 * 8  # 向上取整到最近的 8 的倍数
	
	# 计算填充量
	pad_H = H_pad - H
	pad_W = W_pad - W
	
	# 在高度和宽度上对称填充
	if pad_H > 0 or pad_W > 0:
		# 填充顺序: (左边, 右边, 上边, 下边)
		padding = (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2)
		x = torch.nn.functional.pad(x, padding, mode='constant', value=0)
	
	return x


# 测试用例
if __name__ == "__main__":
	# 测试 1: 110x37 -> 112x40
	x1 = torch.randn(4, 1, 110, 37)
	padded_x1 = pad_to_multiple_of_8(x1)
	print(f"原始形状: {x1.shape}, 填充后形状: {padded_x1.shape}")
	
	# 测试 2: 64x27 -> 64x32
	x2 = torch.randn(4, 1, 64, 27)
	padded_x2 = pad_to_multiple_of_8(x2)
	print(f"原始形状: {x2.shape}, 填充后形状: {padded_x2.shape}")
	
	# 测试 3: 56x40 -> 56x40 (不变)
	x3 = torch.randn(4, 1, 56, 40)
	padded_x3 = pad_to_multiple_of_8(x3)
	print(f"原始形状: {x3.shape}, 填充后形状: {padded_x3.shape}")
