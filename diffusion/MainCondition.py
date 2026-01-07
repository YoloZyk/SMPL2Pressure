import sys
sys.path.insert(0, "/workspace/zyk/SMPL2Pressure/")

import torch
from diffusion.utils.config import get_args
from diffusion.Trainer import train
from diffusion.Sampler import sample

if __name__ == '__main__':
	args = get_args()
	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	if args.state == 'train':
		train(args, device)
	elif args.state == 'sample':
		sample(args, device)
	else:
		print('Invalid state. Choose either "train" or "sample".')
