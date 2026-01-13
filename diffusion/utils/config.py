import argparse


def get_args():
	parser = argparse.ArgumentParser()
	# global
	parser.add_argument('--state', type=str, default='sample', choices=['train', 'sample'], help='train or sample')
	parser.add_argument('--output_dir', type=str, default="./diffusion/output/", help="output directory")
	parser.add_argument('--T', type=int, default=500, help='denoising iterations')
	parser.add_argument('--beta_1', type=float, default=1e-4)
	parser.add_argument('--beta_T', type=float, default=0.028)
	
	# dataset
	parser.add_argument('--dataset', type=str, default='moyo', choices=['moyo', 'pressurepose', 'tip'], help='dataset')
	parser.add_argument('--normal', default=False, help='normalize or not')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	
	# model
	parser.add_argument('--cond_dim', type=int, default=256, help='condition dimension')
	parser.add_argument('--dp_rate', type=float, default=0.15, help='dropout rate')
	parser.add_argument('--channel', type=int, default=128, help='out channel for head layer')
	parser.add_argument('--channel_mult', type=list, default=[1, 2, 2, 2], help='res block channels')
	parser.add_argument('--num_res_blocks', type=int, default=2, help='res block number')
	
	# train
	parser.add_argument('--debug', action='store_true', default=False, help='debug or train')
	parser.add_argument('--note', type=str, default='', help='naming file note')
	parser.add_argument('--epochs', type=int, default=200, help='epoch')
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--print_freq', type=int, default=200, help="print frequency(steps)")
	parser.add_argument('--save_freq', type=int, default=20, help="save checkpoint frequency(epochs)")
	parser.add_argument('--multiplier', type=float, default=2.5, help='warmup multiplier')
	parser.add_argument('--grad_clip', type=float, default=1., help='grad')
	parser.add_argument('--training_load_weight', type=str, default=None, help='continue training')
	# parser.add_argument('--cond_load_weight', type=str, default="/workspace/zyk/SMPL2Pressure/output/checkpoint/cp_250210-141954_cond_best.pth", help='pretrained condition model')
	parser.add_argument('--cond_load_weight', type=str, default="/workspace/zyk/SMPL2Pressure/diffusion/output/pretrain/cp_250217-103117_cond_best.pth", help='pretrained condition model')
	
	# test
	parser.add_argument('--ckpt', type=str, default='ckpt/ckpt_260110-152300_p__epoch_200.pth', help='ckpt for test')
	parser.add_argument('--w', type=float, default=1.8, help='Condition control intensity')
	parser.add_argument('--eta', type=float, default=0.5, help='Randomness  1.0-DDPM')
	parser.add_argument('--num_steps', type=int, default=50, help='steps for DDIM')
	parser.add_argument('--viz', action='store_true', default=False, help='viz')
	
	return parser.parse_args()


