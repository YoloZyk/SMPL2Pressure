import numpy as np
import random
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "/workspace/zyk/SMPL2Pressure/")

from wq.single_dataset import InBedPressureDataset
from wq.utils import *
from lib.utils.static import TIP_PATH
from lib.dataset import create_dataset
from lib.baseline import pmr_pred
from lib.utils.viz_utils import viz_pwm

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def draw_single_pic_test(eval_fake_B, ps_jt, withjt=True, show=True, max_v=300):
    BODY_14_pairs = np.array([
        [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [8, 9], [9, 13], [10, 11], [11, 12]
    ])

    # eval_fake_B = eval_fake_B.cpu()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
    if eval_fake_B.max()>10:
        axes.matshow(eval_fake_B, vmin=10, vmax=max_v)
        # axes.matshow(eval_fake_B)
    else:
        # axes.matshow(eval_fake_B, vmax=max_v)
        axes.matshow(eval_fake_B)
    # axes.set_title(f'{model_name}', fontsize=16)

    if withjt:
        # ps_jt = ps_jt.cpu()
        ps_jtx, ps_jty = ps_jt[:, 0], ps_jt[:, 1]
        # ps_jtx, ps_jty = ps_jt[:, 0] , ps_jt[:, 1]
        axes.scatter(ps_jtx, ps_jty, s=120., color='darkgreen', linewidths=2, edgecolors='gold')
        for m in range(BODY_14_pairs.shape[0]):
            # print(m)
            x = (ps_jtx[BODY_14_pairs[m][0]], ps_jtx[(BODY_14_pairs[m][1])])
            y = (ps_jty[BODY_14_pairs[m][0]], ps_jty[(BODY_14_pairs[m][1])])

            # x=(ps_jtx[0],ps_jty[0])
            # y=(ps_jtx[1],ps_jty[1])
            axes.plot(x, y, color='r')

    axes.set_xticks([])
    axes.set_yticks([])
    plt.axis('off')
    if show:
        plt.show()

    return fig

def draw_single_pic_TIP(eval_fake_B, ps_jt, withjt=True, show=True, max_v=0):
    BODY_15_pairs = np.array([
        [0, 1], [0, 2], [3, 5], [5, 7], [4, 6], [6, 8], [9, 11],
        [11, 13], [10, 12], [12, 14]
    ])

    # eval_fake_B = eval_fake_B.cpu()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
    if eval_fake_B.max()>10:
        # axes.matshow(eval_fake_B, vmin=0, vmax=200)
        axes.matshow(eval_fake_B, vmax=300)
        # axes.matshow(eval_fake_B, vmax=250)
    else:
        # axes.matshow(eval_fake_B, vmax=max_v)
        axes.matshow((eval_fake_B*0.5+0.5)*512, vmax=300)
    # axes.set_title(f'{model_name}', fontsize=16)

    if withjt:
        # ps_jt = ps_jt.cpu()
        ps_jtx, ps_jty = ps_jt[:, 0], ps_jt[:, 1]
        # ps_jtx, ps_jty = ps_jt[:, 0] , ps_jt[:, 1]
        axes.scatter(ps_jtx, ps_jty, s=120., color='darkgreen', linewidths=2, edgecolors='gold')
        for m in range(BODY_15_pairs.shape[0]):
            # print(m)
            x = (ps_jtx[BODY_15_pairs[m][0]], ps_jtx[(BODY_15_pairs[m][1])])
            y = (ps_jty[BODY_15_pairs[m][0]], ps_jty[(BODY_15_pairs[m][1])])

            # x=(ps_jtx[0],ps_jty[0])
            # y=(ps_jtx[1],ps_jty[1])
            axes.plot(x, y, color='r')

    axes.set_xticks([])
    axes.set_yticks([])
    plt.axis('off')
    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    seed = 40
    setup_seed(seed)
    save_fig_dir = "./check_data/"
    os.makedirs(save_fig_dir, exist_ok=True)

    generator = torch.Generator().manual_seed(seed)

    cfgs = {
            'dataset_path': TIP_PATH,
            'dataset_mode': 'unseen_subject',
            'curr_fold': 1,
            'normalize': True,
            'device': 'cpu',
        }
    dataset = InBedPressureDataset(cfgs, mode='test', need_shape=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator, num_workers=0, drop_last=False, pin_memory=True)

    eval_reses = []
    total_num = 0
    step = 0
    max_value = 0

    for i, (sampledImgs, vertices, labels) in enumerate(tqdm(dataloader, desc="Test TIP")):
        # for m in tqdm(range(sampledImgs.shape[0]), desc=f"Drawing {i}"):
        #     fig1 = draw_single_pic_TIP((sampledImgs[m, 0].detach().cpu()) * 512, labels[m].detach().cpu(), show=False)
        #     fig1.savefig(os.path.join(save_fig_dir, 'withjt_' + str(m) + '.png'), bbox_inches='tight', pad_inches=-0.1)

        #     fig2 = draw_single_pic_TIP((sampledImgs[m, 0].detach().cpu()) * 512, labels[m].detach().cpu(), withjt=False, show=False)
        #     fig2.savefig(os.path.join(save_fig_dir, 'nojt_' + str(m) + '.png'), bbox_inches='tight', pad_inches=-0.1)
        # break

        total_num += sampledImgs.shape[0]
        # import pdb; pdb.set_trace()
        sampledImgs = sampledImgs.squeeze(1)
        vertices[:, :, 1] = 1.80 - vertices[:, :, 1]
        vertices[:, :, 2] = -vertices[:, :, 2]
        # vertices[:, :, 2] -= 0.10

        pred = pmr_pred(vertices, 'tip')

        # viz_pwm(vertices, sampledImgs.squeeze(1), pred, None)

        if i < 2:
            for m in range(sampledImgs.shape[0]):
                fig1 = draw_single_pic_TIP((pred[m].detach().cpu()*512/40), labels[m].detach().cpu(), show=False)
                fig1.savefig(os.path.join(save_fig_dir, f'step_{step}' + '_withjt_' + str(m) + '.png'), bbox_inches='tight', pad_inches=-0.1)

                fig2 = draw_single_pic_TIP((pred[m].detach().cpu()*512/40), labels[m].detach().cpu(), withjt=False, show=False)
                fig2.savefig(os.path.join(save_fig_dir, f'step_{step}' + '_nojt_' + str(m) + '.png'), bbox_inches='tight', pad_inches=-0.1)

        if pred.max() > max_value:
            max_value = pred.max()

        metrics = cal_metrics(pred*512/40, (sampledImgs*0.5 + 0.5)*512)
        eval_reses.append(metrics)

        step += 1
    
    print(f'max value: {max_value} total: {total_num} step: {step}')
    metrics_res = np.array(eval_reses)
    test_metrics_res = np.array(metrics_res)
    test_metrics_res = test_metrics_res.sum(axis=0) / total_num
    test_metrics_res = np.around(test_metrics_res, 3)
    out_res = np.vstack((test_metrics_res.reshape(-1, 1), np.array([total_num]).reshape(-1, 1)))

    np.savetxt('wq/test_v2.txt', out_res, delimiter=' ', fmt="%.3f")
    print(f'result saved to test_v2.txt')

