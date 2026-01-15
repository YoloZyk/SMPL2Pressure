import cv2
from cv2 import GaussianBlur
import math
import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
p_max = 4095
# p_max = 512
p_h_value = 222

def TIP_SMPL_beta():
    betas = [
        [-0.8820,  1.2008,  1.2062,  1.2282, -0.0927,  0.1708, -0.6001,  0.2043, 0.9174, -0.1426],
        [-0.6597, 1.3122, 1.0209, 1.1327, 0.1016, 0.1386, -0.9671, 0.2936, 1.2099, -0.2536],
        [-0.5680, 0.9178, 1.1767, 1.3747, 0.0969, 0.2419, -0.2868, 0.1553, 0.6098, 0.0840],
        [-0.8458, 1.1517, 0.9469, 1.0590, -0.1336, 0.1276, -0.6925, 0.0755, 0.8618, -0.3282],
        [-0.6194, 1.2335, 0.9315, 1.0480, 0.0298, 0.1466, -0.7683, 0.1790, 1.0423, -0.2970],
        [-0.9475, 1.1236, 0.5946, 1.2269, -0.1042, 0.1027, -0.9413, 0.4635, 1.0710, -0.2944],
        [-1.0586, 1.1702, 0.8292, 1.3169, -0.2216, 0.1574, -0.8096, 0.3444, 1.0254, -0.3644],
        [-1.2141, 1.2215, 0.7780, 1.2626, -0.4140, 0.0497, -0.7818, 0.3321, 0.8848, -0.4551],
        [-1.3852, 1.1779, 0.4624, 1.3916, -0.4005, 0.0924, -1.0214, 0.4581, 1.1004, -0.6617]
    ]

    return torch.FloatTensor(np.array(betas))

class JointAugmentor:
    def __init__(self):
        self.shoulder_range = (-15, 15)
        self.elbow_range = (-30, 30)
        self.hip_range = (-15, 15)
        self.idxs = [[2, 3, 4], [5, 6, 7], [8, 9, 13], [10, 11, 12]]

    def translate_joints(self, joints, dx=1, dy=1):
        new_joints = joints + np.array([dx, dy])
        # 边界裁剪至[0,55]和[0,39]
        new_joints = np.clip(new_joints, [0, 0], [39, 55])
        return new_joints

    def hierarchical_rotate(self, joints, idxs, part='upper', angle1=None, angle2=None):
        # 示例：右臂层次旋转（肩→肘→腕）
        grand_joint = joints[idxs[0]]  # 右肩索引
        parent_joint = joints[idxs[1]]  # 右肘索引
        child_joint = joints[idxs[2]]  # 右腕索引

        # 肩关节旋转
        if part == 'upper':
            shoulder_angle = np.random.uniform(*self.shoulder_range)
            if angle1:
                shoulder_angle = angle1
            new_elbow = self.rotate_limb(grand_joint, parent_joint, shoulder_angle)
            new_wrist = self.rotate_limb(grand_joint, child_joint, shoulder_angle)

            # 肘关节旋转（基于新肩位置）
            elbow_angle = np.random.uniform(*self.elbow_range)
            if angle2:
                elbow_angle = angle2
            new_wrist = self.rotate_limb(new_elbow, new_wrist, elbow_angle)
            joints[idxs[1]] = new_elbow
            joints[idxs[2]] = new_wrist

        elif part == 'lower':
            hip_angle = np.random.uniform(*self.hip_range)
            if angle1:
                hip_angle = angle1
            new_knee = self.rotate_limb(grand_joint, parent_joint, hip_angle)
            new_ankle = self.rotate_limb(grand_joint, child_joint, hip_angle)
            joints[idxs[1]] = new_knee
            joints[idxs[2]] = new_ankle

        joints = np.clip(joints, [0, 0], [39, 55])
        return joints  # 更新后的关节点坐标

    def rotate_limb(self, parent_joint, child_joint, angle):
        # 将子节点坐标转换为以父节点为原点的相对坐标
        relative_pos = child_joint - parent_joint
        # 旋转矩阵计算
        rad = np.radians(angle)
        rot_matrix = np.array([[np.cos(rad), -np.sin(rad)],
                               [np.sin(rad), np.cos(rad)]])
        new_relative = np.dot(rot_matrix, relative_pos)
        return parent_joint + new_relative


class BatchKinematicAugmentor:
    def __init__(self,
                 limb_rotate_prob=0.3,
                 global_shift_prob=0.5,
                 shift_range=(-2, 2), dataset='SIP'):
        """
        批量处理增强器初始化参数：
        - limb_rotate_prob: 单侧肢体旋转概率
        - global_shift_prob: 整体平移概率
        - shift_range: 平移范围（整数）
        """
        # 肢体定义（基于14关节点索引）
        if dataset == 'TIP':
            self.limbs = {
                'left_arm': [3, 5, 7],  # [肩, 肘, 腕]
                'right_arm': [4, 6, 8],
                'left_leg': [10, 12, 14],  # [髋, 膝, 踝]
                'right_leg': [9, 11, 13]
            }
        elif dataset == 'SIP':
            self.limbs = {
                'left_arm': [5, 6, 7],  # [肩, 肘, 腕]
                'right_arm': [2, 3, 4],
                'left_leg': [10, 11, 12],  # [髋, 膝, 踝]
                'right_leg': [8, 9, 13]
            }

        # 旋转角度约束（单位：度）
        self.angle_ranges = {
            'shoulder': (-15, 15),
            'elbow': (0, 20),  # 只允许向内侧旋转
            'hip': (-10, 10)
        }

        # 概率参数
        self.limb_rotate_prob = limb_rotate_prob
        self.global_shift_prob = global_shift_prob
        self.shift_range = shift_range

    def __call__(self, batch_joints):
        """
        输入: batch_joints - [B, 14, 2] 的批量关节点坐标
        输出: 增强后的批量坐标
        """
        device = batch_joints.device
        joints_np = batch_joints.cpu().numpy()

        # 对每个样本独立增强
        for i in range(joints_np.shape[0]):
            joints_np[i] = self._augment_single(joints_np[i])

        return torch.FloatTensor(joints_np).to(device)

    def _augment_single(self, joints):
        """
        处理单个样本的增强逻辑
        """
        joints = joints.copy()

        # 肢体旋转增强
        for limb_type, indices in self.limbs.items():
            if np.random.rand() < self.limb_rotate_prob:
                self._rotate_limb(joints, indices, limb_type)

        # 全局平移增强
        if np.random.rand() < self.global_shift_prob:
            joints = self._shift_joints(joints)

        return np.clip(joints, [0, 0], [39, 55])

    def _rotate_limb(self, joints, indices, limb_type):
        """
        肢体层次旋转（支持肘关节方向限制）
        """
        shoulder_idx, elbow_idx, wrist_idx = indices

        # 第一级旋转（肩/髋关节）
        joint_type = 'shoulder' if 'arm' in limb_type else 'hip'
        angle = np.random.uniform(*self.angle_ranges[joint_type])
        self._rotate_bone(joints[shoulder_idx], joints[elbow_idx], angle)
        self._rotate_bone(joints[shoulder_idx], joints[wrist_idx], angle)

        # 第二级旋转（肘/膝关节）
        if 'arm' in limb_type:
            angle = np.random.uniform(*self.angle_ranges['elbow'])

            # 肘关节方向修正（右手系下内侧旋转）
            if 'right' in limb_type:
                angle *= -1  # 右侧肘关节反向旋转

            self._rotate_bone(joints[elbow_idx], joints[wrist_idx], angle)

    def _rotate_bone(self, parent, child, angle):
        """
        单骨骼旋转核心算法（保持骨骼长度）
        """
        radians = np.radians(angle)
        offset = child - parent

        # 旋转矩阵（二维）
        rot_matrix = np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)]
        ])

        # 应用旋转
        new_offset = np.dot(rot_matrix, offset)
        child[:] = parent + new_offset

    def _shift_joints(self, joints):
        """
        整像素平移（范围-2~2）
        """
        dx = np.random.randint(*self.shift_range)
        dy = np.random.randint(*self.shift_range)

        # 应用位移并裁剪
        shifted = joints + [dx, dy]
        return shifted  # 压力图尺寸56x40

class BatchKinematicAugmentor_test:
    def __init__(self,
                 limb_rotate_prob=1,
                 global_shift_prob=0,
                 shift_range=(-2, 2), dataset='SIP'):
        """
        批量处理增强器初始化参数：
        - limb_rotate_prob: 单侧肢体旋转概率
        - global_shift_prob: 整体平移概率
        - shift_range: 平移范围（整数）
        """
        # 肢体定义（基于14关节点索引）
        if dataset == 'TIP':
            self.limbs = {
                'left_arm': [3, 5, 7],  # [肩, 肘, 腕]
                'right_arm': [4, 6, 8],
                'left_leg': [10, 12, 14],  # [髋, 膝, 踝]
                'right_leg': [9, 11, 13]
            }
        elif dataset == 'SIP':
            self.limbs = {
                'left_arm': [5, 6, 7],  # [肩, 肘, 腕]
                'right_arm': [2, 3, 4],
                'left_leg': [10, 11, 12],  # [髋, 膝, 踝]
                'right_leg': [8, 9, 13]
            }

        # 旋转角度约束（单位：度）
        self.angle_ranges = {
            'shoulder': (-15, 15),
            'elbow': (0, 20),  # 只允许向内侧旋转
            'hip': (-10, 10)
        }

        # 概率参数
        self.limb_rotate_prob = limb_rotate_prob
        self.global_shift_prob = global_shift_prob
        self.shift_range = shift_range

    def __call__(self, batch_joints, angle1=-100, angle2=-100, move_limbs=None, flip_rotate=False):
        """
        输入: batch_joints - [B, 14, 2] 的批量关节点坐标
        输出: 增强后的批量坐标
        """
        device = batch_joints.device
        joints_np = batch_joints.cpu().numpy()

        # 对每个样本独立增强
        for i in range(joints_np.shape[0]):
            joints_np[i] = self._augment_single(joints_np[i], angle1=angle1, angle2=angle2, move_limbs=move_limbs, flip_rotate=flip_rotate)

        return torch.FloatTensor(joints_np).to(device)

    def _augment_single(self, joints, angle1=-100, angle2=-100, move_limbs=None, flip_rotate=False):
        """
        处理单个样本的增强逻辑
        """
        joints = joints.copy()

        # 肢体旋转增强
        i = 0
        if move_limbs:
            for (limb_type, indices) in self.limbs.items():
                if move_limbs[i]:
                    if i%2==0 and flip_rotate:
                        self._rotate_limb(joints, indices, limb_type, -angle1, angle2)
                    else:
                        self._rotate_limb(joints, indices, limb_type, angle1, angle2)
                i += 1
        else:
            for limb_type, indices in self.limbs.items():
                if np.random.rand() < self.limb_rotate_prob:
                    self._rotate_limb(joints, indices, limb_type, angle1, angle2)

        # 全局平移增强
        if np.random.rand() < self.global_shift_prob:
            joints = self._shift_joints(joints)

        return np.clip(joints, [0, 0], [39, 55])

    def _rotate_limb(self, joints, indices, limb_type, angle1=-100, angle2=-100):
        """
        肢体层次旋转（支持肘关节方向限制）
        """
        shoulder_idx, elbow_idx, wrist_idx = indices

        # 第一级旋转（肩/髋关节）
        joint_type = 'shoulder' if 'arm' in limb_type else 'hip'
        if angle1 == -100:
            angle = np.random.uniform(*self.angle_ranges[joint_type])
        else:
            angle = angle1
        self._rotate_bone(joints[shoulder_idx], joints[elbow_idx], angle)
        self._rotate_bone(joints[shoulder_idx], joints[wrist_idx], angle)

        # 第二级旋转（肘/膝关节）
        if 'arm' in limb_type:
            if angle2 == -100:
                angle = np.random.uniform(*self.angle_ranges['elbow'])
            else:
                angle = angle2

            # 肘关节方向修正（右手系下内侧旋转）
            if 'right' in limb_type:
                angle *= -1  # 右侧肘关节反向旋转

            self._rotate_bone(joints[elbow_idx], joints[wrist_idx], angle)

    def _rotate_bone(self, parent, child, angle):
        """
        单骨骼旋转核心算法（保持骨骼长度）
        """
        radians = np.radians(angle)
        offset = child - parent

        # 旋转矩阵（二维）
        rot_matrix = np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)]
        ])

        # 应用旋转
        new_offset = np.dot(rot_matrix, offset)
        child[:] = parent + new_offset

    def _shift_joints(self, joints):
        """
        整像素平移（范围-2~2）
        """
        dx = np.random.randint(*self.shift_range)
        dy = np.random.randint(*self.shift_range)

        # 应用位移并裁剪
        shifted = joints + [dx, dy]
        return shifted  # 压力图尺寸56x40

def draw_single_pic(p_data,ps_jt=None,withjt=True):
    BODY_14_pairs = np.array([
        [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [8, 9], [9, 13], [10, 11], [11, 12]
    ])
    p_data = p_data.cpu()
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
    axes.matshow(p_data, vmin=10, vmax=300)

    if withjt:
        ps_jt = ps_jt.cpu()
        #print(ps_jt.shape)
        ps_jt = ps_jt.reshape(14, 2).transpose(1, 0)
        ps_jtx, ps_jty = ps_jt[0, :], ps_jt[1, :]
        #print(ps_jtx, ps_jty)
        #ps_jtx, ps_jty = ps_jt[:, 0] , ps_jt[:, 1]
        axes.scatter(ps_jtx,ps_jty,s=120.,color='darkgreen',linewidths=2,edgecolors='gold')
        for m in range(BODY_14_pairs.shape[0]):
            x = (ps_jtx[BODY_14_pairs[m][0]], ps_jtx[(BODY_14_pairs[m][1])])
            y = (ps_jty[BODY_14_pairs[m][0]], ps_jty[(BODY_14_pairs[m][1])])
            axes.plot(x, y, color='r')

    axes.set_xticks([])
    axes.set_yticks([])
    plt.axis('off')

    return fig


def draw_joints(labels):
    BODY_14_pairs = np.array([
        [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [8, 9], [9, 13], [10, 11], [11, 12]
    ])
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5,7))
    ps_jt = labels.detach().cpu()
    ps_jt[:,1]=ps_jt[:,1]*-1
    # print(ps_jt.shape)
    ps_jt = ps_jt.reshape(14, 2).transpose(1, 0)
    ps_jtx, ps_jty = ps_jt[0, :], ps_jt[1, :]
    # print(ps_jtx, ps_jty)
    # ps_jtx, ps_jty = ps_jt[:, 0] , ps_jt[:, 1]
    axes.scatter(ps_jtx, ps_jty, s=120., color='darkgreen', linewidths=2, edgecolors='gold')
    for m in range(BODY_14_pairs.shape[0]):
        x = (ps_jtx[BODY_14_pairs[m][0]], ps_jtx[(BODY_14_pairs[m][1])])
        y = (ps_jty[BODY_14_pairs[m][0]], ps_jty[(BODY_14_pairs[m][1])])
        axes.plot(x, y, color='r')

    axes.set_xticks([])
    axes.set_yticks([])
    plt.axis('off')
    plt.show()

def ssim_ps(img1, img2):
  C1 = (0.01 * p_max)**2
  C2 = (0.03 * p_max)**2
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

#定义psnr函数
def psnr_ps(img1,img2):
    #print(img1.shape,img2.shape)
    mse = np.mean( (img1/p_max - img2/p_max) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_alloutline(img1, img2):
    return get_outline(img1)+get_outline(img2)

def get_SensingThresholdValue(data):
    return get_rms(data.reshape(-1)) * 0.2

def get_outline(data):
    zone = data > get_SensingThresholdValue(data)
    return zone.astype(float)

def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def pcs(img1, img2):

    zone1 = img1 > get_SensingThresholdValue(img1)
    zone2 = img2 > get_SensingThresholdValue(img2)
    zone3 = zone1 + zone2
    zone = zone3 > 0
    all_pts = np.sum(zone)
    threshold = p_max * 0.05
    # threshold = p_max * 0.01
    img1 = img1 * zone
    img2 = img2 * zone

    dis_img = np.absolute(img1 - img2)
    right_pts = np.sum(dis_img < threshold) - np.size(zone) + np.sum(zone)
    return right_pts / all_pts, zone, all_pts

def cal_metrics(imgs1,imgs2):
    # import pdb; pdb.set_trace()
    criterionContent=nn.L1Loss()
    num=imgs1.shape[0]
    ssim_list=[]
    psnr_list=[]
    pcs_list=[]
    L1_list=[]
    for i in range(num):
        img1=imgs1[i]
        img2=imgs2[i]
        img1 = np.array(img1.cpu()).astype(np.float64)
        img2 = np.array(img2.cpu()).astype(np.float64)
        # img1=img1.reshape(img1.shape[0],img1.shape[1])
        # img2=img2.reshape(img2.shape[0],img2.shape[1])
        ssim_list.append(ssim_ps(img1,img2))
        #ssim_list.append(img1,img2)
        psnr_list.append(psnr_ps(img1,img2))
        pcs_list.append(pcs(img1,img2)[0])
        #L1_list.append(criterionContent(img1,img2).item())
        L1_list.append(np.absolute(img1 - img2).mean())
    #print(sum(ssim_list))
    #print(psnr_list)
    #return ssim_list.sum(),psnr_list.sum(),pcs_list.sum(),L1_list.sum()
    return np.array(ssim_list).sum(),np.array(psnr_list).sum(),np.array(pcs_list).sum(),np.array(L1_list).sum()

def norm2raw(data,max_value):
    return (data+1)*(max_value/2)

def ps_features(data):
    body_ps = get_SensingThresholdValue(data)
    #warning_ps = body_ps * 5
    warning_ps = p_h_value

    mask1 = (data > body_ps).astype(float)
    mask2 = (data > warning_ps).astype(float)

    labelled, num = seed_filling(mask2)
    if num > 0:
        return num, np.sum(labelled > 0), np.mean(data[labelled > 0])
    else:
        return 0, 0, 0

def cal_euclidean_realdis(joint_predict, joint_true):
    joint_predict_tmp = torch.from_numpy(joint_predict.copy())
    joint_true_tmp = torch.from_numpy(joint_true.copy())
    # print(joint_true.max())
    # 床单，像素转cm
    # print(joint_true_tmp[:,:,0].max(), joint_true_tmp[:,:,1].max())
    joint_predict_tmp[:, :, 0] = joint_predict_tmp[:, :, 0] * 2.4  # column, 对应40
    joint_predict_tmp[:, :, 1] = joint_predict_tmp[:, :, 1] * 3.1  # row, 对应56
    joint_true_tmp[:, :, 0] = joint_true_tmp[:, :, 0] * 2.4  # column, 对应40
    joint_true_tmp[:, :, 1] = joint_true_tmp[:, :, 1] * 3.1  # row, 对应56
    # print(joint_true.max())
    euclidean = nn.PairwiseDistance(p=2)
    return euclidean(joint_predict_tmp, joint_true_tmp).numpy()

def cal_euclidean_realdis_tensor(joint_predict, joint_true, sensor_x=2.4, sensor_y=3.1):
    joint_predict_tmp = torch.clone(joint_predict)
    joint_true_tmp = torch.clone(joint_true)
    # print(joint_true.max())
    # 床单，像素转cm
    # print(joint_true_tmp[:,:,0].max(), joint_true_tmp[:,:,1].max())
    joint_predict_tmp[:, :, 0] = joint_predict_tmp[:, :, 0] * sensor_x  # column, 对应40
    joint_predict_tmp[:, :, 1] = joint_predict_tmp[:, :, 1] * sensor_y  # row, 对应56
    joint_true_tmp[:, :, 0] = joint_true_tmp[:, :, 0] * sensor_x  # column, 对应40
    joint_true_tmp[:, :, 1] = joint_true_tmp[:, :, 1] * sensor_y  # row, 对应56
    # print(joint_true.max())
    euclidean = nn.PairwiseDistance(p=2)
    return euclidean(joint_predict_tmp, joint_true_tmp)

def seed_filling(image, diag=True):
    '''
    用Seed-Filling算法标记图片中的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        图片数组,零值表示背景,非零值表示特征.

    diag : bool
        指定邻域是否包含四个对角.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol), dtype int
        表示连通域标签的数组,0表示背景,从1开始表示标签.

    nlabel : int
        连通域的个数.
    '''
    # 用-1表示未被标记过的特征像素.
    image = np.asarray(image, dtype=bool)
    nrow, ncol = image.shape
    labelled = np.where(image, -1, 0)

    # 指定邻域的范围.
    if diag:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)
        ]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def get_neighbor_indices(row, col):
        '''获取(row, col)位置邻域的下标.'''
        for (dx, dy) in offsets:
            x = row + dx
            y = col + dy
            if 0 <= x < nrow and 0 <= y < ncol:
                yield x, y

    label = 1
    for row in range(nrow):
        for col in range(ncol):
            # 跳过背景像素和已经标记过的特征像素.
            if labelled[row, col] != -1:
                continue
            # 标记当前位置和邻域内的特征像素.
            current_indices = []
            labelled[row, col] = label
            for neighbor_index in get_neighbor_indices(row, col):
                if labelled[neighbor_index] == -1:
                    labelled[neighbor_index] = label
                    current_indices.append(neighbor_index)
            # 不断寻找与特征像素相邻的特征像素并加以标记,直至再找不到特征像素.
            while current_indices:
                current_index = current_indices.pop()
                labelled[current_index] = label
                for neighbor_index in get_neighbor_indices(*current_index):
                    if labelled[neighbor_index] == -1:
                        labelled[neighbor_index] = label
                        current_indices.append(neighbor_index)
            label += 1

    return labelled, label - 1
def cal_ps_metric(img1, img2):
    # img1:netG data, img2:true data
    num1, area1, mean1 = ps_features(img1)
    num2, area2, mean2 = ps_features(img2)

    #print(mean1,mean2)
    return 1 - np.absolute(num1 - num2) / num2, 1 - np.absolute(area1 - area2) / area2, 1 - np.absolute(
        mean1 - mean2) / mean2

def cal_ps_metrics(imgs1,imgs2):
    num_dis,area_dis,mean_dis=[],[],[]
    len=imgs1.shape[0]
    for i in range(len):
        img1,img2=imgs1[i],imgs2[i]
        img1 = np.array(img1.cpu()).astype(np.float64)
        img2 = np.array(img2.cpu()).astype(np.float64)
        img1 = img1.reshape(img1.shape[1], img1.shape[2])
        img2 = img2.reshape(img2.shape[1], img2.shape[2])

        num,area,mean=cal_ps_metric(img1,img2)
        num_dis.append(num)
        area_dis.append(area)
        mean_dis.append(mean)
        #print(num,area,mean)

    return np.array(num_dis).sum(),np.array(area_dis).sum(),np.array(mean_dis).sum()


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale=2):
        super(upsampling, self).__init__()
        assert isinstance(scale, int)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)
        self.scale = scale

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        xout = self.conv(F.interpolate(input=x, size=(h, w), mode='nearest', align_corners=True))
        return xout

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

def draw_single_pic_SLP(eval_fake_B, ps_jt, withjt=True, show=True, max_v=0.5):
    BODY_14_pairs = np.array([
        [12, 13], [12, 8], [8, 7], [7, 6], [12, 9], [9, 10], [10, 11],
        [2, 1], [1, 0], [3, 4], [4, 5]
    ])

    # eval_fake_B = eval_fake_B.cpu()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
    if eval_fake_B.max() > 10:
        # axes.matshow(eval_fake_B, vmin=0, vmax=200)
        axes.matshow(eval_fake_B, vmax=50)
    else:
        # axes.matshow(eval_fake_B, vmax=max_v)
        axes.matshow(((eval_fake_B*0.5)+0.5)*255, vmax=50)
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

def draw_single_pic_cross(eval_fake_B, ps_jt, withjt=True, max_v=None, min_v=None):
    BODY_14_pairs = np.array([
        [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [8, 9], [9, 13], [10, 11], [11, 12]
    ])

    # eval_fake_B = eval_fake_B.cpu()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
    if max_v:
        # axes.matshow(eval_fake_B, vmin=10, vmax=max_v)
        axes.matshow(eval_fake_B, vmax=max_v, vmin=min_v)
    else:
        # axes.matshow(eval_fake_B, vmax=max_v)
        axes.matshow(eval_fake_B)
    # axes.set_title(f'{model_name}', fontsize=16)

    if withjt:
        # ps_jt = ps_jt.cpu()
        ps_jtx, ps_jty = ps_jt[:, 0], ps_jt[:, 1]
        # ps_jtx, ps_jty = ps_jt[:, 0] , ps_jt[:, 1]
        axes.scatter(ps_jtx, ps_jty, s=120., color='darkgreen', linewidths=2, edgecolors='gold')
        # for m in range(BODY_14_pairs.shape[0]):
        #     # print(m)
        #     x = (ps_jtx[BODY_14_pairs[m][0]], ps_jtx[(BODY_14_pairs[m][1])])
        #     y = (ps_jty[BODY_14_pairs[m][0]], ps_jty[(BODY_14_pairs[m][1])])
        #
        #     # x=(ps_jtx[0],ps_jty[0])
        #     # y=(ps_jtx[1],ps_jty[1])
        #     axes.plot(x, y, color='r')

    axes.set_xticks([])
    axes.set_yticks([])
    plt.axis('off')
    plt.show()

    return fig

def draw_single_joint_test(ps_jt, withjt=True):
    BODY_14_pairs = np.array([
        [0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [8, 9], [9, 13], [10, 11], [11, 12]
    ])

    # eval_fake_B = eval_fake_B.cpu()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
    # if eval_fake_B.max()>1:
    #     axes.matshow(eval_fake_B, vmin=10, vmax=300)
    # else:
    #     axes.matshow(eval_fake_B)
    # axes.set_title(f'{model_name}', fontsize=16)

    if withjt:
        # ps_jt = ps_jt.cpu()
        ps_jtx, ps_jty = ps_jt[:, 0], ps_jt[:, 1]*-1
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
    plt.show()

    return fig

def downstreamwork_dataupsampling(data,joints,device,joint_task=False, img_size=192):
    #data=data.detach().cpu()
    base_noise=0
    hor_margin = (img_size - data.shape[3]) // 2
    ver_margin = (img_size - data.shape[2]) // 2
    data1 = np.zeros((data.shape[0],1, img_size, img_size)).astype(np.float32) - base_noise
    data1, joints = torch.tensor(data1).to(device), torch.tensor(joints).to(device)
    data1[:,:, ver_margin: ver_margin + data.shape[2], hor_margin: hor_margin + data.shape[3]] = data
    #print(joints.shape)
    #joints = joints.reshape(-1, 2, 14).transpose(0, 2, 1)
    joints[:, :, 0] += hor_margin
    joints[:, :, 1] += ver_margin


    if joint_task:
        joints[:, :, :] /= 4  # 192*192 -> 48*48
        joints_heatmap = np.zeros((joints.shape[0], 14, 48*48), dtype=np.float32)
        for i in range(joints.shape[0]):
            joints_heatmap[i] = generate_target(joints[i]).reshape(14, 48*48)
        return data1, torch.tensor(joints_heatmap).to(device)

    return data1, joints


'''def downstreamwork_dataupsampling(data,joints,device,joint_task=False, img_size=192):
    #data=data.detach().cpu()
    base_noise=0
    hor_margin = (img_size - data.shape[3]) // 2
    ver_margin = (img_size - data.shape[2]) // 2
    data1 = np.zeros((data.shape[0],1, img_size, img_size)).astype(np.float32) - base_noise
    data1, joints = torch.tensor(data1).to(device), torch.tensor(joints).to(device)
    data1[:,:, ver_margin: ver_margin + data.shape[2], hor_margin: hor_margin + data.shape[3]] = data
    #print(joints.shape)
    #joints = joints.reshape(-1, 2, 14).transpose(0, 2, 1)
    joints[:, :, 0] += hor_margin
    joints[:, :, 1] += ver_margin


    if joint_task:
        joints[:, :, :] /= 4  # 192*192 -> 48*48
        joints_heatmap = np.zeros((joints.shape[0], 14, 48 * 48), dtype=np.float32)
        for i in range(joints.shape[0]):
            #print(joints.shape,joints[i].shape)
            joints_heatmap[i] = generate_target(joints[i]).reshape(14, 48 * 48)
        return data1, torch.tensor(joints_heatmap).to(device)

    return data1, joints'''

def ht_map2joint_pred(pred_result):
    pred_result = pred_result.detach().cpu()

    rate = 0.75
    pred_joint = []
    for x in range(np.array(pred_result).shape[0]):
        true_joint_per = []
        pred_joint_per = []
        for y in range(14):


            pos = np.argmax(pred_result[x][y])
            pos_1 = pos // 48 * 4
            pos_0 = pos % 48 * 4

            pos = np.argsort(np.array(pred_result[x][y]).reshape(-1))[-2]
            pos_1 = (pos // 48 * 4) * (1 - rate) + pos_1 * rate
            pos_0 = (pos % 48 * 4) * (1 - rate) + pos_0 * rate
            pred_joint_per.append([pos_0, pos_1])

        pred_joint.append(pred_joint_per)
        # print("seen pred frame {} success".format(i))

    pred_joint_xy = torch.tensor(np.array(pred_joint), dtype=float)
    #gt_joint_xy = torch.tensor(np.array(gt_joint), dtype=float)

    return pred_joint_xy

def ht_map2joint(pred_result,gt_result):
    pred_result = pred_result.detach().cpu()
    gt_result = gt_result.detach().cpu()
    #pred_joints, gt_joints = [], []

    rate = 0.75
    gt_joint = []
    pred_joint = []
    for x in range(np.array(pred_result).shape[0]):
        true_joint_per = []
        pred_joint_per = []
        for y in range(14):
            pos = np.argmax(gt_result[x][y])
            pos_1 = pos // 48 * 4
            pos_0 = pos % 48 * 4

            pos = np.argsort(np.array(gt_result[x][y]).reshape(-1))[-2]
            pos_1 = (pos // 48 * 4) * (1 - rate) + pos_1 * rate
            pos_0 = (pos % 48 * 4) * (1 - rate) + pos_0 * rate

            true_joint_per.append([pos_0, pos_1])

            pos = np.argmax(pred_result[x][y])
            pos_1 = pos // 48 * 4
            pos_0 = pos % 48 * 4

            pos = np.argsort(np.array(pred_result[x][y]).reshape(-1))[-2]
            pos_1 = (pos // 48 * 4) * (1 - rate) + pos_1 * rate
            pos_0 = (pos % 48 * 4) * (1 - rate) + pos_0 * rate
            pred_joint_per.append([pos_0, pos_1])
        gt_joint.append(true_joint_per)
        pred_joint.append(pred_joint_per)
        # print("seen pred frame {} success".format(i))

    pred_joint_xy = torch.tensor(np.array(pred_joint), dtype=float)
    gt_joint_xy = torch.tensor(np.array(gt_joint), dtype=float)

    return pred_joint_xy, gt_joint_xy

def generate_gaussian_heatmap(joint_coords, image_size=(64, 64), sigma=5, eps: float = 1e-6 ):
    """
    生成关节点的高斯热图。
    Args:
        image_size: 图像尺寸 (H, W)
        joint_coords: 关节点坐标 (14, 2)
        sigma: 高斯核标准差
    Returns:
        heatmap: 热图 (14, H, W)
    """
    B, N, _ = joint_coords.shape
    grid_x = joint_coords[..., 0]   # (B, 14)
    grid_y = joint_coords[..., 1]  # (B, 14)
    # Step 1: 将归一化坐标转换为图像实际坐标

    H, W = image_size
    x = torch.arange(W, dtype=torch.float32, device=joint_coords.device)  # (W,)
    y = torch.arange(H, dtype=torch.float32, device=joint_coords.device)  # (H,)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W), (H, W)
    # Step 3: 扩展维度以支持广播计算
    # 目标形状: (B, N, H, W)
    xx = xx[None, None, :, :]  # (1, 1, H, W)
    yy = yy[None, None, :, :]  # (1, 1, H, W)
    grid_x = grid_x[:, :, None, None]  # (B, N, 1, 1)
    grid_y = grid_y[:, :, None, None]  # (B, N, 1, 1)

    # Step 4: 计算高斯热图
    heatmap = torch.exp(-(
            ((xx - grid_x) ** 2) / (2 * sigma ** 2 + eps) +
            ((yy - grid_y) ** 2) / (2 * sigma ** 2 + eps)
    ))  # (B, N, H, W)
    return heatmap  # shape: (H, W)


def generate_gaussian_heatmap_grad(
        keypoints: torch.Tensor,  # 输入关节点坐标 (B, num_joints, 2), 范围 [0,1]
        heatmap_size: tuple = (64, 64),  # 热图尺寸 (H, W)
        sigma: float = 2.0  # 高斯核标准差
) -> torch.Tensor:
    """
    生成可导的高斯热图，支持梯度反向传播到输入坐标。
    """
    batch_size, num_joints, _ = keypoints.shape
    H, W = heatmap_size

    # 生成网格坐标 (H, W, 2)
    device = keypoints.device
    x_grid = torch.linspace(0, 1, W, device=device).view(1, 1, W)
    y_grid = torch.linspace(0, 1, H, device=device).view(1, H, 1)

    # 将关节点坐标转换到热图像素位置 (B, num_joints, 2) -> (B, num_joints, 1, 1, 2)
    keypoints_pixel = keypoints.view(batch_size, num_joints, 1, 1, 2)  # [B, J, 1, 1, 2]

    # 计算每个像素位置到关节点的距离 (可导操作)
    dx = x_grid - keypoints_pixel[..., 0]  # [B, J, H, W]
    dy = y_grid - keypoints_pixel[..., 1]  # [B, J, H, W]
    distance_sq = dx ** 2 + dy ** 2

    # 计算高斯热图 (可导)
    gaussian = torch.exp(-distance_sq / (2 * sigma ** 2))  # [B, J, H, W]

    # 归一化到 [0,1]（可选）
    # gaussian = gaussian / (gaussian.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)

    return gaussian  # 输出形状 (B, num_joints, H, W)

def joints_to_heatmaps(joints, image_size=(64, 64), sigma=5):
    """
    将关节点坐标转换为热图。

    Args:
        joints: tensor, 形状为 (B, 14, 2)，归一化范围在 [0, 1]
        image_size: tuple, 图像尺寸 (H, W)
        sigma: float, 高斯函数的标准差

    Returns:
        heatmaps: tensor, 形状为 (B, 14, H, W)
    """
    B = joints.size(0)
    H, W = image_size
    # heatmaps = torch.zeros(B, 14, H, W)

    # 归一化坐标
    joints_normalized = joints

    # for b in range(B):
    #     # 生成单个关节点的热图
    #     heatmap = generate_gaussian_heatmap(joints[b], (H, W), sigma)
    #     heatmaps[b, j] = heatmap
    heatmaps = generate_gaussian_heatmap(joints, (H, W))
    return heatmaps

class PureUpsampling(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(PureUpsampling, self).__init__()
        assert isinstance(scale, int)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        h, w = x.size(2) * self.scale, x.size(3) * self.scale
        if self.mode == 'nearest':
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode)
        else:
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode, align_corners=True)
        return xout

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def info_nce_loss(feats, mode='train'):
    #imgs, _ = batch
    #imgs = torch.cat(imgs, dim=0)

    # Encode all images
    #feats = self.convnet(imgs)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / 0.07
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Logging loss
    # Get ranking position of positive example
    # Logging ranking metrics

    return nll

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        #print(output.shape, target.shape)
        batch_size = output.size(0)
        num_joints = output.size(1)
        # print('output shape', output.size())
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)  # split along first dimension with
        # size 1  a list??    # N x n_jt split into [njt: Nxn_pix?]  a list?
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()  # N x long list
            heatmap_gt = heatmaps_gt[idx].squeeze()
            # if self.use_target_weight:
            #     loss += 0.5 * self.criterion(
            #         heatmap_pred.mul(target_weight[:, idx]),
            #         heatmap_gt.mul(target_weight[:, idx])
            #     )
            # else:
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class DamagedPointRepair(nn.Module):
    def __init__(self, lh, lw, kernel_size=3, thre_times=5, thre_point=1000):
        super(DamagedPointRepair, self).__init__()
        self.lh = lh
        self.lw = lw
        self.kernel_size = kernel_size
        self.thre_times = thre_times
        self.thre_point = thre_point
        self.img_coeff = self.gen_coeff_matrix()
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.float) / (kernel_size ** 2)

    def forward(self, img):

        mask = self.CheckBadPoint(img)

        x_list, y_list = np.where(mask == True)

        for x_index, y_index in zip(x_list, y_list):
            num = 0
            ele_sum = 0
            if x_index - 1 >= 0:
                num += 1
                ele_sum += img[x_index - 1, y_index]
            if x_index + 1 < self.lh:
                num += 1
                ele_sum += img[x_index + 1, y_index]
            if y_index - 1 >= 0:
                num += 1
                ele_sum += img[x_index, y_index - 1]
            if y_index + 1 < self.lw:
                num += 1
                ele_sum += img[x_index, y_index + 1]
            img[x_index, y_index] = ele_sum // num

        return img


    def gen_coeff_matrix(self):

        kernel_num = self.kernel_size ** 2

        img_coeff = np.ones((self.lh, self.lw), dtype=np.float)

        img_coeff[0, 0] = img_coeff[0, -1] = img_coeff[-1, 0] = img_coeff[-1, -1] = \
            kernel_num / (self.kernel_size // 2 + 1) ** 2

        img_coeff[0, 1: -1] = img_coeff[-1, 1: -1] = img_coeff[1: -1, 0] = img_coeff[1: -1, -1] = \
            self.kernel_size / (self.kernel_size // 2 + 1)

        return img_coeff


    def CheckBadPoint(self, img):

        img_filtered = cv2.filter2D(img, -1, self.kernel, borderType=0)

        img_mean = img_filtered * self.img_coeff

        mask = img > self.thre_times * img_mean

        mask = mask | (img > self.thre_point)

        return mask

class DamagedTrackRepair(nn.Module):
    def __init__(self, lh, lw, thre_value=30, thre_times=4, thre_num=5, mode=3):
        '''
        :param lh:
        :param lw:
        :param thre_value:
        :param thre_times:
        :param thre_num:
        :param mode: 1(30, 4, _)
                    2(_, 10, 5)
                    3(_, 4, 5)
        '''
        super(DamagedTrackRepair, self).__init__()
        self.lh = lh
        self.lw = lw
        self.mode = mode
        self.thre_value = thre_value
        self.thre_times = thre_times
        self.thre_num = thre_num
        self.x_track_index = []
        self.y_track_index = []
        self.mat = np.zeros((56, 40), dtype=np.int)
        if self.mode == 2:
            self.kernel_h = np.array([[1 / thre_times, 0, 1 / thre_times]], dtype=np.float)
            self.kernel_v = np.array([1 / thre_times, 0, 1 / thre_times], dtype=np.float)
        elif self.mode == 3:
            self.kernel_l = np.array([[1 / thre_times, 0, 0]], dtype=np.float)
            self.kernel_r = np.array([[0, 0, 1 / thre_times]], dtype=np.float)
            self.kernel_u = np.array([1 / thre_times, 0, 0], dtype=np.float)
            self.kernel_d = np.array([0, 0, 1 / thre_times], dtype=np.float)

    def check_bad_track_mod_1(self, data):
        x_track_index = []
        y_track_index = []
        self.mat = np.where(data > self.thre_value, 1, 0)
        x_sum = np.sum(self.mat, axis=1)
        y_sum = np.sum(self.mat, axis=0)
        for j in range(1, self.lw - 1):
            if y_sum[j - 1] > self.thre_times * y_sum[j] and y_sum[j + 1] > self.thre_times * y_sum[j]:
                y_track_index.append(j)
        for j in range(1, self.lh - 1):
            if x_sum[j - 1] > self.thre_times * x_sum[j] and x_sum[j + 1] > self.thre_times * x_sum[j]:
                x_track_index.append(j)
        return x_track_index, y_track_index

    def check_bad_track_mod_2(self, data):
        x_track_index = []
        y_track_index = []
        self.mat = data - cv2.filter2D(data, -1, self.kernel_h, borderType=0)
        self.mat = np.where(self.mat < 0, 1, 0)
        x_sum = np.sum(self.mat, axis=1)
        self.mat = data - cv2.filter2D(data, -1, self.kernel_v, borderType=0)
        self.mat = np.where(self.mat < 0, 1, 0)
        y_sum = np.sum(self.mat, axis=0)
        for j in range(1, self.lw - 1):
            if y_sum[j] > self.thre_num:
                y_track_index.append(j)
        for j in range(1, self.lh - 1):
            if x_sum[j] > self.thre_num:
                x_track_index.append(j)
        return x_track_index, y_track_index

    def check_bad_track_mod_3(self, data):
        x_track_index = []
        y_track_index = []
        self.mat = np.logical_and(data - cv2.filter2D(data, -1, self.kernel_r, borderType=0) < 0,
                                  data - cv2.filter2D(data, -1, self.kernel_l, borderType=0) < 0
                                  )
        self.mat = np.where(self.mat, 1, 0)
        y_sum = np.sum(self.mat, axis=0)
        self.mat = np.logical_and(data - cv2.filter2D(data, -1, self.kernel_u, borderType=0) < 0,
                                  data - cv2.filter2D(data, -1, self.kernel_d, borderType=0) < 0
                                  )
        self.mat = np.where(self.mat, 1, 0)
        x_sum = np.sum(self.mat, axis=1)
        for j in range(self.lw):
            if y_sum[j] > self.thre_num:
                y_track_index.append(j)
        for j in range(self.lh):
            if x_sum[j] > self.thre_num:
                x_track_index.append(j)
        return x_track_index, y_track_index

    def forward(self, data):
        if self.mode == 1:
            x_track_index, y_track_index = self.check_bad_track_mod_1(data)
        elif self.mode == 2:
            x_track_index, y_track_index = self.check_bad_track_mod_2(data)
        else:
            x_track_index, y_track_index = self.check_bad_track_mod_3(data)
        for val in y_track_index:
            data[:, val] = (data[:, val - 1] + data[:, val + 1]) // 2
        for val in x_track_index:
            data[val, :] = (data[val - 1, :] + data[val + 1, :]) // 2
        return data



def cal_euclidean_dis(pred, gt):
    euclidean = nn.PairwiseDistance(p=2)
    return euclidean(torch.from_numpy(pred), torch.from_numpy(gt)).numpy()
'''
class ValidFlip(nn.Module):
    def __init__(self, p, type=2):
        super(ValidFlip, self).__init__()
        self.typeSet = [-1, 0, 1]
        self.type = type
        self.p = p

    def forward(self, img_list):
        [img, joint] = img_list
        if random.uniform(0, 1) < self.p:
            return img_list
        else:
            if self.type == 2:
                self.type = random.choice(self.typeSet)

            if joint is not None:
                flip_joint = joint.copy()
                if self.type in [-1, 1]:
                    flip_joint[:, 0] = img.shape[1] - flip_joint[:, 0]
                if self.type in [0, -1]:
                    flip_joint[:, 1] = img.shape[0] - flip_joint[:, 1]
            else:
                flip_joint = None
            return [cv2.flip(img, self.type), flip_joint]

class ValidRotate(nn.Module):
    def __init__(self, p, threshold=0.05):
        super(ValidRotate, self).__init__()
        self.p = p
        self.threshold = threshold

    def forward(self, img_list):

        [img, joint] = img_list
        angles = [-30, 30, -20, 20, -10, 10, 5, -5, 0]
        angles = [random.randint(0, 180) - 90 for i in range(1)]
        if random.uniform(0, 1) < self.p:
            return img_list

        for angle in angles:

            sum_img = np.sum(img)

            trans = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
            tran_img = cv2.warpAffine(img, trans, (img.shape[1], img.shape[0]))

            rot_mat = np.array([[math.cos(angle * math.pi / 180), -math.sin(angle * math.pi / 180)],
                                [math.sin(angle * math.pi / 180), math.cos(angle * math.pi / 180)]])
            o_point = [img.shape[1] // 2, img.shape[0] // 2]
            if joint is not None:
                rot_joint = joint.copy()
                rot_joint[:, :2] = np.dot(joint[:, :2] - o_point, rot_mat) + o_point
            else:
                rot_joint = None
            # if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold and (not np.sum(rot_joint < 0)) and \
            #        (not np.sum(rot_joint[:, 0] >= img.shape[1])):
            #if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold:
            #     return [tran_img, rot_joint]
            return [tran_img, rot_joint]

class ValidShift(nn.Module):
    def __init__(self, p, threshold=0.05):
        super(ValidShift, self).__init__()
        self.p = p
        self.threshold = threshold

    def forward(self, img_list):
        [img, joint] = img_list
        shift = [12, 10, 8, 6, 4, 2, 1, 0]

        if random.uniform(0, 1) < self.p:
            return img_list

        sum_img = np.sum(img)

        if np.sum(img[:img.shape[0] // 2, :]) > 0.5 * sum_img:
            y_direction = 1
        else:
            y_direction = -1

        if np.sum(img[:, :img.shape[1] // 2]) > 0.5 * sum_img:
            x_direction = -1
        else:
            x_direction = 1

        for sft in shift:

            trans = np.float32([[1, 0, x_direction * sft], [0, 1, y_direction * sft]])
            tran_img = cv2.warpAffine(img, trans, (img.shape[1], img.shape[0]))

            if joint is not None:

                sft_joint = joint.copy()

                sft_joint[:, 0] += x_direction * sft
                sft_joint[:, 1] += y_direction * sft
            else:
                sft_joint = None
            
            #if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold and (not np.sum(sft_joint < 0)) and \
                    #(not np.sum(sft_joint[:, 0] > img.shape[1])) and \
                    #(not np.sum(sft_joint[:, 1] > img.shape[0])):
            
            if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold:
                return [tran_img, sft_joint]

        return img_list

'''

class Addnoise(nn.Module):
    def __init__(self, p=0, noise="gaussian",mean=0.0, variance=1.0, amplitude=20.0):
        super(Addnoise, self).__init__()
        self.p = p
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def forward(self, img_list):
        [img, img2, img3, joint] = img_list
        if random.uniform(0, 1) < self.p:
            return img_list
        else:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img1 = N + img
            img1[img1 < 0] = 0
            #print(self.type, self.type, self.type)
            return [img1, img2, img3, joint]

class ValidFlip(nn.Module):
    def __init__(self, p, type=2):
        super(ValidFlip, self).__init__()
        self.typeSet = [-1, 0, 1]
        self.type = type
        self.p = p

    def forward(self, img_list):
        [img, img2, img3, joint] = img_list
        if random.uniform(0, 1) < self.p:
            return img_list
        else:
            if self.type == 2:
                self.type = random.choice(self.typeSet)
                #self.type = -1

            if joint is not None:
                flip_joint = joint.copy()
                if self.type in [-1, 1]:
                    flip_joint[:, 0] = img.shape[1] - flip_joint[:, 0]
                if self.type in [0, -1]:
                    flip_joint[:, 1] = img.shape[0] - flip_joint[:, 1]
            else:
                flip_joint = None
            #print(self.type, self.type, self.type)
            return [cv2.flip(img, self.type), cv2.flip(img2, self.type), cv2.flip(img3, self.type), flip_joint]


class ValidRotate(nn.Module):
    def __init__(self, p, threshold=0.05):
        super(ValidRotate, self).__init__()
        self.p = p
        self.threshold = threshold

    def forward(self, img_list):

        [img, img2, img3, joint] = img_list
        angles = [-30, 30, -20, 20, -10, 10, 5, -5, 0]
        angles = [random.randint(0, 180) - 90 for i in range(1)]
        if random.uniform(0, 1) < self.p:
            return img_list

        for angle in angles:

            sum_img = np.sum(img)

            trans = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
            tran_img = cv2.warpAffine(img, trans, (img.shape[1], img.shape[0]))

            tran_img2 = cv2.warpAffine(img2, trans, (img.shape[1], img.shape[0]))

            tran_img3 = cv2.warpAffine(img3, trans, (img.shape[1], img.shape[0]))

            rot_mat = np.array([[math.cos(angle * math.pi / 180), -math.sin(angle * math.pi / 180)],
                                [math.sin(angle * math.pi / 180), math.cos(angle * math.pi / 180)]])
            o_point = [img.shape[1] // 2, img.shape[0] // 2]
            if joint is not None:
                rot_joint = joint.copy()
                rot_joint[:, :2] = np.dot(joint[:, :2] - o_point, rot_mat) + o_point
            else:
                rot_joint = None
            # if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold and (not np.sum(rot_joint < 0)) and \
            #        (not np.sum(rot_joint[:, 0] >= img.shape[1])):
            #if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold:
            #     return [tran_img, rot_joint]
            return [tran_img, tran_img2, tran_img3, rot_joint]


class ValidShift(nn.Module):
    def __init__(self, p, threshold=0.05):
        super(ValidShift, self).__init__()
        self.p = p
        self.threshold = threshold

    def forward(self, img_list):
        [img, img2, img3, joint] = img_list
        shift = [2, 1, 0]

        if random.uniform(0, 1) < self.p:
            return img_list

        sum_img = np.sum(img)

        if np.sum(img[:img.shape[0] // 2, :]) > 0.5 * sum_img:
            y_direction = 1
        else:
            y_direction = -1

        if np.sum(img[:, :img.shape[1] // 2]) > 0.5 * sum_img:
            x_direction = -1
        else:
            x_direction = 1

        for sft in shift:

            trans = np.float32([[1, 0, x_direction * sft], [0, 1, y_direction * sft]])
            tran_img = cv2.warpAffine(img, trans, (img.shape[1], img.shape[0]))
            tran_img2 = cv2.warpAffine(img2, trans, (img.shape[1], img.shape[0]))
            tran_img3 = cv2.warpAffine(img3, trans, (img.shape[1], img.shape[0]))

            if joint is not None:

                sft_joint = joint.copy()

                sft_joint[:, 0] += x_direction * sft
                sft_joint[:, 1] += y_direction * sft
            else:
                sft_joint = None
            
           # if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold and (not np.sum(sft_joint < 0)) and \
                    #(not np.sum(sft_joint[:, 0] > img.shape[1])) and \
                    #(not np.sum(sft_joint[:, 1] > img.shape[0])):
            
            if abs(1 - np.sum(tran_img) / sum_img) <= self.threshold:
                return [tran_img, tran_img2, tran_img3, sft_joint]

        return img_list

def generate_target(joints, sz_hm=[48, 48], sigma=2, gType='gaussian'):
    '''
	:param joints:  [num_joints, 3]
	:param joints_vis: n_jt vec     #  original n_jt x 3
	:param sigma: for gaussian gen, 3 sigma rule for effective area.  hrnet default 2.
	:return: target, target_weight(1: visible, 0: invisible),  n_jt x 1
	history: gen directly at the jt position, stride should be handled outside
	'''
    n_jt = len(joints)  #

    target_weight = np.ones((n_jt, 1), dtype=np.float32)
    # # target_weight[:, 0] = joints_vis[:, 0]
    # target_weight[:, 0] = joints_vis  # wt equals to vis

    assert gType == 'gaussian', \
        'Only support gaussian map now!'

    if gType == 'gaussian':
        target = np.zeros((n_jt,
                           sz_hm[1],
                           sz_hm[0]),
                          dtype=np.float32)

        tmp_size = sigma * 3

        #print(n_jt)
        for joint_id in range(n_jt):
            # feat_stride = self.image_size / sz_hm
            # mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            # mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            mu_x = int(joints[joint_id][0] + 0.5)  # in hm joints could be in middle,  0.5 to biased to the position.
            mu_y = int(joints[joint_id][1] + 0.5)

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= sz_hm[0] or ul[1] >= sz_hm[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], sz_hm[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], sz_hm[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], sz_hm[0])
            img_y = max(0, ul[1]), min(br[1], sz_hm[1])

            # target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    # print('min max', target.min(), target.max())
    # if self.use_different_joints_weight:
    # 	target_weight = np.multiply(target_weight, self.joints_weight)

    return target

class EarlyStopping(nn.Module):
    def __init__(self, p, length):
        super(EarlyStopping, self).__init__()
        self.eval_loss = []
        self.len = length
        self.p = p
        self.state = False

    def forward(self, x):

        self.eval_loss.append(x)

        if len(self.eval_loss) <= self.len:
                return False
        else:
            min_eval_loss = min(self.eval_loss)
            self.state = True
            for loss in self.eval_loss[-self.len:]:
                if loss / min_eval_loss - 1 <= self.p:
                    self.state = False
            return self.state

def loss_cal(label_path, pre, true):
    pass

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def vis_img_keypoints(img, joints, kp_thresh=0.4, alpha=1):
    img_rsz = cv2.resize(img, (int(img.shape[1] * 256 / img.shape[0]), 256))
    img_pseudo = cv2.applyColorMap(img_rsz, cv2.COLORMAP_JET)
    jt_rsz = joints * 1.33
    jt_t = jt_rsz.T
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, jt_t.shape[1] + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    kp_img = img_pseudo.copy()

    for i in range(jt_t.shape[1]):
        p1 = jt_t[0, i].astype(np.int32), jt_t[1, i].astype(np.int32)
        cv2.circle(
            kp_img, p1, radius=3, color=colors[i], thickness=1, lineType=cv2.LINE_AA
        )

    img_skel = cv2.addWeighted(img_pseudo, 1.0 - alpha, kp_img, alpha, 0)
    return img_skel

def data_preprocess(data, samplingScale=1, if_denoise=True, blurType=None, erode=False, sobelFilter=False):
    if if_denoise:
        DeNoise = DamagedPointRepair(data.shape[1], data.shape[2])
        for i in range(data.shape[0]):
            data[i] = DeNoise(data[i])

    #  Normalization
    #data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # -------------------------

    data = up_sampling(data, samplingScale)

    data = smoothing(data, blurType)

    if erode:
        pass

    if sobelFilter:
        pass

    return data

def up_sampling(data, scale):
    '''
    Copide form Tao Guo
    :param data:
    :param scale: times
    :return:
    '''
    if scale == 1:
        return data
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]

    new_rows, new_cols = int(data.shape[1] * scale), int(data.shape[2] * scale)
    resize_data = np.zeros((data.shape[0], new_rows, new_cols), dtype=np.float32)

    for i in range(data.shape[0]):
        img = resize(data[i, :, :], (new_cols, new_rows))
        resize_data[i, :, :] = img.astype(np.float32)

    return resize_data


def get_gaussian_kernel(kernel_size=5, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def smoothing(data, type=None):
    '''
    Copied from Tao Guo
    :param data:
    :param run:
    :return:
    '''
    if type == None:
        return data

    elif type == 'gaussian':

        for i in range(data.shape[0]):
            img = cv2.GaussianBlur(data[i, 0, :, :], (3, 3), 0)
            blur_layer = get_gaussian_kernel()
            #img = blur_layer(data[i, :, :])
            data[i, 0, :, :] = img

    elif type == 'median':
        for i in range(data.shape[0]):
            img = cv2.medianBlur(data[i, 0, :, :], 3)
            data[i, 0, :, :] = img

    elif type == 'median_x':
        for i in range(data.shape[0]):
            img = cv2.medianBlur(data[i, :, :], 3)
            data[i, :, :] = img

    elif type == 'mean':
        for i in range(data.shape[0]):
            img = cv2.blur(data[i, 0, :, :], (3, 3))
            data[i, 0, :, :] = img

    elif type == 'mean_x':
        for i in range(data.shape[0]):
            img = cv2.blur(data[i, 0, :, :], (1, 3))
            data[i, 0, :, :] = img

    elif type == 'mean_y':
        for i in range(data.shape[0]):
            img = cv2.blur(data[i, 0, :, :], (3, 1))
            data[i, 0, :, :] = img

    return data


def erosion_opencv(image, kernel=np.ones((1, 3), np.uint8)):

    return cv2.erode(image, kernel)

def sobel_filter_opencv(image, b_horizontial=False):

    if b_horizontial:
        return cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    else:
        return cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)

if __name__ == '__main__':
    '''
    unpro_path = r'G:\dataset\PostureData_smartbedsheet_v1.0\smart_bedsheet_data_set'
    div_path = r'G:\dataset\loso'
    #dataset_generate(unpro_path, div_path)
    #slp_dataset_generate(r"G:\dataset\SLP_dataset")

    #for i in range(13):
        #loso_dataset_generate(unpro_path, div_path, i)

    data = np.load(os.path.join(r"G:\dataset\SLP_dataset", "train_data.npy"))

    '''
    '''

    flip = ValidFlip(1, -1)
    shift = ValidShift(1, 0.05)
    rotate = ValidRotate(1, 0.05)

    i = 1000
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(data[i], vmax=100)
    plt.xticks([])
    plt.yticks([])
    plt.title("Pressure Data", fontsize=18)
    plt.subplot(2, 3, 2)
    repair = data[i]
    plt.imshow(repair, vmax=100)
    plt.xticks([])
    plt.yticks([])
    plt.title("Denoise", fontsize=18)
    test = smoothing(up_sampling(repair, 4), type='Gaussian')[0]
    plt.subplot(2, 3, 3)
    plt.imshow(test, vmax=200)
    plt.xticks([])
    plt.yticks([])
    plt.title("Gaussian", fontsize=18)
    test_flip = flip(test)
    plt.subplot(2, 3, 4)
    plt.imshow(test_flip, vmax=100)
    plt.xticks([])
    plt.yticks([])
    plt.title("Flip", fontsize=18)
    test_sft = shift(test)
    plt.subplot(2, 3, 5)
    plt.imshow(test_sft, vmax=100)
    plt.xticks([])
    plt.yticks([])
    plt.title("Shift", fontsize=18)
    test_rt = rotate(test)
    plt.subplot(2, 3, 6)
    plt.imshow(test_rt, vmax=100)
    plt.xticks([])
    plt.yticks([])
    plt.title("Rotate", fontsize=18)
    plt.show()
    '''

    data = np.load(r'/workspace/wzy1999/pm_data/data_pro.npy')
    label = np.load(r'/workspace/wzy1999/pm_data/label_pro.npy')

    a = np.zeros((data.shape[0], 64, 28))
    for i in range(data.shape[0]):
        a[i] = np.pad(data[i], ((0, 0), (0, 1)), 'constant', constant_values=(0, 0))

    data = up_sampling(a.astype('uint8'), 3)

    print(data.shape)

    plt.imshow(a[10000])
    plt.show()

    import random

    length = data.shape[0]

    index = list(range(length))

    random.shuffle(index)

    index = np.array(index)

    data, label = data[index], label[index]

    np.save(r"/workspace/wzy1999/pm_data/train_data.npy", data[: length * 8 // 10])
    np.save(r"/workspace/wzy1999/pm_data/train_label.npy", label[: length * 8 // 10])

    np.save(r"/workspace/wzy1999/pm_data/test_data.npy", data[length * 8 // 10: length * 9 // 10])
    np.save(r"/workspace/wzy1999/pm_data/test_label.npy", label[length * 8 // 10: length * 9 // 10])

    np.save(r"/workspace/wzy1999/pm_data/eval_data.npy", data[length * 9 // 10:])
    np.save(r"/workspace/wzy1999/pm_data/eval_label.npy", label[length * 9 // 10:])


