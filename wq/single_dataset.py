import os

import h5py
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import torchvision as tv

dataset_idx_mapping = [
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

name_group_map = {
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

three_fold = [
    ['wyc', 'lgr', 'wyx'],
    ['wq', 'nmt', 'twj'],
    ['xft', 'zyk', 'lz'],
]


class InBedPressureDataset(Dataset):
    def __init__(self, cfgs, mode='train', need_shape=False, path2=None, ratio=0, train_ratio=0.9):
        self.cfgs = cfgs
        self.dataset_name = cfgs['dataset_path']
        self.model_type = cfgs['dataset_mode']
        self.curr_fold = cfgs['curr_fold']
        self.mode = mode
        self.normalize = cfgs['normalize']
        self.need_shape = need_shape

        self.segments = []
        self.db_segmemts = []

        self.data_len = 0

        self.data = {}
        # self.db_segmemts = {}
        self.info = {
            'date': [],
            'name': [],
            'idx': [],
            'corner': [],
            'sensor_position': []
        }

        if self.model_type == 'unseen_group':
            if self.mode == 'train':
                for name in name_group_map:
                    if name == 'wyc':
                        for idx in [10, 11, 14]:
                            print(f'load train dataset: {idx}')
                            self.load_db(idx)
                    else:
                        for idx in range(name_group_map[name][0], name_group_map[name][1])[:-2]:
                            print(f'load train dataset: {idx}')
                            self.load_db(idx)

            elif self.mode == 'eval':
                for name in name_group_map:
                    if name == 'wyc':
                        idx = name_group_map[name][1] - 3
                    else:
                        idx = name_group_map[name][1] - 2
                    print(f'load val dataset: {idx}')
                    self.load_db(idx)
            else:
                for name in name_group_map:
                    if name == 'wyc':
                        idx = name_group_map[name][1] - 2
                    else:
                        idx = name_group_map[name][1] - 1
                    print(f'load test dataset: {idx}')
                    self.load_db(idx)

        elif self.model_type == 'unseen_subject':
            if self.mode == 'train' or self.mode == 'eval':
                for fold, name_list in enumerate(three_fold):
                    if fold != self.curr_fold - 1:
                        for name in three_fold[fold]:
                            for idx in range(name_group_map[name][0], name_group_map[name][1]):
                                print(f'load train dataset: {idx}')
                                self.load_db(idx)

            elif self.mode == 'test':
                for name in three_fold[self.curr_fold - 1]:
                    for idx in range(name_group_map[name][0], name_group_map[name][1]):
                        print(f'load val dataset: {idx}')
                        self.load_db(idx)

        pressure = self.data['pressure'].astype(np.float32)
        joints = self.data['keypoints_pi']
        shape = self.data['betas']
        if self.model_type == 'unseen_subject':
            if self.mode == 'train' or self.mode == 'eval':
                setup_seed(40)
                shuffle_ix = np.random.permutation(np.arange(pressure.shape[0]))
                # pressure, joints = pressure[shuffle_ix], joints[shuffle_ix]
                pressure, joints, shape = pressure[shuffle_ix], joints[shuffle_ix], shape[shuffle_ix]
                train_len = int(pressure.shape[0] * train_ratio)
                # print(train_len)
                if self.mode == 'train':
                    # pressure, joints = pressure[:train_len], joints[:train_len]
                    pressure, joints, shape = pressure[:train_len], joints[:train_len], shape[:train_len]
                else:
                    # pressure, joints = pressure[train_len:], joints[train_len:]
                    pressure, joints, shape = pressure[train_len:], joints[train_len:], shape[train_len:]
                print(pressure.shape)

        if path2 and ratio > 0:

            f2 = h5py.File(path2+'pure_data_fromdownstream_train_labels_augmented.h5', 'r')

            data_2 = f2['data_pure']
            print(data_2.shape)

            fake_len = int(pressure.shape[0] * ratio)

            data_2 = data_2[:fake_len].reshape(-1, 56, 40)
            pos_info_2 = f2['pos'][:fake_len].astype(np.float32)
            pos_info_2 = pos_info_2.reshape(-1, 15, 2)
            f2.close()

            joints = np.vstack((joints, pos_info_2))
            pressure = np.vstack((pressure, data_2))
            # joints = self.data['keypoints_pi']

        if self.normalize:

            pressure[pressure > 512] = 512
            # pressure = pressure / (512 - np.min(pressure))
            pressure = pressure / (512 )
            pressure = (pressure - 0.5) * 2
            #better to consider min_pressure as a fix value?


        self.data['pressure'] = pressure.copy()
        self.data['keypoints_pi'] = joints.copy()
        self.data['betas'] = shape.copy()

        print(pressure.shape)


    def load_fake_label2d(self):
        # path = '/workspace/wzy1999/projects/PIDHMR/results/KP/pimesh_None_20240830_2013/test.npz'
        # path = '/workspace/wzy1999/projects/PIDHMR/results/KP/pimesh_None_20240831_1423/test.npz'
        path = '/workspace/wzy1999/projects/PIDHMR/results/KP/pimesh_None_None_None_20240909_2147/test.npz'
        kp_db = np.load(path)
        self.test_kp_db = kp_db['kp_2d']

    def load_fake_label3d(self):
        # path = '/workspace/wzy1999/projects/PIDHMR/results/KP/pimesh_None_20240830_2013/test.npz'
        # path = '/workspace/wzy1999/projects/PIDHMR/results/KP/3D/pimesh_None_20240831_1438/test.npz'
        path = '/workspace/wzy1999/projects/PIDHMR/results/KP/3D/pimesh_None_None_None_20240911_1043/test.npz'

        if self.model_type == 'unseen_subject':
            paths = ['/workspace/wzy1999/projects/PIDHMR/results/KP/unsubject/1/3D/pimesh_None_None_None_20240919_2121/test.npz',
                     '/workspace/wzy1999/projects/PIDHMR/results/KP/unsubject/2/3D/pimesh_None_None_None_20240919_2123/test.npz',
                     '/workspace/wzy1999/projects/PIDHMR/results/KP/unsubject/3/3D/pimesh_None_None_None_20240919_2123/test.npz']
            path = paths[self.curr_fold-1]
        kp_db = np.load(path)
        self.test_3dkp_db = kp_db['kp_3d']

    def load_db(self, idx):
        db = dict(np.load(os.path.join(self.dataset_name, f'data_{idx}.npz'),
                     allow_pickle=True))

        sensor_position = db['infer_sensor_position']
        segments = db['segments']

        data = {
            'pressure': db['pressure'],
            # 'binary_pressure': db['binary_pressure'],
            # 'keypoints_pi': (sensor_position[0] + np.array([0, 55 * 0.0311]) - db['keypoints_meter_smooth']) / np.array(
            #             [-0.0195, 0.0311]),
            'keypoints_pi': db['keypoints_meter_smooth'] / np.array([0.0195, 0.0311]),
            'betas': db['label_betas'],
            'pose': db['label_pose'],
            'trans': db['label_trans'],
            'verts': db['label_verts'],
            # 'keypoints_3d': db['label_kp_3d']
            # 'verts': db['label_verts'],
            # 'keypoints_3d': db['label_joints'][:, :25, :]
        }

        # if self.model_type == 'unseen_group':
        #     protocol = 'protocol_1'
        # else:
        #     protocol = f'protocol_{self.curr_fold + 1}'
        #
        # if self.feature_ori == 'static':
        #     data['img_feats'] = db['static_feats_' + protocol]
        # elif self.feature_ori =='temporal':
        #     data['img_feats'] = db['temp_feats_' + protocol]
        # data['pred_smpl'] = db['pred_theta_' + protocol]


        for segment in segments:

            self.info['name'].append(db['name'])
            self.info['date'].append(db['date'])
            self.info['sensor_position'].append(db['infer_sensor_position'])
            self.info['corner'].extend(db['bed_corner_shift'][segment[0]: segment[1]])
            self.info['idx'].append(idx)

            if not len(self.data):
                for key in data.keys():
                    self.data[key] = data[key][segment[0]: segment[1]]
                self.segments.append(np.array(segment) - segment[0] + self.data_len)
                self.db_segmemts.append(segment)
                self.data_len += segment[1] - segment[0]
            else:
                for key in data.keys():
                    self.data[key] = np.concatenate([self.data[key], data[key][segment[0]: segment[1]]], axis=0)
                self.segments.append(np.array(segment) - segment[0] + self.data_len)
                self.db_segmemts.append(segment)
                self.data_len += segment[1] - segment[0]




    def __getitem__(self, index):
        # return {
        #     'images': torch.from_numpy(self.data['pressure'][index]),
        #     'gt_keypoints_2d': torch.from_numpy(self.data['keypoints_pi'][index]),
        # }
        transform = tv.transforms.Compose([
            # convert it to a tensor
            tv.transforms.ToTensor(),
        ])

        pure_as_tensor = transform(self.data['pressure'][index]).type(torch.FloatTensor)
        if self.need_shape:
            # return pure_as_tensor, (self.data['keypoints_pi'][index].astype(np.float32), self.data['betas'][index].astype(np.float32))
            return pure_as_tensor, np.concatenate((self.data['keypoints_pi'][index].astype(np.float32).reshape(-1), self.data['betas'][index].astype(np.float32).reshape(-1)))
        return pure_as_tensor, self.data['verts'][index], self.data['keypoints_pi'][index].astype(np.float32)

    def __len__(self):
        # return self.data_len
        return self.data['pressure'].shape[0]
        # return 100


    def get_segments(self):
        return self.segments

    def get_data_len(self):
        return self.data_len









