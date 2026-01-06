PressurePose_PATH = "/workspace/zyk/public_data/pressurepose/synth"
MoYo_PATH = "/workspace/zyk/public_data/moyo"
TIP_PATH = "/workspace/zyk/public_data/wzy_opt_dataset_w_feats"

SMPL_MODEL = "/workspace/zyk/smpl_models/"
HD_SMPL_MODEL = '/workspace/zyk/SMPL2Pressure/assets/smpl_neutral_hd_vert_regressor_sparse.npz'

SMPL_PART_BOUNDS = '/workspace/zyk/SMPL2Pressure/assets/smpl_segments_bounds.pkl'
FID_TO_PART = '/workspace/zyk/SMPL2Pressure/assets/fid_to_part.pkl'
PART_VID_FID = '/workspace/zyk/SMPL2Pressure/assets/smpl_part_vid_fid.pkl'
HD_SMPL_MAP = '/workspace/zyk/SMPL2Pressure/assets/smpl_neutral_hd_sample_from_mesh_out.pkl'


DATASET_META = {
    'tip': {
        'max_p': 512.0,
        'crop_size': [56, 40], 
        'path': TIP_PATH
    }, 
    'pressurepose': {
        'max_p': 100.0, 
        'crop_size': [64, 27], 
        'path': PressurePose_PATH
    }, 
    'moyo': {
        'max_p': 64.0, 
        'crop_size': [110, 37], 
        'path': MoYo_PATH
    }
}

