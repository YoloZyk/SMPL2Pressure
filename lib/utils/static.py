PressurePose_PATH = "/workspace/zyk/public_data/pressurepose/synth"
MoYo_PATH = "/workspace/zyk/public_data/moyo"
TIP_PATH = "/workspace/zyk/public_data/wzy_opt_dataset_w_feats"

SMPL_MODEL = "/workspace/zyk/smpl_models/"

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

