import os
import gc
import enum
import cv2
import torch
import argparse

from sfm_tools.feature_extract_match.utils.utils import remove_db, get_img_pairs
from sfm_tools.feature_extract_match.model.detect_match import extract_by_superpoint, match_by_superglue, filter_match_by_adalam
from sfm_tools.feature_extract_match.model.colmapClass import import_into_colmap

class GeneralConfig(enum.Enum):
    gim_lightglue_model_path = os.path.join(os.path.dirname(__file__), 'third_party/sp_sg_models/gim_lightglue_100h.ckpt')

    # keypoints paras
    feature_confs = {
        'superpoint':{
            'output': 'superpoint',
            'model': {
                'max_num_keypoints': 2048,
                'force_num_keypoints': True,
                'detection_threshold': 0.0,
                'nms_radius': 3,
                'trainable': False,
                'weights': gim_lightglue_model_path,
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': None,
            },
        },
    }

    # matched paras
    matcher_confs = {
        'superpoint':{
            'output': 'superglue',
            'model': {
                'name': 'superglue',
                'weights': gim_lightglue_model_path,
                'filter_threshold': 0.1,
                'flash': False,
                'checkpointed': True,
            },
        },
    }

    # adalam paras
    adalam_confs = {
        'area_ratio': 100,
        'search_expansion': 4,
        'ransac_iters': 128,          
        'min_inliers': 6,  
        'min_confidence': 200,          
        'orientation_difference_threshold': None,
        'scale_rate_threshold': None,
        'detected_scale_rate_threshold': 5,
        'refit': True,
        'force_seed_mnn': True,
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    }

class ImageMatchingDB:
    def __init__(self, data_root):
        self.device = torch.device('cuda:0')
        self.spsg_conf = GeneralConfig
        self.data_root = data_root

    def reconstruction(self, feature_dir, img_dir, sparse_init_dir):
        gc.collect()
        database_path = f"{feature_dir}/colmap.db"
        remove_db(database_path)
        import_into_colmap(img_dir, feature_dir=feature_dir, database_path=database_path, sparse_init_dir=sparse_init_dir)

    def run(self):
        gc.collect()

        img_dir = os.path.join(self.data_root, "images")  
        seg_dir = os.path.join(self.data_root, "segs")

        img_fnames = []
        for cam in os.listdir(img_dir):
            for img_name in os.listdir(os.path.join(img_dir, cam)):
                img_path = os.path.join(img_dir, cam, img_name)
                img_fnames.append(img_path)
        
        feature_dir = os.path.join(self.data_root, "colmap/feature_sp_sg")
        os.makedirs(feature_dir, exist_ok=True)

        # extract_features_by_superpoint
        extract_by_superpoint(img_fnames, device=self.device, feature_dir=feature_dir, seg_dir=seg_dir, config=self.spsg_conf)

        # create_match_pairs
        # exhaustive
        index_pairs = get_img_pairs(img_fnames)
        with open(os.path.join(feature_dir, 'match_pair.txt'), 'w') as pair_f:
            for pair_ids in index_pairs:
                line = img_fnames[pair_ids[0]] + ' ' + img_fnames[pair_ids[1]] + ' ' + str(pair_ids[0]) + ' ' + str(pair_ids[1]) + "\n"
                pair_f.write(line)
        print("index_pairs:{}".format(len(index_pairs)))

        # generate_matches_by_superglue
        match_by_superglue(img_fnames, index_pairs, feature_dir=feature_dir, device=self.device, config=self.spsg_conf)

        # filter matches by adalam
        filter_match_by_adalam(img_fnames, index_pairs, feature_dir=feature_dir, device=self.device, config=self.spsg_conf)

        # write result to colmap db
        sparse_init_dir = os.path.join(self.data_root, "colmap/sparse_init")
        self.reconstruction(feature_dir, img_dir, sparse_init_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="feature extraction and matching")
    parser.add_argument("--root_path", help="path to 3dgs format results")

    args = parser.parse_args()
    extract_match_db = ImageMatchingDB(args.root_path)
    extract_match_db.run()