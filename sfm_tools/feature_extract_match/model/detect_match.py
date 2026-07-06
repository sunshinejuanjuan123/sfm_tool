import torch
import cv2
from types import SimpleNamespace
import numpy as np
import h5py
from tqdm import tqdm
import collections.abc as collections
import os
from PIL import Image

from sfm_tools.feature_extract_match.model.superpoint import SuperPoint
from sfm_tools.feature_extract_match.model.superglue import SuperGlue
from adalam import adalam

class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self, conf, image_names):
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.paths = [i for i in image_names]
        self.max_edge_size = conf['resize_max']
    
    @staticmethod
    def read_image(impath):
        grayim = cv2.imread(impath)
        grayim = cv2.cvtColor(grayim, cv2.COLOR_BGR2GRAY)

        if grayim is None:
            raise Exception("Error reading image %s " % impath)
    
        return grayim
    
    def process_img(self, img):
        if self.max_edge_size is not None:
            ori_h, ori_w = img.shape[:2]
            new_h, new_w = None, None
            if ori_h > ori_w and ori_h > self.max_edge_size:
                new_h = self.max_edge_size
                new_w = int(new_h / ori_h * ori_w)
            elif ori_w > ori_h and ori_w > self.max_edge_size:
                new_w = self.max_edge_size
                new_h = int(new_w / ori_w * ori_h)

            if new_h is not None:
                img = cv2.resize(img, (new_w, new_h))
        img_gray = img
        inp = torch.from_numpy(img_gray / 255.).float()[None]
        return img_gray, inp
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        img0 = self.read_image(impath=path)
        size = img0.shape[:2][::-1] # (w, h)
        img_gray, inp = self.process_img(img0)

        data = {
            'name': str(path),
            'image': inp,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.paths)
    
def map_tensor(input_, func):
    if isinstance(input_, torch.Tensor):
        return func(input_)
    elif isinstance(input_, (str, bytes)):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    else:
        raise TypeError(
            f'input must be tensor, dict or list; found {type(input_)}')
    
def superpoint_model_init(config, device=torch.device('cpu')):
    conf = config.feature_confs.value['superpoint']
    superpoint_model = SuperPoint(conf['model']).eval().to(device)
    return superpoint_model

@torch.no_grad()
def extract_by_superpoint(img_fnames,
                          device=torch.device('cpu'),
                          seg_dir=None,
                          feature_dir='.featureout',
                          config=None):
    conf = config.feature_confs.value['superpoint']['preprocessing']
    model = superpoint_model_init(config, device=device)
    loader = ImageDataset(conf, img_fnames)
    loader = torch.utils.data.DataLoader(loader, num_workers=4)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
            h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
            h5py.File(f'{feature_dir}/score.h5', mode='w') as f_score, \
            h5py.File(f'{feature_dir}/imagesize.h5', mode='w') as f_size, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for data in tqdm(loader):
            img_fname = "/".join(data['name'][0].split('/')[-2:])
            pred = model(map_tensor(data, lambda x: x.to(device)))
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            pred['image_size'] = original_size = data['original_size'][0].numpy()
            if 'keypoints' in pred:
                size = np.array(data['image'].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
            
            lafs = pred['keypoints']
            scores = pred['scores']
            descs = pred['descriptors']
            desc_dim = descs.shape[-1]
            descs = descs.reshape(-1, desc_dim)
            descs = descs.T

            seg_img_path = os.path.join(seg_dir, img_fname.replace(".jpg", ".png"))
            pil_mask = Image.open(seg_img_path)
            mask_image = np.array(pil_mask, dtype="int64")

            new_lafs, new_scores, new_descs = [], [], []
            for idx in range(lafs.shape[0]):
                x, y = lafs[idx]
                if mask_image[int(y), int(x)] not in [55, 64, 19, 61]:
                    new_lafs.append(lafs[idx])
                    new_scores.append(scores[idx])
                    new_descs.append(descs[idx])
            
            new_lafs = np.array(new_lafs)
            new_scores = np.array(new_scores)
            new_descs = np.array(new_descs).T

            f_laf[img_fname] = new_lafs

            f_kp[img_fname] = new_lafs
            f_score[img_fname] = new_scores
            f_desc[img_fname] = new_descs
            f_size[img_fname] = np.array(data['original_size'][0])
    return 

def clear_partial_match_files(feature_dir: str) -> None:
    for fn in ("matches.h5", "matches_score.h5", "matches_adalam.h5"):
        path = os.path.join(feature_dir, fn)
        if os.path.isfile(path):
            os.remove(path)


def feature_extract_ready(feature_dir: str) -> bool:
    lafs_path = os.path.join(feature_dir, "lafs.h5")
    if not os.path.isfile(lafs_path):
        return False
    try:
        with h5py.File(lafs_path, "r") as f_laf:
            if len(f_laf.keys()) == 0:
                return False
            for cam in f_laf.keys():
                group = f_laf[cam]
                if not isinstance(group, h5py.Group) or len(group.keys()) == 0:
                    return False
                sample = next(iter(group.keys()))
                _ = group[sample].shape
        return True
    except (OSError, KeyError, StopIteration):
        return False


def read_feature_bundle(f_laf, f_desc, f_score, name: str):
    return {
        "keypoints": f_laf[name].__array__(),
        "descriptors": f_desc[name].__array__(),
        "scores": f_score[name].__array__(),
    }


def superglue_model_init(config, device=torch.device('cpu')):
    conf = config.matcher_confs.value['superpoint']
    superglue_model = SuperGlue(conf['model']).eval().to(device)
    return superglue_model

@torch.no_grad()
def match_by_superglue(img_fnames,
                       index_pairs,
                       feature_dir='.featureout',
                       device=torch.device('cpu'),
                       config=None,
                       debug=False):
    model = superglue_model_init(config, device=device)
    matched = set()
    clear_partial_match_files(feature_dir)

    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/score.h5', mode='r') as f_score, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/imagesize.h5', mode='r') as f_size, \
            h5py.File(f'{feature_dir}/matches.h5', mode='w') as match_file, \
            h5py.File(f'{feature_dir}/matches_score.h5', mode='w') as matches_score_file:

        skipped = 0
        for pair in tqdm(index_pairs, smoothing=.1):
            name0, name1 = img_fnames[pair[0]], img_fnames[pair[1]]
            name1 = name1.replace('\n', '')
            name0 = "/".join(name0.split('/')[-2:])
            name1 = "/".join(name1.split('/')[-2:])

            # avoid to recompute duplicates to save time
            if len({(name0, name1), (name1, name0)} & matched):
                continue
            if name0 not in f_laf or name1 not in f_laf:
                skipped += 1
                continue

            try:
                feat0 = read_feature_bundle(f_laf, f_desc, f_score, name0)
                feat1 = read_feature_bundle(f_laf, f_desc, f_score, name1)
            except (KeyError, OSError, ValueError) as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"Skip pair ({name0}, {name1}): {exc}")
                continue

            if feat0["keypoints"].size == 0 or feat1["keypoints"].size == 0:
                skipped += 1
                continue

            data = {
                'keypoints0': feat0["keypoints"],
                'descriptors0': feat0["descriptors"],
                'scores0': feat0["scores"],
                'keypoints1': feat1["keypoints"],
                'descriptors1': feat1["descriptors"],
                'scores1': feat1["scores"],
            }
            
            data = {
                k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()
            }

            data['image0'] = torch.empty((
                                            1, 
                                            1,
                                        ) + tuple(f_size[name0])[::-1])
            data['image1'] = torch.empty(( 
                                            1,
                                            1,
                                        ) + tuple(f_size[name1])[::-1])

            pred = model(data)
            matches = pred['matches0'][0].cpu().short().numpy()
            matching_score = pred['matching_scores0'][0].cpu().half().numpy()
            group = match_file.require_group(name0)
            group_score = matches_score_file.require_group(name0)
            group.create_dataset(name1, data=matches)
            group_score.create_dataset(name1, data=matching_score)
            matched |= {(name0, name1), (name1, name0)}
        if skipped:
            print(f"match_by_superglue skipped {skipped} pairs")
    return

def filter_match_by_adalam(img_fnames,
                           index_pairs,
                           feature_dir='.featureout',
                           device=torch.device('cpu'),
                           config=None):
    ADALAM_CONFIG = config.adalam_confs.value
    adalam_filter = adalam.AdalamFilter(ADALAM_CONFIG)

    matches_path = f'{feature_dir}/matches.h5'
    if not os.path.isfile(matches_path):
        raise FileNotFoundError(f"missing matches file: {matches_path}")
    adalam_path = f'{feature_dir}/matches_adalam.h5'
    if os.path.isfile(adalam_path):
        os.remove(adalam_path)

    keypoints_file = h5py.File(f'{feature_dir}/keypoints.h5', 'r')
    matches_file = h5py.File(f'{feature_dir}/matches.h5', 'r+')
    matches_score_file = h5py.File(f'{feature_dir}/matches_score.h5', 'r+')
    adalam_match_file = h5py.File(f'{feature_dir}/matches_adalam.h5', 'w')

    matched = set()
    for pair in tqdm(index_pairs, smoothing=.1):
        name0, name1 = img_fnames[pair[0]], img_fnames[pair[1]]
        name1 = name1.replace('\n', '')
        cam0, name0 = name0.split('/')[-2:]
        cam1, name1 = name1.split('/')[-2:]

        if len({(cam0+'/'+name0, cam1+'/'+name1), (cam1+'/'+name1, cam0+'/'+name0)} & matched):
            continue

        pair0 = cam0+"/"+name0+"/"+cam1+"/"+name1
        pair1 = cam1+"/"+name1+"/"+cam0+"/"+name0
        if pair0 in matches_file:
            pair = pair0
            keypoints0 = keypoints_file[cam0+'/'+name0].__array__()
            keypoints1 = keypoints_file[cam1+'/'+name1].__array__()
        elif pair1 in matches_file:
            pair = pair1
            keypoints0 = keypoints_file[cam1+'/'+name1].__array__()
            keypoints1 = keypoints_file[cam0+'/'+name0].__array__()
        else:
            continue
        
        keypoints0 = torch.tensor(keypoints0, device=device)
        keypoints1 = torch.tensor(keypoints1, device=device)

        matches = torch.tensor(matches_file[pair].__array__(),
                                device=device,
                                dtype=torch.long)
        matching_scores = torch.tensor(matches_score_file[pair].__array__(),
                                        device=device)
        
        index = torch.where(matches != -1)[0]

        if len(index) == 0:
            matches_file.__delitem__(pair)
            continue
        
        keypoints0_valid = keypoints0[index]
        matches_valid = matches[index]
        matching_scores_valid = matching_scores[index]

        filtered_matches = adalam_filter.filter_matches(
            k1=keypoints0_valid,
            k2=keypoints1,
            putative_matches=matches_valid,
            scores=matching_scores_valid)
        
        final_matches = -torch.ones(matches.shape, device=device, dtype=torch.long)
        final_match_index = filtered_matches[:, 0]
        final_match_index = index[final_match_index]
        final_matches[final_match_index] = matches[final_match_index]
        
        group0 = "/".join(pair.split('/')[:2])
        group1 = "/".join(pair.split('/')[2:])
        final_matches = final_matches.cpu().numpy()

        adalam_matches = np.concatenate((np.arange(final_matches.shape[0])[..., None], final_matches[..., None]), axis=1)
        adalam_matches = adalam_matches[adalam_matches[:, 1] >= 0]
        grp = adalam_match_file.require_group(group0)
        grp.create_dataset(group1, data=adalam_matches)

        matched |= {(cam0+'/'+name0, cam1+'/'+name1), (cam1+'/'+name1, cam0+'/'+name0)}

    adalam_match_file.close()
    print("finished adalam filter matches")
    return 