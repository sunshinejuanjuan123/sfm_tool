import torch
import cv2
from types import SimpleNamespace
import numpy as np
import h5py
from tqdm import tqdm
import collections.abc as collections
import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from superpoint import SuperPoint
from superglue import SuperGlue


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

def superglue_model_init(config, device=torch.device('cpu')):
    conf = config.matcher_confs.value['superpoint']
    superglue_model = SuperGlue(conf['model']).eval().to(device)
    return superglue_model

@torch.no_grad()
def match_by_superglue(img_fnames,
                       index_pairs,
                       feature_dir='.featureout',
                       device=torch.device('cpu'),
                       min_matches=15,
                       config=None,
                       debug=False):
    model = superglue_model_init(config, device=device)
    ransac_confs = config.ransac_confs.value
    matched = set()

    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/score.h5', mode='r') as f_score, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/imagesize.h5', mode='r') as f_size, \
            h5py.File(f'{feature_dir}/matches.h5', mode='w') as match_file, \
            h5py.File(f'{feature_dir}/matches_ransac.h5', mode='w') as matches_ransac_file:
        
        for pair in tqdm(index_pairs, smoothing=.1):
            name0, name1 = img_fnames[pair[0]], img_fnames[pair[1]]
            name1 = name1.replace('\n', '')
            name0 = "/".join(name0.split('/')[-2:])
            name1 = "/".join(name1.split('/')[-2:])

            # avoid to recompute duplicates to save time
            if len({(name0, name1), (name1, name0)} & matched):
                continue
            if name0 not in f_laf or name1 not in f_laf:
                continue

            data = {'keypoints0': f_laf[name0].__array__(), 'descriptors0': f_desc[name0].__array__(), 'scores0': f_score[name0].__array__(), 
                    'keypoints1': f_laf[name1].__array__(), 'descriptors1': f_desc[name1].__array__(), 'scores1': f_score[name1].__array__()}
            
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
            matches = np.concatenate((np.arange(matches.shape[0])[..., None], matches[..., None]), axis=1)

            matches = matches[matches[:, 1] >= 0]
            n_matches = matches.shape[0]

            if ransac_confs['open']:
                matches_ransac = matches.copy()
                matches_ransac = matches_ransac[matches_ransac[:, 1] >= 0]
                mkpts0 = data['keypoints0'].detach().cpu().numpy()[0, matches_ransac[:, 0]]
                mkpts1 = data['keypoints1'].detach().cpu().numpy()[0, matches_ransac[:, 1]]
                if mkpts0.shape[0] > 15 and mkpts1.shape[0] > 15:
                    try:
                        F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, ransac_confs['method'],
                                                            ransac_confs['ransacReprojThreshold'],
                                                            ransac_confs['confidence'],
                                                            ransac_confs['maxIters'])
                        matches_ransac = matches_ransac[inliers[:, 0] == 1]
                        n_matches_ransac = matches_ransac.shape[0]
                    except:
                        n_matches = 0
                else:
                    matches_ransac = matches
                    n_matches_ransac = matches.shape[0]
            else:
                matches_ransac = matches
                n_matches_ransac = matches.shape[0]
            
            if n_matches >= min_matches:
                group = match_file.require_group(name0)
                group_ransac = matches_ransac_file.require_group(name0)
                group.create_dataset(name1, data=matches)
                group_ransac.create_dataset(name1, data=matches_ransac)
            
            matched |= {(name0, name1), (name1, name0)}
    return 
        