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
from superglue import LightGlue
from AdaLAM.adalam import adalam

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
    detector = SuperPoint(conf['model']).eval().to(device)
    state_dict = torch.load(conf['model']['weights'], map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict.pop(k)
        if k.startswith('superpoint.'):
            state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
    detector.load_state_dict(state_dict)
    return detector

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
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
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
            descs = pred['descriptors']

            mask_img_path = os.path.join(seg_dir, img_fname.replace(".jpg", ".png"))
            pil_mask = Image.open(mask_img_path)
            mask_image = np.array(pil_mask, dtype="int64")

            new_lafs, new_descs = [], []
            for idx in range(lafs.shape[0]):
                x, y = lafs[idx]
                if mask_image[int(y), int(x)] not in [55, 64, 19, 61]:
                    new_lafs.append(lafs[idx])
                    new_descs.append(descs[idx])
            new_lafs = np.array(new_lafs)
            new_descs = np.array(new_descs)

            f_kp[img_fname] = new_lafs
            f_desc[img_fname] = new_descs
            f_size[img_fname] = np.array(data['original_size'][0])

    return

def superglue_model_init(config, device=torch.device('cpu')):
    conf = config.matcher_confs.value['superpoint']
    superglue_model = LightGlue(conf['model']).eval().to(device)
    state_dict = torch.load(conf['model']['weights'], map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('superpoint.'):
            state_dict.pop(k)
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
    superglue_model.load_state_dict(state_dict)
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

    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/imagesize.h5', mode='r') as f_size, \
            h5py.File(f'{feature_dir}/matches.h5', mode='w') as match_file, \
            h5py.File(f'{feature_dir}/matches_score.h5', mode='w') as matches_score_file:

        for pair in tqdm(index_pairs, smoothing=.1):
            name0, name1 = img_fnames[pair[0]], img_fnames[pair[1]]
            name1 = name1.replace('\n', '')
            name0 = "/".join(name0.split('/')[-2:])
            name1 = "/".join(name1.split('/')[-2:])

            # Avoid to recompute duplicates to save time
            if len({(name0, name1), (name1, name0)} & matched):
                continue
            if name0 not in f_laf or name1 not in f_laf:
                continue

            data = {'keypoints0': f_laf[name0].__array__(), 'descriptors0': f_desc[name0].__array__(), 
                    'keypoints1': f_laf[name1].__array__(), 'descriptors1': f_desc[name1].__array__(),
                    'image_size0': f_size[name0].__array__(), 'image_size1': f_size[name1].__array__()}

            data = {
                k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()
            }

            pred = model(data)
            matches = pred['matches0'][0].cpu().short().numpy()
            matching_score = pred['scores'][0].cpu().half().numpy()
            group = match_file.require_group(name0)
            group_score = matches_score_file.require_group(name0)
            group.create_dataset(name1, data=matches)
            group_score.create_dataset(name1, data=matching_score)
            matched |= {(name0, name1), (name1, name0)}
    return

def filter_match_by_adalam(img_fnames,
                           index_pairs,
                           feature_dir='.featureout',
                           device=torch.device('cpu'),
                           config=None):
    ADALAM_CONFIG = config.adalam_confs.value
    adalam_filter = adalam.AdalamFilter(ADALAM_CONFIG)

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
        matching_scores_valid = matching_scores

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