from pathlib import Path
import torch
from torch import nn
import numpy as np

def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum supression to remove nearby points"""
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
    
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """Removes keypoints too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) * (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

# def sample_descriptors(keypoints, descriptors, s: int = 8):
#     """Interpolate descriptors at keypoint locations"""
#     b, c, h, w = descriptors.shape
#     keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
#     keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
#     descriptors = torch.nn.functional.grid_sample(
#         descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=False)
#     descriptors = torch.nn.functional.normalize(
#         descriptors.reshape(b, c, -1), p=2, dim=1)
#     return descriptors

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

class SuperPoint(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 8,
        'keypoint_threshold': 0.001,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1a = torch.nn.BatchNorm2d(c1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.bn1b = torch.nn.BatchNorm2d(c1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.bn2a = torch.nn.BatchNorm2d(c2)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.bn2b = torch.nn.BatchNorm2d(c2)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.bn3a = torch.nn.BatchNorm2d(c3)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.bn3b = torch.nn.BatchNorm2d(c3)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.bn4a = torch.nn.BatchNorm2d(c4)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn4b = torch.nn.BatchNorm2d(c4)

        # Detector Head
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = torch.nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.bnPb = torch.nn.BatchNorm2d(65)

        # Descriptor Head
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = torch.nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, self.config['descriptor_dim'], kernel_size=1, stride=1, padding=0)
        self.bnDb = torch.nn.BatchNorm2d(self.config['descriptor_dim'])

        path = self.config['weight']
        self.load_state_dict(torch.load(str(path))['model_state_dict'])
        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max keypoints\" must be positive or \"-1\"')
        
        print('Load SuperPoint model')
    
    def forward(self, data):

        x = self.bn1a(self.relu(self.conv1a(data['image'])))
        x = self.bn1b(self.relu(self.conv1b(x)))
        x = self.pool(x)
        x = self.bn2a(self.relu(self.conv2a(x)))
        x = self.bn2b(self.relu(self.conv2b(x)))
        x = self.pool(x)
        x = self.bn3a(self.relu(self.conv3a(x)))
        x = self.bn3b(self.relu(self.conv3b(x)))
        x = self.pool(x)
        x = self.bn4a(self.relu(self.conv4a(x)))
        x = self.bn4b(self.relu(self.conv4b(x)))

        # Detector Head
        cPa = self.bnPa(self.relu(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))

        # Descriptor Head
        cDa = self.bnDa(self.relu(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))

        scores, descriptors = semi, desc
        
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                    top_k_keypoints(k, s, self.config['max_keypoints'])
                    for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
