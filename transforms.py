import cv2
import math
import torch
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        h, w, c = image.shape
        if random.random() < self.p:
            image = np.fliplr(image)
            for t in target:
                bx, by, bw, bh = t['bbox']
                bx = w - bx - bw
                t['bbox'] = [bx, by, bw, bh]
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        h, w, c = image.shape
        ws, hs = self.size[0] / w, self.size[1] / h
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LANCZOS4)
        for t in target:
            bx, by, bw, bh = t['bbox']
            t['bbox'] = [bx*ws, by*hs, bw*ws, bh*hs]
        return image, target


class ToTensor:
    def __init__(self):
        return

    def __call__(self, image, target):
        label = label_encode(image, target)
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).reshape(-1, 1, 1)
        self.std = torch.FloatTensor(std).reshape(-1, 1, 1)

    def __call__(self, image, target):
        image = (image - self.mean) / self.std
        return image, target


def label_encode(image, target, num_classes=91, R=4):
    H, W, C = image.shape
    h, w = H // R, W // R

    gt_k = np.zeros((num_classes, h, w), dtype=np.float32)
    gt_o = np.zeros((2, h, w), dtype=np.float32)
    gt_s = np.zeros((2, h, w), dtype=np.float32)

    for t in target:
        c = t['category_id']
        bx, by, bw, bh = t['bbox']
        ex = (bx + bw / 2) / R
        ey = (by + bh / 2) / R
        ew = bw / R
        eh = bh / R
        exi = int(ex)
        eyi = int(ey)
        exo = ex - exi
        eyo = ey - eyi

        # TODO: check radius
        radius = int(math.sqrt(ew * eh)) // 4
        if radius < 1:
            radius = 1
        diameter = radius * 2 + 1

        kernel1d = cv2.getGaussianKernel(diameter, diameter / 6)
        kernel1d /= max(kernel1d)
        kernel2d = np.outer(kernel1d, kernel1d.T)

        kx0 = exi - radius
        kx1 = exi + radius + 1
        ky0 = eyi - radius
        ky1 = eyi + radius + 1

        if kx0 < 0:
            kx0_ = -kx0
            kx0 = 0
        else:
            kx0_ = 0
        if kx1 > w:
            kx1_ = diameter - kx1 + w
            kx1 = w
        else:
            kx1_ = diameter
        if ky0 < 0:
            ky0_ = -ky0
            ky0 = 0
        else:
            ky0_ = 0
        if ky1 > h:
            ky1_ = diameter - ky1 + h
            ky1 = h
        else:
            ky1_ = diameter

        gt_k[c, ky0:ky1, kx0:kx1] = np.maximum(
            gt_k[c, ky0:ky1, kx0:kx1], kernel2d[ky0_:ky1_, kx0_:kx1_])

        gt_k[c, eyi, exi] = 1
        gt_o[0, eyi, exi] = exo
        gt_o[1, eyi, exi] = eyo
        gt_s[0, eyi, exi] = ew
        gt_s[1, eyi, exi] = eh

        gt = np.concatenate([gt_o, gt_s, gt_k], 0)

    return gt


class KeyPointExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        y = self.pool(x)
        x = x - 1.0 * (x < y)
        return x

def label_decode(gt, R=4):
    gt_k = gt[4:]
    gt_o = gt[0:2]
    gt_s = gt[2:4]

    kpe = KeyPointExtractor()(torch.from_numpy(gt_k).unsqueeze(0)).numpy().squeeze()

    points = np.where(kpe > 0.5)
    decoded = []
    for c, x, y in zip(points[0], points[2], points[1]):
        xf = (x + gt_o[0, y, x])
        yf = (y + gt_o[1, y, x])

        wf = gt_s[0, y, x]
        hf = gt_s[1, y, x]

        bx = (xf - wf / 2) * R
        by = (yf - hf / 2) * R
        bw = wf * R
        bh = hf * R

        bbox = [bx, by, bw, bh]

        decoded.append({
            'bbox': bbox,
            'category_id': c,
        })
    
    return decoded
    