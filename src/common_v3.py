import json, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from common import SEBlock, ConvBlock, count_params  # reuse building blocks

IMG_SIZE = 96

with open("splits.json") as f:
    _data = json.load(f)
CLASSES = _data["classes"]
SPLITS = _data["splits"]
NUM_CLASSES = len(CLASSES)


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def light_augment(img):
    """Lighter augmentation than v1: CLAHE + mild geometric only.
    Drops the strong sensor-noise / wide-rotation / illumination jitter
    that likely hurt the v1 'proposed' model on this small, clean,
    lab-condition dataset."""
    img = img.copy()
    if random.random() < 0.5:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.2:
        img = cv2.flip(img, 0)
    if random.random() < 0.4:
        angle = random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.3:
        # mild brightness jitter only (no heavy gamma / noise)
        beta = random.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return img


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


class AppleLeafDatasetV3(Dataset):
    def __init__(self, split, augment=False):
        self.items = list(SPLITS[split])
        self.augment = augment
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        if path not in self._cache:
            self._cache[path] = load_image(path)
        img = self._cache[path]
        if self.augment:
            img = light_augment(img)
        return to_tensor(img), label


class AttentiveNetV3(nn.Module):
    """Same SE-attention backbone as v1's proposed model, just run at
    higher resolution with a lighter training recipe."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.b1 = ConvBlock(3, 32, use_attention=True)
        self.b2 = ConvBlock(32, 64, use_attention=True)
        self.b3 = ConvBlock(64, 128, use_attention=True)
        self.b4 = ConvBlock(128, 256, use_attention=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class LabelSmoothCE(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, target):
        n_classes = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(true_dist * logp).sum(dim=1)
        if self.weight is not None:
            w = self.weight.gather(0, target)
            loss = loss * w
        return loss.mean()
