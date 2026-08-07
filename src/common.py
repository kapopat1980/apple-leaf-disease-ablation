import os, json, random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 64

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


def physically_inspired_augment(img, strength=1.0):
    """CLAHE contrast enhancement + illumination simulation + sensor noise,
    inspired by Fidan2026's physically-inspired augmentation strategy."""
    img = img.copy()
    # CLAHE on L channel (texture / contrast enhancement)
    if random.random() < 0.7:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Illumination simulation (brightness / gamma jitter)
    if random.random() < 0.7:
        gamma = random.uniform(1 - 0.35 * strength, 1 + 0.35 * strength)
        inv = 1.0 / max(gamma, 1e-3)
        table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
        img = cv2.LUT(img, table)
    # Sensor noise simulation
    if random.random() < 0.5:
        sigma = 6.0 * strength
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Standard geometric augmentation
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.3:
        img = cv2.flip(img, 0)
    if random.random() < 0.5:
        angle = random.uniform(-25, 25)
        M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT)
    return img


class AppleLeafDataset(Dataset):
    def __init__(self, split, augment=False, minority_boost=None):
        """
        split: 'train' | 'val' | 'test'
        augment: whether to apply physically-inspired augmentation (train only)
        minority_boost: dict {class_idx: multiplier} to oversample minority
                         classes via repeated (re-augmented) indices, rather
                         than naive duplication of raw pixels.
        """
        self.items = list(SPLITS[split])
        self.augment = augment
        if minority_boost:
            boosted = []
            for path, label in self.items:
                mult = minority_boost.get(label, 1)
                for _ in range(mult):
                    boosted.append((path, label))
            self.items = boosted
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        if path not in self._cache:
            self._cache[path] = load_image(path)
        img = self._cache[path]
        if self.augment:
            img = physically_inspired_augment(img)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float(), label


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, use_attention=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.attn = SEBlock(out_c) if use_attention else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return self.pool(x)


class BaselineCNN(nn.Module):
    """Plain lightweight CNN, no attention -- used as the control model."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.b1 = ConvBlock(3, 32, use_attention=False)
        self.b2 = ConvBlock(32, 64, use_attention=False)
        self.b3 = ConvBlock(64, 128, use_attention=False)
        self.b4 = ConvBlock(128, 256, use_attention=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class AttentiveFocalNet(nn.Module):
    """Proposed model: lightweight CNN backbone + SE channel attention.
    Paired externally with focal loss + class weighting + physically
    inspired augmentation, per the design recommendations."""
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of per-class weights or None

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss
        return loss.mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters())
