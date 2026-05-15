import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import (
    IMAGE_SIZE, BATCH_SIZE, SHAPES, COLORS, TRAIN_COLORS, TEST_COLORS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class ColoredDSprites(Dataset):
    def __init__(self, split='train', transform=None):
        self.root = os.path.join(DATA_DIR, split)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.samples = []
        self.color_names = TRAIN_COLORS if split == 'train' else TEST_COLORS
        for shape_idx, shape in enumerate(SHAPES):
            for color_idx, color_name in enumerate(COLORS):
                if color_name not in self.color_names:
                    continue
                combo_dir = os.path.join(self.root, f'{shape}_{color_name}')
                if not os.path.isdir(combo_dir):
                    continue
                for fname in os.listdir(combo_dir):
                    if fname.endswith('.png'):
                        self.samples.append({
                            'path': os.path.join(combo_dir, fname),
                            'shape_idx': shape_idx,
                            'color_idx': color_idx,
                            'shape_name': shape,
                            'color_name': color_name,
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        img_t = self.transform(img)
        return img_t, s['shape_idx'], s['color_idx'], s['shape_name'], s['color_name']


class ProbeEvalDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.root = os.path.join(DATA_DIR, split)
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.combos = []
        self.color_names = TRAIN_COLORS if split == 'train' else TEST_COLORS
        for shape_idx, shape in enumerate(SHAPES):
            for color_idx, color_name in enumerate(COLORS):
                if color_name not in self.color_names:
                    continue
                combo_dir = os.path.join(self.root, f'{shape}_{color_name}')
                if os.path.isdir(combo_dir):
                    files = sorted([f for f in os.listdir(combo_dir) if f.endswith('.png')])
                    self.combos.append({
                        'shape_idx': shape_idx,
                        'color_idx': color_idx,
                        'shape_name': shape,
                        'color_name': color_name,
                        'dir': combo_dir,
                        'files': files,
                    })

    def __len__(self):
        return len(self.combos)

    def __getitem__(self, idx):
        c = self.combos[idx]
        idx0 = random.randint(0, len(c['files']) - 1)
        img = Image.open(os.path.join(c['dir'], c['files'][idx0])).convert('RGB')
        return self.transform(img), c['shape_idx'], c['color_idx'], c['shape_name'], c['color_name']


def get_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_ds = ColoredDSprites('train', transform)
    test_ds = ColoredDSprites('test', transform)
    train_eval_ds = ProbeEvalDataset('train', transform)
    test_eval_ds = ProbeEvalDataset('test', transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    test_eval_loader = DataLoader(test_eval_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, train_eval_loader, test_eval_loader, train_ds, test_ds
