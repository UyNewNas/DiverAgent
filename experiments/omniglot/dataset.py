import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from .config import IMAGE_SIZE, K_SHOT


class OmniglotFewShot(Dataset):
    def __init__(self, root, background=True, k_shot=K_SHOT, transform=None):
        self.k_shot = k_shot
        full = datasets.Omniglot(root=root, background=background, download=True, transform=transform)
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(full):
            self.class_to_indices.setdefault(label, []).append(idx)
        self.full_dataset = full
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        cls = self.classes[idx]
        indices = self.class_to_indices[cls]
        selected = random.sample(indices, min(self.k_shot + 1, len(indices)))
        support_imgs = torch.stack([self.full_dataset[i][0] for i in selected[:self.k_shot]])
        target_img = self.full_dataset[selected[-1]][0]
        return support_imgs, target_img, cls


def get_omniglot_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    base_dataset = OmniglotFewShot(data_dir, background=True, k_shot=K_SHOT, transform=transform)
    novel_dataset = OmniglotFewShot(data_dir, background=False, k_shot=K_SHOT, transform=transform)

    train_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    novel_loader = DataLoader(novel_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, novel_loader, base_dataset, novel_dataset
