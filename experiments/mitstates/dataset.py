import os, sys, random, zipfile, io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import IMAGE_SIZE, BATCH_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ZIP_PATH = os.path.join(DATA_DIR, 'mitstates.zip')
EXTRACT_DIR = os.path.join(DATA_DIR, 'release_dataset')


def ensure_downloaded():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f'MIT-States zip not found at {ZIP_PATH}. '
            'Please download manually from: '
            'http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip'
        )
    return ZIP_PATH


def build_pair_index(zip_path):
    pairs = {}
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if '__MACOSX' in name:
                continue
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parts = name.split('/')
            if len(parts) < 4:
                continue
            obj_dir = parts[2]
            if not obj_dir.startswith('adj '):
                continue
            obj = obj_dir[4:]
            fname = parts[-1]
            if not fname or fname.startswith('._'):
                continue
            if obj not in pairs:
                pairs[obj] = []
            pairs[obj].append((name, fname))
    return pairs


def split_objects(pairs, test_ratio=0.2, seed=42):
    objects = sorted(pairs.keys())
    rng = random.Random(seed)
    rng.shuffle(objects)
    n_test = max(1, int(len(objects) * test_ratio))
    test_objs = set(objects[:n_test])
    train_objs = set(objects[n_test:])

    train_samples = []
    test_samples = []
    for obj in train_objs:
        for zip_name, _ in pairs[obj]:
            train_samples.append((zip_name, obj, ('various', obj)))
    for obj in test_objs:
        for zip_name, _ in pairs[obj]:
            test_samples.append((zip_name, obj, ('various', obj)))

    print(f'Train objects: {len(train_objs)}, Test objects: {len(test_objs)}')
    print(f'Train images: {len(train_samples)}, Test images: {len(test_samples)}')
    return train_samples, test_samples


class MITStatesZipDataset(Dataset):
    def __init__(self, zip_path, samples, transform=None):
        self.zip_path = zip_path
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.attrs = sorted(set(s[2][0] for s in self.samples))
        self.objs = sorted(set(s[1] for s in self.samples))
        self.attr_to_idx = {a: i for i, a in enumerate(self.attrs)}
        self.obj_to_idx = {o: i for i, o in enumerate(self.objs)}
        self._zf = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zip_name, obj, (attr, _) = self.samples[idx]
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, 'r')
        data = self._zf.read(zip_name)
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img_t = self.transform(img)
        return img_t, self.attr_to_idx[attr], self.obj_to_idx[obj], attr, obj


def get_dataloaders(batch_size=BATCH_SIZE):
    zip_path = ensure_downloaded()
    pairs = build_pair_index(zip_path)
    train_samples, test_samples = split_objects(pairs)

    train_ds = MITStatesZipDataset(zip_path, train_samples)
    test_ds = MITStatesZipDataset(zip_path, test_samples)

    num_attrs = len(train_ds.attrs)
    num_objs = len(train_ds.objs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, train_ds, test_ds, num_attrs, num_objs
