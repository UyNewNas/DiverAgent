import os, sys, random, zipfile, urllib.request
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import IMAGE_SIZE, BATCH_SIZE, DATA_URL

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'states_and_transformations')

TRAIN_PAIRS = [
    ('ripe', 'tomato'), ('rotten', 'tomato'), ('fresh', 'tomato'),
    ('ripe', 'apple'), ('rotten', 'apple'),
    ('fresh', 'apple'), ('ripe', 'banana'), ('rotten', 'banana'),
    ('fresh', 'banana'), ('ripe', 'orange'), ('fresh', 'orange'),
    ('ripe', 'lemon'), ('fresh', 'lemon'), ('ripe', 'strawberry'),
    ('fresh', 'strawberry'), ('whole', 'bread'), ('sliced', 'bread'),
    ('toasted', 'bread'), ('dirty', 'car'), ('clean', 'car'),
    ('old', 'car'), ('new', 'car'), ('dirty', 'bicycle'), ('clean', 'bicycle'),
    ('old', 'bicycle'), ('new', 'bicycle'), ('dirty', 'motorcycle'),
    ('clean', 'motorcycle'), ('dirty', 'bus'), ('clean', 'bus'),
    ('closed', 'door'), ('open', 'door'), ('closed', 'window'),
    ('open', 'window'), ('closed', 'drawer'), ('open', 'drawer'),
    ('closed', 'cabinet'), ('open', 'cabinet'), ('closed', 'box'),
    ('open', 'box'), ('on', 'lamp'), ('off', 'lamp'), ('on', 'light'),
    ('off', 'light'), ('on', 'computer'), ('off', 'computer'),
    ('full', 'bottle'), ('empty', 'bottle'), ('full', 'cup'),
    ('empty', 'cup'), ('full', 'glass'), ('empty', 'glass'),
    ('cooked', 'meat'), ('raw', 'meat'), ('cooked', 'chicken'),
    ('raw', 'chicken'), ('cooked', 'fish'), ('raw', 'fish'),
    ('cooked', 'egg'), ('raw', 'egg'), ('lit', 'candle'),
    ('unlit', 'candle'), ('lit', 'cigarette'), ('unlit', 'cigarette'),
    ('inflated', 'balloon'), ('deflated', 'balloon'),
    ('peeled', 'orange'), ('unpeeled', 'orange'), ('peeled', 'banana'),
    ('unpeeled', 'banana'), ('peeled', 'apple'), ('whole', 'apple'),
]

TEST_PAIRS = [
    ('rotten', 'orange'), ('rotten', 'lemon'), ('rotten', 'strawberry'),
    ('old', 'motorcycle'), ('old', 'bus'), ('dirty', 'door'),
    ('dirty', 'window'), ('dirty', 'drawer'), ('toasted', 'bread_slice'),
    ('raw', 'egg'), ('cooked', 'egg'), ('cooked', 'fish'),
    ('unlit', 'candle'), ('deflated', 'balloon'),
    ('peeled', 'lemon'), ('unpeeled', 'lemon'),
]


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, 'states_and_transformations.zip')
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(zip_path):
            print(f'Downloading MIT-States from {DATA_URL}...')
            urllib.request.urlretrieve(DATA_URL, zip_path)
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR)
    return EXTRACT_DIR


def collect_pairs_samples(root_dir, pair_list):
    samples = []
    for attr, obj in pair_list:
        pair_name = f'{attr} {obj}'
        pair_dir = os.path.join(root_dir, pair_name)
        if not os.path.isdir(pair_dir):
            continue
        for fname in os.listdir(pair_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    'path': os.path.join(pair_dir, fname),
                    'attr_name': attr,
                    'obj_name': obj,
                    'pair_name': pair_name,
                })
    return samples


class MITStatesDataset(Dataset):
    def __init__(self, root_dir, pair_list, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.samples = collect_pairs_samples(root_dir, pair_list)
        self.attrs = sorted(set(s['attr_name'] for s in self.samples))
        self.objs = sorted(set(s['obj_name'] for s in self.samples))
        self.attr_to_idx = {a: i for i, a in enumerate(self.attrs)}
        self.obj_to_idx = {o: i for i, o in enumerate(self.objs)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        img_t = self.transform(img)
        return (
            img_t,
            self.attr_to_idx[s['attr_name']],
            self.obj_to_idx[s['obj_name']],
            s['attr_name'],
            s['obj_name'],
        )


def get_dataloaders(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    root = EXTRACT_DIR
    if not os.path.exists(root):
        root = download_and_extract()

    train_ds = MITStatesDataset(root, TRAIN_PAIRS, transform)
    test_ds = MITStatesDataset(root, TEST_PAIRS, transform)

    num_attrs = len(train_ds.attrs)
    num_objs = len(train_ds.objs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, train_ds, test_ds, num_attrs, num_objs
