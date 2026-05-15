import os, sys, random, numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.dsprites.config import (
    IMAGE_SIZE, NUM_SHAPES, NUM_COLORS, COLORS,
    SHAPES, TRAIN_COLORS, TEST_COLORS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def draw_shape(draw, cx, cy, size, shape, color_tuple):
    r, g, b = [int(c * 255) for c in color_tuple]
    fill = (r, g, b)
    hs = size // 2
    if shape == 'square':
        draw.rectangle([cx - hs, cy - hs, cx + hs, cy + hs], fill=fill)
    elif shape == 'ellipse':
        draw.ellipse([cx - hs, cy - hs // 2, cx + hs, cy + hs // 2], fill=fill)
    elif shape == 'heart':
        pts = []
        for t in np.linspace(0, 2 * np.pi, 40):
            x = 16 * np.sin(t) ** 3
            y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
            pts.append((cx + int(x * hs / 16), cy - int(y * hs / 16)))
        draw.polygon(pts, fill=fill)


def generate_split(split_dir, color_list, samples_per_combo=500):
    os.makedirs(split_dir, exist_ok=True)
    for shape_idx, shape_name in enumerate(SHAPES):
        for color_idx, color_name in enumerate(COLORS):
            if color_name not in color_list:
                continue
            color_tuple = COLORS[color_name]
            combo_dir = os.path.join(split_dir, f'{shape_name}_{color_name}')
            os.makedirs(combo_dir, exist_ok=True)
            for i in range(samples_per_combo):
                img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                size = random.randint(16, 42)
                cx = random.randint(size // 2 + 4, IMAGE_SIZE - size // 2 - 4)
                cy = random.randint(size // 2 + 4, IMAGE_SIZE - size // 2 - 4)
                draw_shape(draw, cx, cy, size, shape_name, color_tuple)
                img.save(os.path.join(combo_dir, f'{i:04d}.png'))


def main():
    print(f'Generating colored dSprites: {NUM_SHAPES} shapes x {len(COLORS)} colors')
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')
    print(f'Train colors: {TRAIN_COLORS}')
    print(f'Test colors:  {TEST_COLORS}')
    generate_split(train_dir, TRAIN_COLORS)
    generate_split(test_dir, TEST_COLORS)
    train_count = 0
    test_count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        train_count += sum(1 for f in files if 'train' in root and f.endswith('.png'))
        test_count += sum(1 for f in files if 'test' in root and f.endswith('.png'))
    print(f'Train images: {train_count}')
    print(f'Test images:  {test_count}')
    print('Done.')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()
