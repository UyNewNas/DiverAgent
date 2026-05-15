import os, sys
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.dsprites.config import IMAGE_SIZE, K_OUTPUTS, DEVICE, BATCH_SIZE
from experiments.dsprites.dataset import get_dataloaders
from experiments.dsprites.backbone import DSpritesCBDP

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    _, _, train_eval, test_eval, _, _ = get_dataloaders(BATCH_SIZE)
    model = DSpritesCBDP().to(DEVICE)
    ckpt = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    for split_name, loader in [('train', train_eval), ('test', test_eval)]:
        os.makedirs(os.path.join(RESULT_DIR, split_name), exist_ok=True)
        for combo_idx, (x, shape_idx, color_idx, shape_name, color_name) in enumerate(loader):
            x = x.to(DEVICE)
            with torch.no_grad():
                recon, obj_e, attr_e, _, _, _, _ = model.backbone(
                    x, shape_idx.to(DEVICE), color_idx.to(DEVICE)
                )
                outputs_k, _, _ = model.forward_divergent(obj_e, attr_e)

            for i in range(min(4, x.size(0))):
                img_row = [x[i].cpu()]
                img_row.append(recon[i].cpu())
                for k in range(K_OUTPUTS):
                    img_row.append(outputs_k[i, k].cpu())
                grid = torch.stack(img_row)
                grid_img = vutils.make_grid(grid, nrow=len(img_row), padding=2, normalize=True)
                fname = f'{shape_name[i]}_{color_name[i]}_sample{i}.png'
                vutils.save_image(grid_img, os.path.join(RESULT_DIR, split_name, fname))
            break

    print('Visualization saved.')


if __name__ == '__main__':
    main()
