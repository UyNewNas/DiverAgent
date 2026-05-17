import os, sys
import torch
import torchvision.utils as vutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.mitstates.config import IMAGE_SIZE, K_OUTPUTS, DEVICE, BATCH_SIZE
from experiments.mitstates.dataset import get_dataloaders
from experiments.mitstates.backbone import MITStatesCBDP

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    train_loader, test_loader, _, _, num_attrs, num_objs = get_dataloaders(BATCH_SIZE)
    model = MITStatesCBDP(num_objects=num_objs, num_attrs=num_attrs).to(DEVICE)
    ckpt = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    for split_name, loader in [('train', train_loader), ('test', test_loader)]:
        out_dir = os.path.join(RESULT_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)

        seen = set()
        for x, attr_idx, obj_idx, attr_name, obj_name in loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                _, obj_e, attr_e, _, _, _, _, _ = model.backbone(
                    x, attr_idx.to(DEVICE), obj_idx.to(DEVICE)
                )
                recon = model.backbone.decode(obj_e)
                outputs_k, _, _ = model.forward_divergent(obj_e, attr_e)

            for i in range(min(2, x.size(0))):
                pair_key = f'{attr_name[i]}_{obj_name[i]}'
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                img_row = [x[i].cpu(), recon[i].cpu()]
                for k in range(K_OUTPUTS):
                    img_row.append(outputs_k[i, k].cpu())
                grid = torch.stack(img_row)
                grid_img = vutils.make_grid(grid, nrow=len(img_row), padding=2, normalize=True)
                fname = f'{pair_key}.png'
                vutils.save_image(grid_img, os.path.join(out_dir, fname))
            if len(seen) >= 8:
                break

    print('Visualization saved.')


if __name__ == '__main__':
    main()
