import os, sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.dsprites.config import (
    IMAGE_SIZE, BATCH_SIZE, BACKBONE_EPOCHS, BACKBONE_LR, DEVICE,
)
from experiments.dsprites.dataset import get_dataloaders
from experiments.dsprites.backbone import DSpritesBackbone

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def main():
    print(f'Device: {DEVICE}')
    train_loader, _, _, _, _, _ = get_dataloaders(BATCH_SIZE)

    model = DSpritesBackbone().to(DEVICE)
    print(f'Backbone params: {model.count_parameters():,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=BACKBONE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BACKBONE_EPOCHS)

    best_loss = float('inf')
    for epoch in range(BACKBONE_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{BACKBONE_EPOCHS}')
        for x, shape_idx, color_idx, _, _ in pbar:
            x = x.to(DEVICE)
            shape_idx = shape_idx.to(DEVICE)
            color_idx = color_idx.to(DEVICE)

            recon, _, _, _, _, obj_l, attr_l, ce_s, ce_c = model(x, shape_idx, color_idx)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + 0.1 * obj_l + 0.1 * attr_l + 0.5 * ce_s + 0.5 * ce_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'backbone_best.pt'))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'backbone_final.pt'))
    print(f'Training complete. Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
