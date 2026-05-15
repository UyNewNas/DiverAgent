import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.omniglot.config import (
    IMAGE_SIZE, LATENT_DIM, NOISE_DIM, K_SHOT, BATCH_SIZE,
    BACKBONE_EPOCHS, BACKBONE_LR, DEVICE,
)
from experiments.omniglot.dataset import get_omniglot_loaders
from experiments.omniglot.backbone import OmniglotBackbone

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def main():
    print(f'Device: {DEVICE}')
    print(f'Image size: {IMAGE_SIZE}, K-shot: {K_SHOT}, Latent: {LATENT_DIM}')

    train_loader, novel_loader, base_dataset, novel_dataset = get_omniglot_loaders(
        DATA_DIR, BATCH_SIZE
    )
    print(f'Base classes: {len(base_dataset.classes)}, Novel classes: {len(novel_dataset.classes)}')

    model = OmniglotBackbone(latent_dim=LATENT_DIM, noise_dim=NOISE_DIM).to(DEVICE)
    print(f'Backbone params: {model.count_parameters():,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=BACKBONE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BACKBONE_EPOCHS)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(BACKBONE_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{BACKBONE_EPOCHS}')
        for support, target, _ in pbar:
            support = support.to(DEVICE)
            target = target.to(DEVICE)

            generated, _ = model(support, target)
            loss = criterion(generated, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'backbone_best.pt'))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'backbone_final.pt'))
    print(f'Training complete. Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
