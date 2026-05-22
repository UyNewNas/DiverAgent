import os, sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.analogy.config import (
    BATCH_SIZE, BACKBONE_EPOCHS, BACKBONE_LR, DEVICE,
)
from experiments.analogy.dataset import get_dataloaders
from experiments.analogy.backbone import AnalogyBackbone

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f'Device: {DEVICE}')
    train_loader, test_loader, embeddings, vocab, train_ds, test_ds = \
        get_dataloaders(BATCH_SIZE)
    print(f'Train: {len(train_ds)}, Test: {len(test_ds)}')

    model = AnalogyBackbone(embeddings=embeddings).to(DEVICE)
    print(f'Backbone params: {model.count_parameters():,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=BACKBONE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=BACKBONE_EPOCHS
    )

    best_loss = float('inf')
    log_path = os.path.join(LOG_DIR, 'train_stage1.log')
    log_f = open(log_path, 'w')

    for epoch in range(BACKBONE_EPOCHS):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_align = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{BACKBONE_EPOCHS}')
        for head_idx, rel_idx, tail_idx, _, _ in pbar:
            head_idx = head_idx.to(DEVICE)
            rel_idx = rel_idx.to(DEVICE)
            tail_idx = tail_idx.to(DEVICE)

            tail_pred, _, _, _, align_loss = model(head_idx, rel_idx)
            true_tail = model.word_embed(tail_idx)
            pred_n = F.normalize(tail_pred, dim=-1)
            true_n = F.normalize(true_tail, dim=-1)
            cos_loss = 1.0 - (pred_n * true_n).sum(dim=-1).mean()
            loss = cos_loss + 0.1 * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += cos_loss.item()
            total_align += align_loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cos': f'{cos_loss.item():.4f}',
            })

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_align = total_align / len(train_loader)
        msg = (f'Epoch {epoch+1}: loss={avg_loss:.4f} '
               f'recon={avg_recon:.4f} align={avg_align:.4f}')
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, 'backbone_best.pt'))
            print(f'  best so far: {best_loss:.4f}')

    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, 'backbone_final.pt'))
    log_f.close()
    print(f'Training complete. Best loss: {best_loss:.4f}')
    print(f'Log saved to {log_path}')


if __name__ == '__main__':
    main()
