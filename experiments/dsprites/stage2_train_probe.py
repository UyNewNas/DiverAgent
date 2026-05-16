import os, sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.dsprites.config import (
    BATCH_SIZE, PROBE_EPOCHS, PROBE_LR, NOVELTY_MEMORY_SIZE, DEVICE,
)
from experiments.dsprites.dataset import get_dataloaders
from experiments.dsprites.backbone import DSpritesCBDP

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def build_memory_bank(model, loader, max_samples=NOVELTY_MEMORY_SIZE):
    print('Building memory bank...')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x, shape_idx, color_idx, _, _ in tqdm(loader, desc='Memory'):
            x = x.to(DEVICE)
            _, obj_emb, _, _, _, _, _, _, _ = model.backbone(x, shape_idx.to(DEVICE), color_idx.to(DEVICE))
            embeddings.append(obj_emb.cpu())
            if len(torch.cat(embeddings)) >= max_samples:
                break
    all_z = torch.cat(embeddings)[:max_samples]
    model.novelty_loss.update_memory(all_z.to(DEVICE))
    model.novelty_loss.is_full = True
    print(f'Memory bank filled with {len(all_z)} samples.')


def main():
    train_loader, _, _, _, _, _ = get_dataloaders(BATCH_SIZE)

    model = DSpritesCBDP().to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, 'backbone_best.pt')
    if os.path.exists(ckpt_path):
        model.backbone.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print('Loaded pretrained backbone.')
    else:
        print('ERROR: No backbone checkpoint. Run stage1 first.')
        return

    stats = model.parameter_stats()
    print(f'Backbone: {stats["backbone"]:,} params')
    print(f'Probe:    {stats["probe"]:,} params ({stats["probe_ratio"]:.1f}% of backbone)')
    print(f'Gate:     {stats["gate"]:,} params')

    build_memory_bank(model, train_loader)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        list(model.probe.parameters()) + list(model.gate.parameters()),
        lr=PROBE_LR,
    )

    print(f'\nTraining directional probe (K={model.k_outputs} outputs)...')
    for epoch in range(PROBE_EPOCHS):
        model.train()
        epoch_losses = {'diversity': 0.0, 'plausibility': 0.0, 'novelty': 0.0, 'total': 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Probe Epoch {epoch+1}/{PROBE_EPOCHS}')
        for x, shape_idx, color_idx, _, _ in pbar:
            x = x.to(DEVICE)

            with torch.no_grad():
                _, obj_emb, attr_emb, _, _, _, _, _, _ = model.backbone(
                    x, shape_idx.to(DEVICE), color_idx.to(DEVICE)
                )

            outputs_k, z_k, _ = model.forward_divergent(obj_emb, attr_emb)
            loss, loss_dict = model.compute_divergent_loss(outputs_k, x, z_k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1
            pbar.set_postfix({
                'total': f'{loss_dict["total"]:.4f}',
                'div': f'{loss_dict["diversity"]:.4f}',
                'pla': f'{loss_dict["plausibility"]:.4f}',
                'nov': f'{loss_dict["novelty"]:.4f}',
            })

        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        print(f'Epoch {epoch+1}: ' + ', '.join(f'{k}={v:.4f}' for k, v in epoch_losses.items()))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cbdp_full.pt'))
    print('Probe training complete. Model saved.')


if __name__ == '__main__':
    main()
