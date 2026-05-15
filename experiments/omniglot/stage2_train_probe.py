import os
import sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.omniglot.config import (
    IMAGE_SIZE, LATENT_DIM, NOISE_DIM, K_SHOT, K_OUTPUTS, BATCH_SIZE,
    PROBE_EPOCHS, PROBE_LR, NOVELTY_MEMORY_SIZE, DEVICE,
)
from experiments.omniglot.dataset import get_omniglot_loaders
from experiments.omniglot.backbone import OmniglotCBDP

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def build_memory_bank(model, loader, max_samples=NOVELTY_MEMORY_SIZE):
    print('Building memory bank...')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for support, _, _ in tqdm(loader, desc='Memory'):
            support = support.to(DEVICE)
            class_emb = model.backbone.encode_set(support)
            embeddings.append(class_emb.cpu())
            if len(torch.cat(embeddings)) >= max_samples:
                break
    all_z = torch.cat(embeddings)[:max_samples]
    model.novelty_loss.update_memory(all_z.to(DEVICE))
    model.novelty_loss.is_full = True
    print(f'Memory bank filled with {len(all_z)} samples.')


def main():
    train_loader, novel_loader, base_dataset, novel_dataset = get_omniglot_loaders(
        DATA_DIR, BATCH_SIZE
    )

    model = OmniglotCBDP(latent_dim=LATENT_DIM, noise_dim=NOISE_DIM, k_outputs=K_OUTPUTS).to(DEVICE)

    backbone_path = os.path.join(SAVE_DIR, 'backbone_best.pt')
    if os.path.exists(backbone_path):
        model.backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
        print('Loaded pretrained backbone.')
    else:
        print('ERROR: No backbone checkpoint found. Run stage1 first.')
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE_EPOCHS)

    print(f'\nTraining divergent probe (K={K_OUTPUTS} outputs)...')
    for epoch in range(PROBE_EPOCHS):
        model.train()
        epoch_losses = {'diversity': 0.0, 'plausibility': 0.0, 'novelty': 0.0, 'total': 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Probe Epoch {epoch+1}/{PROBE_EPOCHS}')
        for support, target, _ in pbar:
            support = support.to(DEVICE)
            target = target.to(DEVICE)

            with torch.no_grad():
                class_emb = model.backbone.encode_set(support)

            outputs_k, z_perturbed, _ = model.forward_divergent(class_emb)
            loss, loss_dict = model.compute_divergent_loss(outputs_k, target, z_perturbed)

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

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        print(f'Epoch {epoch+1}: ' + ', '.join(f'{k}={v:.4f}' for k, v in epoch_losses.items()))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cbdp_full.pt'))
    print('Probe training complete. Model saved.')


if __name__ == '__main__':
    main()
