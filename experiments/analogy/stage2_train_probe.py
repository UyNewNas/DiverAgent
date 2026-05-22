import os, sys
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.analogy.config import (
    BATCH_SIZE, PROBE_EPOCHS, PROBE_LR, NOVELTY_MEMORY_SIZE, DEVICE,
)
from experiments.analogy.dataset import get_dataloaders
from experiments.analogy.backbone import AnalogyCBDP

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'results')


def build_memory_bank(model, loader, max_samples=NOVELTY_MEMORY_SIZE):
    print('Building memory bank...')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for head_idx, rel_idx, tail_idx, _, _ in tqdm(loader, desc='Memory'):
            head_idx = head_idx.to(DEVICE)
            tail_idx = tail_idx.to(DEVICE)
            true_tail = model.backbone.word_embed(tail_idx)
            embeddings.append(true_tail.cpu())
            if len(torch.cat(embeddings)) >= max_samples:
                break
    all_z = torch.cat(embeddings)[:max_samples]
    model.novelty_loss.update_memory(all_z.to(DEVICE))
    model.novelty_loss.is_full = True
    print(f'Memory bank filled with {len(all_z)} tail embeddings.')


def main():
    train_loader, test_loader, embeddings, vocab, train_ds, test_ds = \
        get_dataloaders(BATCH_SIZE)
    print(f'Train: {len(train_ds)}, Test: {len(test_ds)}')

    model = AnalogyCBDP(embeddings=embeddings).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, 'backbone_best.pt')
    if os.path.exists(ckpt_path):
        model.backbone.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print('Loaded pretrained backbone.')
    else:
        print('ERROR: No backbone checkpoint. Run stage1 first.')
        return

    stats = model.parameter_stats()
    print(f'Backbone: {stats["backbone"]:,} total ({stats["backbone_trainable"]:,} trainable)')
    print(f'Probe:    {stats["probe"]:,} params ({stats["probe_ratio"]:.2f}% of backbone)')
    print(f'Gate:     {stats["gate"]:,} params')

    build_memory_bank(model, train_loader)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        list(model.probe.parameters()) + list(model.gate.parameters()),
        lr=PROBE_LR,
    )

    log_path = os.path.join(LOG_DIR, 'train_stage2.log')
    log_f = open(log_path, 'w')

    print(f'\nTraining directional probe (K={model.k_outputs} outputs)...')
    for epoch in range(PROBE_EPOCHS):
        model.train()
        epoch_losses = {'diversity': 0.0, 'plausibility': 0.0, 'novelty': 0.0, 'total': 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Probe Epoch {epoch+1}/{PROBE_EPOCHS}')
        for head_idx, rel_idx, tail_idx, _, _ in pbar:
            head_idx = head_idx.to(DEVICE)
            rel_idx = rel_idx.to(DEVICE)
            tail_idx = tail_idx.to(DEVICE)

            with torch.no_grad():
                rel_vec, head_emb, rel_emb = model.backbone.encode_relation(
                    head_idx, rel_idx
                )
                true_tail = model.backbone.word_embed(tail_idx)

            outputs_k, z_k, _ = model.forward_divergent(head_emb, rel_vec, rel_emb)
            loss, loss_dict = model.compute_divergent_loss(outputs_k, true_tail, z_k)

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
        msg = (f'Epoch {epoch+1}: ' +
               ', '.join(f'{k}={v:.4f}' for k, v in epoch_losses.items()))
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cbdp_full.pt'))
    log_f.close()
    print('Probe training complete. Model saved.')


if __name__ == '__main__':
    main()
