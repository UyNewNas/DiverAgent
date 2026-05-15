import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from diver_agent.cbdp import CBDP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
LATENT_DIM = 32
K_OUTPUTS = 4
PROBE_EPOCHS = 10
PROBE_LR = 1e-3
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def get_mnist_loaders():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def build_memory_bank(model, loader, max_samples=5000):
    print('Building memory bank...')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc='Memory'):
            x = x.to(DEVICE)
            z = model.backbone.encode(x)
            embeddings.append(z.cpu())
            if len(torch.cat(embeddings)) >= max_samples:
                break
    all_z = torch.cat(embeddings)[:max_samples]
    model.novelty_loss.update_memory(all_z.to(DEVICE))
    print(f'Memory bank filled with {min(len(all_z), max_samples)} samples.')


def main():
    model = CBDP(latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS).to(DEVICE)

    backbone_path = os.path.join(SAVE_DIR, 'backbone_best.pt')
    if os.path.exists(backbone_path):
        model.backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
        print('Loaded pretrained backbone.')
    else:
        print('WARNING: No pretrained backbone found. Run train_backbone.py first.')
        return

    stats = model.parameter_stats()
    print(f'Backbone: {stats["backbone"]:,} params')
    print(f'Probe:    {stats["probe"]:,} params ({stats["probe_ratio"]:.1f}% of backbone)')
    print(f'Gate:     {stats["gate"]:,} params')

    train_loader, test_loader = get_mnist_loaders()

    build_memory_bank(model, train_loader)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        list(model.probe.parameters()) + list(model.gate.parameters()),
        lr=PROBE_LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE_EPOCHS)

    print('\nTraining divergent probe...')
    for epoch in range(PROBE_EPOCHS):
        model.train()
        epoch_losses = {'diversity': 0.0, 'plausibility': 0.0, 'novelty': 0.0, 'total': 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Probe Epoch {epoch+1}/{PROBE_EPOCHS}')
        for x, _ in pbar:
            x = x.to(DEVICE)
            outputs_k, z_perturbed, _ = model.forward_divergent(x)
            loss, loss_dict = model.compute_divergent_loss(outputs_k, x, z_perturbed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss_dict["total"]:.4f}'})

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        print(f'Epoch {epoch+1}: ' + ', '.join(f'{k}={v:.4f}' for k, v in epoch_losses.items()))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cbdp_full.pt'))
    print('Probe training complete. Model saved.')


if __name__ == '__main__':
    main()
