import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys
import random
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from diver_agent.cbdp import CBDP
from diver_agent.losses import diversity_loss, plausibility_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LATENT_DIM = 32
K_OUTPUTS = 4
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def get_test_loader(held_out_digit=9, n_samples=200):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    indices = [i for i, (_, y) in enumerate(test_dataset) if y == held_out_digit]
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    subset = Subset(test_dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)


def compute_novelty_score(outputs_k, memory_loader, model, threshold=0.95):
    memory_embeddings = []
    model.eval()
    with torch.no_grad():
        for x, _ in memory_loader:
            x = x.to(DEVICE)
            z = model.backbone.encode(x)
            memory_embeddings.append(z.cpu())
    memory = torch.cat(memory_embeddings)[:5000].to(DEVICE)
    memory_norm = F.normalize(memory, dim=-1)

    batch, K, C, H, W = outputs_k.shape
    novelties = []
    with torch.no_grad():
        for i in range(batch):
            imgs = outputs_k[i]
            z_k = model.backbone.encode(imgs)
            z_norm = F.normalize(z_k, dim=-1)
            sim = torch.mm(z_norm, memory_norm.t())
            max_sim, _ = sim.max(dim=1)
            novel = (max_sim < threshold).float().mean()
            novelties.append(novel.item())
    return np.mean(novelties)


def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = CBDP(latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if not os.path.exists(ckpt_path):
        print('No trained model found. Run train_backbone.py and train_probe.py first.')
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    mem_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    mem_indices = random.sample(range(len(mem_dataset)), 2000)
    memory_loader = DataLoader(Subset(mem_dataset, mem_indices), batch_size=BATCH_SIZE, shuffle=False)

    for held_out_digit in range(10):
        print(f'\n{"="*60}')
        print(f'Evaluating on held-out digit: {held_out_digit}')
        print(f'{"="*60}')

        test_loader = get_test_loader(held_out_digit=held_out_digit)

        backbone_mse = 0.0
        probe_mse = 0.0
        all_div_scores = []
        all_plausibility_scores = []
        n_batches = 0
        first_outputs_k = None

        with torch.no_grad():
            for x, _ in tqdm(test_loader, desc='Evaluating'):
                x = x.to(DEVICE)

                x_recon, z = model.forward_convergent(x)
                backbone_mse += F.mse_loss(x_recon, x).item()

                outputs_k, z_perturbed, _ = model.forward_divergent(x)
                if first_outputs_k is None:
                    first_outputs_k = outputs_k

                min_mse = F.mse_loss(
                    outputs_k, x.unsqueeze(1).expand_as(outputs_k), reduction='none'
                ).view(x.size(0), K_OUTPUTS, -1).mean(dim=-1).min(dim=1)[0].mean()
                probe_mse += min_mse.item()

                div_score = 1.0 - diversity_loss(outputs_k).item()
                all_div_scores.append(div_score)

                pla_score = 1.0 - plausibility_loss(outputs_k, x).item()
                all_plausibility_scores.append(pla_score)

                n_batches += 1

        backbone_mse /= n_batches
        probe_mse /= n_batches
        mean_div = np.mean(all_div_scores)
        mean_pla = np.mean(all_plausibility_scores)

        novelty = compute_novelty_score(first_outputs_k, memory_loader, model)

        dci_probe = (mean_pla * mean_div * novelty) ** (1 / 3)

        print(f'\n--- Results for digit {held_out_digit} ---')
        print(f'Backbone MSE (convergent):       {backbone_mse:.6f}')
        print(f'Probe best MSE (divergent):       {probe_mse:.6f}')
        print(f'Diversity score (higher=better):  {mean_div:.4f}')
        print(f'Plausibility score (higher=better): {mean_pla:.4f}')
        print(f'Novelty score (higher=better):    {novelty:.4f}')
        print(f'DCI (geometric mean):             {dci_probe:.4f}')

        results = {
            'held_out_digit': held_out_digit,
            'backbone_mse': backbone_mse,
            'probe_best_mse': probe_mse,
            'diversity': mean_div,
            'plausibility': mean_pla,
            'novelty': novelty,
            'dci': dci_probe,
        }

        with open(os.path.join(RESULTS_DIR, f'results_digit_{held_out_digit}.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nAll results saved to {RESULTS_DIR}')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    evaluate()
