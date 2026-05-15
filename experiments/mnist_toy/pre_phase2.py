import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys
import random
import json
import copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from diver_agent.cbdp import CBDP
from diver_agent.losses import diversity_loss, plausibility_loss, divergent_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
LATENT_DIM = 32
K_OUTPUTS = 4
PROBE_EPOCHS = 20
PROBE_LR = 1e-3
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def get_mnist_loaders():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset


def build_memory_bank(model, loader):
    print('Building memory bank...')
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc='Memory'):
            x = x.to(DEVICE)
            z = model.backbone.encode(x)
            embeddings.append(z.cpu())
            if len(torch.cat(embeddings)) >= 5000:
                break
    all_z = torch.cat(embeddings)[:5000]
    model.novelty_loss.update_memory(all_z.to(DEVICE))
    model.novelty_loss.is_full = True
    print(f'Memory bank filled with {len(all_z)} samples.')


def load_backbone(model):
    backbone_path = os.path.join(SAVE_DIR, 'backbone_best.pt')
    model.backbone.load_state_dict(torch.load(backbone_path, map_location=DEVICE))


def train_probe(model, train_loader, epochs, lr, lambdas):
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        list(model.probe.parameters()) + list(model.gate.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'diversity': 0.0, 'plausibility': 0.0, 'novelty': 0.0, 'total': 0.0}
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for x, _ in pbar:
            x = x.to(DEVICE)
            outputs_k, z_perturbed, _ = model.forward_divergent(x)
            loss, loss_dict = divergent_loss(
                outputs_k, x, z_perturbed, model.novelty_loss,
                lambda_diversity=lambdas[0],
                lambda_plausibility=lambdas[1],
                lambda_novelty=lambdas[2],
            )
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
    return epoch_losses


def evaluate_model(model, test_loader, memory_loader, tau=0.3):
    model.eval()
    backbone_mse = 0.0
    probe_mse = 0.0
    all_cos_sim = []
    all_pla = []
    n_batches = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            x_recon, z = model.forward_convergent(x)
            backbone_mse += F.mse_loss(x_recon, x).item()

            outputs_k, z_perturbed, _ = model.forward_divergent(x)
            min_mse = F.mse_loss(
                outputs_k, x.unsqueeze(1).expand_as(outputs_k), reduction='none'
            ).view(x.size(0), K_OUTPUTS, -1).mean(dim=-1).min(dim=1)[0].mean()
            probe_mse += min_mse.item()

            B, K, C, H, W = outputs_k.shape
            flat = outputs_k.view(B, K, -1)
            flat_n = F.normalize(flat, dim=-1)
            sim = torch.bmm(flat_n, flat_n.transpose(1, 2))
            mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
            pw = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
            all_cos_sim.append(pw.mean().item())

            pla = 1.0 - plausibility_loss(outputs_k, x).item()
            all_pla.append(pla)
            n_batches += 1

    backbone_mse /= n_batches
    probe_mse /= n_batches
    cos_sim = np.mean(all_cos_sim)
    pla = np.mean(all_pla)

    memory_embs = []
    with torch.no_grad():
        for mx, _ in memory_loader:
            mx = mx.to(DEVICE)
            z = model.backbone.encode(mx)
            memory_embs.append(z.cpu())
    memory = torch.cat(memory_embs)[:5000].to(DEVICE)
    mem_norm = F.normalize(memory, dim=-1)

    novelties = []
    with torch.no_grad():
        model.eval()
        for x, _ in test_loader:
            x = x.to(DEVICE)
            outputs_k, _, _ = model.forward_divergent(x)
            B = outputs_k.size(0)
            for i in range(B):
                imgs = outputs_k[i]
                z_k = model.backbone.encode(imgs)
                z_n = F.normalize(z_k, dim=-1)
                sim = torch.mm(z_n, mem_norm.t())
                max_sim, _ = sim.max(dim=1)
                nov = (max_sim < 0.95).float().mean()
                novelties.append(nov.item())
    novelty = np.mean(novelties)

    div_qual = max(0.0, min(1.0, 1.0 - abs(cos_sim - tau)))
    dci = (pla * div_qual * novelty) ** (1 / 3)

    return {
        'backbone_mse': backbone_mse,
        'probe_mse': probe_mse,
        'cosine_sim': cos_sim,
        'diversity_quality': div_qual,
        'plausibility': pla,
        'novelty': novelty,
        'dci': dci,
    }


def build_memory_loader(dataset):
    indices = random.sample(range(len(dataset)), 2000)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=64, shuffle=False)


def ablation():
    print('\n' + '=' * 60)
    print('补强 1: Ablation Study — 逐项移除 λ 看三损失各自贡献')
    print('=' * 60)

    train_loader, test_loader, train_dataset, test_dataset = get_mnist_loaders()
    memory_loader = build_memory_loader(train_dataset)

    configs = [
        ('λ_div=0', (0.0, 0.8, 0.5)),
        ('λ_pla=0', (0.15, 0.0, 0.5)),
        ('λ_nov=0', (0.15, 0.8, 0.0)),
        ('full',    (0.15, 0.8, 0.5)),
    ]

    results = {}
    for label, lambdas in configs:
        print(f'\n--- {label} ---')
        model = CBDP(latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS).to(DEVICE)
        load_backbone(model)
        build_memory_bank(model, train_loader)
        model.novelty_loss.is_full = True
        final = train_probe(model, train_loader, PROBE_EPOCHS, PROBE_LR, lambdas)
        print(f'Final: div={final["diversity"]:.4f} pla={final["plausibility"]:.4f} '
              f'nov={final["novelty"]:.4f} total={final["total"]:.4f}')

        metrics = evaluate_model(model, test_loader, memory_loader)
        results[label] = metrics
        print(f'DCI={metrics["dci"]:.4f} cos_sim={metrics["cosine_sim"]:.4f} '
              f'pla={metrics["plausibility"]:.4f} nov={metrics["novelty"]:.4f}')

    print('\n--- Ablation Summary ---')
    print(f'{"Config":<12} {"cos_sim":>8} {"pla":>8} {"nov":>8} {"DCI":>8}')
    for label, m in results.items():
        print(f'{label:<12} {m["cosine_sim"]:>8.4f} {m["plausibility"]:>8.4f} '
              f'{m["novelty"]:>8.4f} {m["dci"]:>8.4f}')

    with open(os.path.join(RESULTS_DIR, 'ablation_results.json'), 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)


def random_baseline():
    print('\n' + '=' * 60)
    print('补强 2: Random Perturbation Baseline — 随机扰动 vs 可学习探针')
    print('=' * 60)

    _, test_loader, train_dataset, test_dataset = get_mnist_loaders()
    memory_loader = build_memory_loader(train_dataset)

    model = CBDP(latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS).to(DEVICE)
    load_backbone(model)
    model.eval()

    scales = [0.05, 0.1, 0.3, 0.5]
    results = {}

    for scale in scales:
        print(f'\n--- Random scale={scale} ---')

        def random_probe(z):
            batch = z.size(0)
            eps = torch.randn(batch, K_OUTPUTS, LATENT_DIM, device=DEVICE) * scale
            z_perturbed = z.unsqueeze(1) + eps
            return z_perturbed, eps

        original = model.probe.forward
        model.probe.forward = random_probe
        metrics = evaluate_model(model, test_loader, memory_loader)
        model.probe.forward = original

        results[f'random_{scale}'] = metrics
        print(f'DCI={metrics["dci"]:.4f} cos_sim={metrics["cosine_sim"]:.4f} '
              f'pla={metrics["plausibility"]:.4f} nov={metrics["novelty"]:.4f}')

    print(f'\nLoading trained CBDP for comparison...')
    ckpt_path = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        metrics = evaluate_model(model, test_loader, memory_loader)
        results['cbdp_trained'] = metrics
        print(f'CBDP: DCI={metrics["dci"]:.4f} cos_sim={metrics["cosine_sim"]:.4f}')

    print('\n--- Random Baseline Summary ---')
    print(f'{"Config":<14} {"cos_sim":>8} {"pla":>8} {"nov":>8} {"DCI":>8}')
    for label, m in results.items():
        print(f'{label:<14} {m["cosine_sim"]:>8.4f} {m["plausibility"]:>8.4f} '
              f'{m["novelty"]:>8.4f} {m["dci"]:>8.4f}')

    with open(os.path.join(RESULTS_DIR, 'random_baseline_results.json'), 'w') as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)


def gate_verify():
    print('\n' + '=' * 60)
    print('补强 3: Gate 行为验证 — 训练中加 L_gate 损失，记录每数字激活分布')
    print('=' * 60)

    train_loader, test_loader, train_dataset, test_dataset = get_mnist_loaders()
    memory_loader = build_memory_loader(train_dataset)
    target_ratio = 0.5

    model = CBDP(latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS).to(DEVICE)
    load_backbone(model)
    build_memory_bank(model, train_loader)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        list(model.probe.parameters()) + list(model.gate.parameters()), lr=PROBE_LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE_EPOCHS)

    print(f'\nTraining with L_gate = (gate_avg - {target_ratio})^2 ...')
    for epoch in range(PROBE_EPOCHS):
        model.train()
        gate_vals = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{PROBE_EPOCHS}')
        for x, _ in pbar:
            x = x.to(DEVICE)
            outputs_k, z_perturbed, _ = model.forward_divergent(x)
            loss, loss_dict = model.compute_divergent_loss(outputs_k, x, z_perturbed)

            gate_val = model.gate(model.backbone.encode(x))[1]
            l_gate = (gate_val.mean() - target_ratio) ** 2
            total = loss + 0.1 * l_gate
            gate_vals.append(gate_val.mean().item())

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'gate_avg': f'{gate_val.mean().item():.3f}'})
        scheduler.step()
        print(f'Epoch {epoch+1}: loss={loss_dict["total"]:.4f} gate_avg={np.mean(gate_vals):.3f}')

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'cbdp_gate.pt'))

    model.eval()
    digit_gate_means = []
    results = []

    for digit in range(10):
        indices = [i for i, (_, y) in enumerate(test_dataset) if y == digit]
        subset = Subset(test_dataset, indices[:200])
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

        gate_per_digit = []
        support_mse_per_digit = 0.0
        n = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE)
                gate_val = model.gate(model.backbone.encode(x))[1]
                gate_per_digit.extend(gate_val.tolist())

                x_recon, _ = model.forward_convergent(x)
                support_mse_per_digit += F.mse_loss(x_recon, x).item()
                n += 1

        mean_gate = np.mean(gate_per_digit)
        digit_gate_means.append(mean_gate)
        results.append({
            'digit': digit,
            'gate_mean': mean_gate,
            'gate_std': np.std(gate_per_digit),
            'backbone_mse': support_mse_per_digit / n,
        })
        print(f'Digit {digit}: gate_mean={mean_gate:.3f}')

    print(f'\nGate target: {target_ratio}, overall mean: {np.mean(digit_gate_means):.3f}')

    with open(os.path.join(RESULTS_DIR, 'gate_verify_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    metrics = evaluate_model(model, test_loader, memory_loader)
    print(f'DCI with gate loss: {metrics["dci"]:.4f}')
    with open(os.path.join(RESULTS_DIR, 'gate_verify_dci.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    print('Pre-Phase 2 Ground Reinforcement')
    ablation()
    random_baseline()
    gate_verify()
    print(f'\nAll results saved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
