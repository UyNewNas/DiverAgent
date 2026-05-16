import json, os, sys, random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.dsprites.config import (
    IMAGE_SIZE, BATCH_SIZE, K_OUTPUTS, DEVICE,
    NUM_SHAPES, NUM_COLORS, LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY,
    SHAPES, COLORS,
)
from experiments.dsprites.dataset import get_dataloaders
from experiments.dsprites.backbone import (
    DSpritesCBDP, diversity_loss, plausibility_loss,
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def backbone_baseline(model, obj_emb, k=K_OUTPUTS):
    z_k = obj_emb.unsqueeze(1).expand(-1, k, -1)
    B, K, D = z_k.shape
    z_flat = z_k.reshape(B * K, D)
    flat_out = model.backbone.decode(z_flat)
    C = 3
    return flat_out.view(B, K, C, IMAGE_SIZE, IMAGE_SIZE)


def cosine_diversity_value(outputs_k):
    B, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(B, K, -1)
    flat_n = F.normalize(flat, dim=-1)
    sim = torch.bmm(flat_n, flat_n.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pw = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return pw.mean().item()


def plausibility_mse_score(outputs_k, target):
    return (1.0 - plausibility_loss(outputs_k, target).item())


def compute_novelty_score(outputs_k, image_memory):
    B, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(B * K, -1)
    flat_n = F.normalize(flat, dim=-1)
    mem_flat = image_memory.flatten(1).to(outputs_k.device)
    mem_n = F.normalize(mem_flat, dim=-1)
    sim = torch.mm(flat_n, mem_n.t())
    max_sim, _ = sim.max(dim=1)
    return (1.0 - max_sim.mean()).item()


def build_image_memory(loader, max_samples=3000):
    print('Building image memory...')
    memory_list = []
    for x, *_ in loader:
        memory_list.append(x)
        if len(torch.cat(memory_list)) >= max_samples:
            break
    return torch.cat(memory_list)[:max_samples]


def evaluate(model, loader, image_memory, tau=0.3):
    model.eval()
    metrics = {
        'cos_sim': [], 'pla': [], 'novelty': [], 'dci': [],
        'backbone_cos_sim': [], 'backbone_pla': [], 'backbone_novelty': [], 'backbone_dci': [],
    }

    for x, shape_idx, color_idx, _, _ in tqdm(loader, desc='Evaluate'):
        x = x.to(DEVICE)

        with torch.no_grad():
            _, obj_emb, attr_emb, _, _, _, _, _, _ = model.backbone(
                x, shape_idx.to(DEVICE), color_idx.to(DEVICE)
            )

            bb_outputs = backbone_baseline(model, obj_emb, k=K_OUTPUTS)
            bb_cos = cosine_diversity_value(bb_outputs)
            bb_pla = plausibility_mse_score(bb_outputs, x)
            bb_nov = compute_novelty_score(bb_outputs, image_memory)

            outputs_k, _, _ = model.forward_divergent(obj_emb, attr_emb)
            pr_cos = cosine_diversity_value(outputs_k)
            pr_pla = plausibility_mse_score(outputs_k, x)
            pr_nov = compute_novelty_score(outputs_k, image_memory)

        div_q_bb = max(0.0, min(1.0, 1.0 - abs(bb_cos - tau)))
        div_q_pr = max(0.0, min(1.0, 1.0 - abs(pr_cos - tau)))
        dci_bb = (bb_pla * div_q_bb * bb_nov) ** (1 / 3)
        dci_pr = (pr_pla * div_q_pr * pr_nov) ** (1 / 3)

        for k, v in [
            ('cos_sim', pr_cos), ('pla', pr_pla), ('novelty', pr_nov), ('dci', dci_pr),
            ('backbone_cos_sim', bb_cos), ('backbone_pla', bb_pla),
            ('backbone_novelty', bb_nov), ('backbone_dci', dci_bb),
        ]:
            metrics[k].append(v)

    return {k: np.mean(v) for k, v in metrics.items()}


def shape_color_retention(model, loader):
    model.eval()
    shape_ret = []
    color_ret = []
    total = 0

    with torch.no_grad():
        for x, shape_idx, color_idx, _, _ in tqdm(loader, desc='Retention'):
            x = x.to(DEVICE)
            _, obj_emb, attr_emb, _, _, _, _, _, _ = model.backbone(
                x, shape_idx.to(DEVICE), color_idx.to(DEVICE)
            )
            outputs_k, _, _ = model.forward_divergent(obj_emb, attr_emb)
            best_out = outputs_k[:, 0]

            for i in range(x.size(0)):
                feat_orig = model.backbone.encoder(x[i:i+1])
                obj_orig = model.backbone.object_proj(feat_orig)
                attr_orig = model.backbone.attr_proj(feat_orig)

                feat_dec = model.backbone.encoder(best_out[i:i+1])
                obj_dec = model.backbone.object_proj(feat_dec)
                attr_dec = model.backbone.attr_proj(feat_dec)

                obj_sim = F.cosine_similarity(obj_orig, obj_dec).item()
                attr_sim = F.cosine_similarity(attr_orig, attr_dec).item()
                shape_ret.append(obj_sim)
                color_ret.append(attr_sim)
                total += 1

    if total == 0:
        return 0.0, 0.0
    return np.mean(shape_ret), np.mean(color_ret)


def ablation_experiment(model, train_loader, eval_loader, image_memory):
    configs = [
        ('λ_div=0', (0.0, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY)),
        ('λ_pla=0', (LAMBDA_DIVERSITY, 0.0, LAMBDA_NOVELTY)),
        ('λ_nov=0', (LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, 0.0)),
        ('full', (LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY)),
    ]
    results = {}
    for label, lambdas in configs:
        print(f'\n--- Ablation: {label} ---')
        import copy
        m = copy.deepcopy(model).to(DEVICE)
        m.freeze_backbone()
        opt = torch.optim.Adam(list(m.probe.parameters()) + list(m.gate.parameters()), lr=1e-3)
        import torch.nn.functional as F
        for ep in range(10):
            for x, shape_idx, color_idx, _, _ in train_loader:
                x = x.to(DEVICE)
                with torch.no_grad():
                    _, obj_e, attr_e, _, _, _, _, _, _ = m.backbone(x, shape_idx.to(DEVICE), color_idx.to(DEVICE))
                ok, zk, _ = m.forward_divergent(obj_e, attr_e)
                ld = diversity_loss(ok)
                lp = plausibility_loss(ok, x)
                ln = m.novelty_loss(zk)
                total = lambdas[0] * (ld - 0.7) ** 2 + lambdas[1] * lp + lambdas[2] * ln
                opt.zero_grad()
                total.backward()
                opt.step()
        metrics = evaluate(m, eval_loader, image_memory)
        results[label] = metrics
        print(f'  DCI={metrics["dci"]:.4f} cos_sim={metrics["cos_sim"]:.4f}')
    return results


def random_baseline(model, eval_loader, image_memory):
    results = {}
    for scale in [0.05, 0.1, 0.3, 0.5]:
        print(f'\n--- Random baseline: scale={scale} ---')
        def fake_probe_forward(self, obj_emb, attr_emb):
            B, D = obj_emb.shape
            transport = torch.randn(B, K_OUTPUTS, D, device=DEVICE) * scale
            z_k = obj_emb.unsqueeze(1) + transport + attr_emb.unsqueeze(1)
            BF, KD = z_k.reshape(B * K_OUTPUTS, D).shape
            injections = [
                torch.zeros(B * K_OUTPUTS, 256, 1, 1, device=DEVICE),
                torch.zeros(B * K_OUTPUTS, 128, 1, 1, device=DEVICE),
                torch.zeros(B * K_OUTPUTS, 64, 1, 1, device=DEVICE),
                torch.zeros(B * K_OUTPUTS, 32, 1, 1, device=DEVICE),
            ]
            return z_k, transport, injections

        orig = model.probe.forward
        model.probe.forward = fake_probe_forward.__get__(model.probe, type(model.probe))
        metrics = evaluate(model, eval_loader, image_memory)
        model.probe.forward = orig
        results[f'random_{scale}'] = metrics
        print(f'  DCI={metrics["dci"]:.4f} cos_sim={metrics["cos_sim"]:.4f}')
    return results


def main():
    train_loader, test_loader, train_eval_loader, test_eval_loader, _, _ = get_dataloaders(BATCH_SIZE)

    model = DSpritesCBDP().to(DEVICE)
    ckpt = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if not os.path.exists(ckpt):
        print('ERROR: No CBDP checkpoint. Run stage1 and stage2 first.')
        return
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print('Loaded CBDP model.')

    stats = model.parameter_stats()
    print(f'Probe: {stats["probe"]:,} params ({stats["probe_ratio"]:.1f}% of backbone)')

    image_memory = build_image_memory(train_loader)

    print('\n' + '=' * 60)
    print('Phase 2a: colored dSprites Concept Combination Evaluation')
    print('=' * 60)

    print('\n--- Train Combos (seen shape+color) ---')
    train_metrics = evaluate(model, train_eval_loader, image_memory)
    print(f'Backbone: cos={train_metrics["backbone_cos_sim"]:.4f} pla={train_metrics["backbone_pla"]:.4f} '
          f'nov={train_metrics["backbone_novelty"]:.4f} DCI={train_metrics["backbone_dci"]:.4f}')
    print(f'CBDP:     cos={train_metrics["cos_sim"]:.4f} pla={train_metrics["pla"]:.4f} '
          f'nov={train_metrics["novelty"]:.4f} DCI={train_metrics["dci"]:.4f}')

    print('\n--- Test Combos (ZERO-SHOT shape+magenta/cyan) ---')
    test_metrics = evaluate(model, test_eval_loader, image_memory)
    print(f'Backbone: cos={test_metrics["backbone_cos_sim"]:.4f} pla={test_metrics["backbone_pla"]:.4f} '
          f'nov={test_metrics["backbone_novelty"]:.4f} DCI={test_metrics["backbone_dci"]:.4f}')
    print(f'CBDP:     cos={test_metrics["cos_sim"]:.4f} pla={test_metrics["pla"]:.4f} '
          f'nov={test_metrics["novelty"]:.4f} DCI={test_metrics["dci"]:.4f}')

    dci_gap = abs(test_metrics["dci"] - train_metrics["dci"])
    print(f'DCI gap (train - test): {dci_gap:.4f}')

    print('\n--- Shape/Color Retention (cosine similarity original vs decoded) ---')
    train_s_ret, train_c_ret = shape_color_retention(model, train_eval_loader)
    test_s_ret, test_c_ret = shape_color_retention(model, test_eval_loader)
    print(f'Train: shape_ret={train_s_ret:.4f} color_ret={train_c_ret:.4f}')
    print(f'Test:  shape_ret={test_s_ret:.4f} color_ret={test_c_ret:.4f}')

    print('\n--- Random Baseline ---')
    rand_results = random_baseline(model, test_eval_loader, image_memory)

    print('\n--- Ablation (test combos) ---')
    abl_results = ablation_experiment(model, train_loader, test_eval_loader, image_memory)

    all_results = {
        'train': train_metrics,
        'test': test_metrics,
        'dci_gap': dci_gap,
        'retention': {
            'train_shape_ret': train_s_ret,
            'train_color_ret': train_c_ret,
            'test_shape_ret': test_s_ret,
            'test_color_ret': test_c_ret,
        },
        'random_baseline': {k: v for k, v in rand_results.items()},
        'ablation': {k: v for k, v in abl_results.items()},
    }
    path = os.path.join(RESULT_DIR, 'phase2a_results.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f'\nResults saved to {path}')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
