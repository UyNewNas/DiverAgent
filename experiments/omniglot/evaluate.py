import json
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.omniglot.config import (
    IMAGE_SIZE, LATENT_DIM, NOISE_DIM, K_SHOT, K_OUTPUTS, BATCH_SIZE, DEVICE,
)
from experiments.omniglot.dataset import get_omniglot_loaders
from experiments.omniglot.backbone import OmniglotCBDP

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def backbone_baseline(model, class_emb, k=K_OUTPUTS):
    class_emb_expanded = class_emb.unsqueeze(1).expand(-1, k, -1)
    B, K, D = class_emb_expanded.shape
    flat_emb = class_emb_expanded.reshape(B * K, D)
    noise = torch.randn(B * K, model.backbone.noise_dim, device=DEVICE)
    flat_out = model.backbone.decode(flat_emb, noise)
    _, C, H, W = flat_out.shape
    return flat_out.view(B, K, C, H, W)


def cosine_diversity(outputs_k):
    B, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(B, K, -1)
    flat_norm = F.normalize(flat, dim=-1)
    sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pairwise = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return pairwise.mean().item()


def plausibility_score(outputs_k, support_images):
    B, K, C, H, W = outputs_k.shape
    _, S, _, _, _ = support_images.shape
    flat_out = outputs_k.view(B, K, -1)
    flat_sup = support_images.view(B, S, -1)
    flat_out_n = F.normalize(flat_out, dim=-1)
    flat_sup_n = F.normalize(flat_sup, dim=-1)
    sim = torch.bmm(flat_out_n, flat_sup_n.transpose(1, 2))
    best_sim, _ = sim.max(dim=2)
    avg_best_sim = best_sim.mean().item()
    return max(0.0, min(1.0, avg_best_sim))


def compute_novelty_score(outputs_k, memory):
    B, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(B * K, -1)
    flat_norm = F.normalize(flat, dim=-1)
    mem_flat = memory.flatten(1).to(outputs_k.device)
    mem_norm = F.normalize(mem_flat, dim=-1)
    sim = torch.mm(flat_norm, mem_norm.t())
    max_sim, _ = sim.max(dim=1)
    return (1.0 - max_sim.mean()).item()


def build_image_memory(loader, model, max_samples=2000):
    print('Building image feature memory for novelty...')
    model.eval()
    memory_list = []
    with torch.no_grad():
        for support, target, _ in loader:
            support = support.to(DEVICE)
            class_emb = model.backbone.encode_set(support)
            noise = torch.randn(class_emb.size(0), model.backbone.noise_dim, device=DEVICE)
            recon = model.backbone.decode(class_emb, noise)
            memory_list.append(recon.cpu())
            if len(torch.cat(memory_list)) >= max_samples:
                break
    return torch.cat(memory_list)[:max_samples]


def evaluate(model, loader, image_memory, tag, tau=0.3):
    model.eval()
    results = {
        'tag': tag,
        'backbone_cos_sim': [],
        'backbone_pla': [],
        'backbone_dci': [],
        'probe_cos_sim': [],
        'probe_pla': [],
        'probe_novelty': [],
        'probe_dci': [],
    }

    for support, target, _ in tqdm(loader, desc=f'Evaluate {tag}'):
        support = support.to(DEVICE)
        target = target.to(DEVICE)

        with torch.no_grad():
            class_emb = model.backbone.encode_set(support)

            bb_outputs = backbone_baseline(model, class_emb, k=K_OUTPUTS)
            bb_cos = cosine_diversity(bb_outputs)
            bb_pla = plausibility_score(bb_outputs, support)

            outputs_k, _, _ = model.forward_divergent(class_emb)
            pr_cos = cosine_diversity(outputs_k)
            pr_pla = plausibility_score(outputs_k, support)
            pr_nov = compute_novelty_score(outputs_k, image_memory)

        div_qual_bb = max(0.0, min(1.0, 1.0 - abs(bb_cos - tau)))
        dci_bb = (bb_pla * div_qual_bb) ** 0.5
        results['backbone_cos_sim'].append(bb_cos)
        results['backbone_pla'].append(bb_pla)
        results['backbone_dci'].append(dci_bb)

        div_qual_pr = max(0.0, min(1.0, 1.0 - abs(pr_cos - tau)))
        dci_pr = (pr_pla * div_qual_pr * pr_nov) ** (1 / 3)
        results['probe_cos_sim'].append(pr_cos)
        results['probe_pla'].append(pr_pla)
        results['probe_novelty'].append(pr_nov)
        results['probe_dci'].append(dci_pr)

    agg = {}
    for k, v in results.items():
        if k != 'tag' and v:
            agg[k] = sum(v) / len(v)
        else:
            agg[k] = v

    agg['dci_improvement'] = (agg['probe_dci'] - agg['backbone_dci']) / max(agg['backbone_dci'], 1e-8) * 100
    return agg


def main():
    train_loader, novel_loader, base_dataset, novel_dataset = get_omniglot_loaders(
        DATA_DIR, BATCH_SIZE
    )

    model = OmniglotCBDP(latent_dim=LATENT_DIM, noise_dim=NOISE_DIM, k_outputs=K_OUTPUTS).to(DEVICE)

    ckpt_path = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: No CBDP checkpoint at {ckpt_path}. Run stage1 and stage2 first.')
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print('Loaded CBDP model.')

    image_memory = build_image_memory(train_loader, model)

    print('\n' + '=' * 60)
    print('Phase 1: Omniglot Few-Shot Generation Evaluation')
    print('=' * 60)

    base_results = evaluate(model, train_loader, image_memory, 'base_classes')

    novel_results = evaluate(model, novel_loader, image_memory, 'novel_classes')

    print('\n--- Base Classes (seen during training) ---')
    print(f'Backbone: cos_sim={base_results["backbone_cos_sim"]:.4f}, '
          f'pla={base_results["backbone_pla"]:.4f}, DCI={base_results["backbone_dci"]:.4f}')
    print(f'CBDP:     cos_sim={base_results["probe_cos_sim"]:.4f}, '
          f'pla={base_results["probe_pla"]:.4f}, '
          f'novelty={base_results["probe_novelty"]:.4f}, '
          f'DCI={base_results["probe_dci"]:.4f}')
    print(f'DCI improvement: {base_results["dci_improvement"]:+.1f}%')

    print('\n--- Novel Classes (held-out, few-shot) ---')
    print(f'Backbone: cos_sim={novel_results["backbone_cos_sim"]:.4f}, '
          f'pla={novel_results["backbone_pla"]:.4f}, DCI={novel_results["backbone_dci"]:.4f}')
    print(f'CBDP:     cos_sim={novel_results["probe_cos_sim"]:.4f}, '
          f'pla={novel_results["probe_pla"]:.4f}, '
          f'novelty={novel_results["probe_novelty"]:.4f}, '
          f'DCI={novel_results["probe_dci"]:.4f}')
    print(f'DCI improvement: {novel_results["dci_improvement"]:+.1f}%')

    all_results = {'base': base_results, 'novel': novel_results}
    result_path = os.path.join(RESULT_DIR, 'phase1_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {result_path}')


if __name__ == '__main__':
    main()
