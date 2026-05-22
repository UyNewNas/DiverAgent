import json, os, sys, random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.analogy.config import (
    BATCH_SIZE, K_OUTPUTS, DEVICE, TAU, EMBED_DIM,
    LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY,
    RELATION_TYPES,
)
from experiments.analogy.dataset import get_dataloaders
from experiments.analogy.backbone import (
    AnalogyCBDP, diversity_loss, plausibility_loss,
)
from experiments.analogy.tee_logger import setup_logger

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def build_tail_memory(model, loader, max_samples=3000):
    print('Building tail embedding memory...')
    memory_list = []
    with torch.no_grad():
        for head_idx, rel_idx, tail_idx, _, _ in tqdm(loader, desc='Memory'):
            tail_idx = tail_idx.to(DEVICE)
            tail_emb = model.backbone.word_embed(tail_idx).cpu()
            memory_list.append(tail_emb)
            if len(torch.cat(memory_list)) >= max_samples:
                break
    return torch.cat(memory_list)[:max_samples]


def backbone_baseline(model, rel_vec, k=K_OUTPUTS):
    z_k = rel_vec.unsqueeze(1).expand(-1, k, -1)
    B, K, D = z_k.shape
    z_flat = z_k.reshape(B * K, D)
    flat_out = model.backbone.decode(z_flat)
    return flat_out.view(B, K, EMBED_DIM)


def cosine_pairwise_sim(outputs_k):
    B, K, D = outputs_k.shape
    flat_n = F.normalize(outputs_k.view(B * K, D), dim=-1)
    flat_n = flat_n.view(B, K, D)
    sim = torch.bmm(flat_n, flat_n.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pw = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return pw.mean().item()


def plausibility_score(outputs_k, target):
    return (1.0 - plausibility_loss(outputs_k, target).item())


def novelty_vs_memory(outputs_k, tail_memory):
    B, K, D = outputs_k.shape
    flat_q = outputs_k.view(B * K, D)
    flat_q_n = F.normalize(flat_q, dim=-1)
    mem_n = F.normalize(tail_memory.to(outputs_k.device), dim=-1)
    sim = torch.mm(flat_q_n, mem_n.t())
    max_sim, _ = sim.max(dim=1)
    return (1.0 - max_sim.mean()).item()


def analogy_accuracy(outputs_k, true_tail):
    B, K, D = outputs_k.shape
    true_n = F.normalize(true_tail, dim=-1)
    out_n = F.normalize(outputs_k.view(B * K, D), dim=-1).view(B, K, D)
    cos = torch.bmm(out_n, true_n.unsqueeze(-1)).squeeze(-1)
    best_k = cos.argmax(dim=1)
    sorted_k = cos.argsort(dim=1, descending=True)
    top3 = sorted_k[:, :3]
    true_col = torch.arange(B, device=outputs_k.device)
    top1_acc = (best_k == 0).float().mean().item()
    top3_hits = (top3 == 0).any(dim=1).float().mean().item()
    return top1_acc, top3_hits


def distinct_k(outputs_k, threshold=0.7):
    B, K, D = outputs_k.shape
    out_n = F.normalize(outputs_k, dim=-1)
    sim = torch.bmm(out_n, out_n.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pw = (sim * mask)
    distinct_count = (pw < threshold).float().sum(dim=(1, 2)) / (K * (K - 1))
    return distinct_count.mean().item()


def evaluate(model, loader, tail_memory, tau=TAU):
    model.eval()
    metrics = {
        'cos_sim': [], 'pla': [], 'novelty': [], 'dci': [],
        'bb_cos_sim': [], 'bb_pla': [], 'bb_novelty': [], 'bb_dci': [],
        'top1_acc': [], 'top3_acc': [], 'distinct_k': [],
    }

    for head_idx, rel_idx, tail_idx, _, _ in tqdm(loader, desc='Evaluate'):
        head_idx = head_idx.to(DEVICE)
        rel_idx = rel_idx.to(DEVICE)
        tail_idx = tail_idx.to(DEVICE)

        with torch.no_grad():
            rel_vec, head_emb, rel_emb = model.backbone.encode_relation(
                head_idx, rel_idx
            )
            true_tail = model.backbone.word_embed(tail_idx)

            bb_outputs = backbone_baseline(model, rel_vec, k=K_OUTPUTS)
            bb_cos = cosine_pairwise_sim(bb_outputs)
            bb_pla = plausibility_score(bb_outputs, true_tail)
            bb_nov = novelty_vs_memory(bb_outputs, tail_memory)

            outputs_k, z_k, _ = model.forward_divergent(head_emb, rel_vec, rel_emb)
            pr_cos = cosine_pairwise_sim(outputs_k)
            pr_pla = plausibility_score(outputs_k, true_tail)
            pr_nov = novelty_vs_memory(outputs_k, tail_memory)
            top1, top3 = analogy_accuracy(outputs_k, true_tail)
            dk = distinct_k(outputs_k)

        div_q_bb = max(0.0, min(1.0, 1.0 - abs(bb_cos - tau)))
        div_q_pr = max(0.0, min(1.0, 1.0 - abs(pr_cos - tau)))
        dci_bb = (bb_pla * div_q_bb * bb_nov) ** (1 / 3) if bb_pla > 0 and bb_nov > 0 else 0.0
        dci_pr = (pr_pla * div_q_pr * pr_nov) ** (1 / 3) if pr_pla > 0 and pr_nov > 0 else 0.0

        for k, v in [
            ('cos_sim', pr_cos), ('pla', pr_pla), ('novelty', pr_nov), ('dci', dci_pr),
            ('bb_cos_sim', bb_cos), ('bb_pla', bb_pla),
            ('bb_novelty', bb_nov), ('bb_dci', dci_bb),
            ('top1_acc', top1), ('top3_acc', top3), ('distinct_k', dk),
        ]:
            metrics[k].append(v)

    return {k: np.mean(v) for k, v in metrics.items()}


def random_baseline(model, eval_loader, tail_memory):
    results = {}
    for scale in [0.05, 0.1, 0.3, 0.5]:
        print(f'\n--- Random baseline: scale={scale} ---')

        def fake_probe_forward(self, head_emb, rel_vec, rel_emb):
            B, D = rel_vec.shape
            transport = torch.randn(B, K_OUTPUTS, D, device=DEVICE) * scale
            z_k = rel_vec.unsqueeze(1) + transport
            B2, K2, D2 = z_k.shape
            z_flat = z_k.reshape(B2 * K2, D2)
            zeros = [
                torch.zeros(B2 * K2, HIDDEN_DIM, device=DEVICE)
                for _ in range(4)
            ]
            return z_k, transport, zeros

        from experiments.analogy.config import HIDDEN_DIM
        orig = model.probe.forward
        model.probe.forward = fake_probe_forward.__get__(model.probe, type(model.probe))
        metrics = evaluate(model, eval_loader, tail_memory)
        model.probe.forward = orig
        results[f'random_{scale}'] = metrics
        print(f'  DCI={metrics["dci"]:.4f} cos_sim={metrics["cos_sim"]:.4f}')
    return results


def ablation_experiment(model, train_loader, eval_loader, tail_memory, embeddings):
    configs = [
        ('λ_div=0', (0.0, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY)),
        ('λ_pla=0', (LAMBDA_DIVERSITY, 0.0, LAMBDA_NOVELTY)),
        ('λ_nov=0', (LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, 0.0)),
        ('full', (LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY)),
    ]
    results = {}
    for label, lambdas in configs:
        print(f'\n--- Ablation: {label} ---')
        m = AnalogyCBDP(embeddings=embeddings).to(DEVICE)
        m.backbone.load_state_dict(model.backbone.state_dict())
        m.freeze_backbone()
        opt = torch.optim.Adam(
            list(m.probe.parameters()) + list(m.gate.parameters()), lr=1e-3
        )
        for ep in range(10):
            for head_idx, rel_idx, tail_idx, _, _ in train_loader:
                head_idx = head_idx.to(DEVICE)
                rel_idx = rel_idx.to(DEVICE)
                tail_idx = tail_idx.to(DEVICE)
                with torch.no_grad():
                    rv, he, re = m.backbone.encode_relation(head_idx, rel_idx)
                    tt = m.backbone.word_embed(tail_idx)
                ok, zk, _ = m.forward_divergent(he, rv, re)
                ld = diversity_loss(zk)
                lp = plausibility_loss(ok, tt)
                ln = m.novelty_loss(ok)
                total = lambdas[0] * (ld - 0.7) ** 2 + lambdas[1] * lp + lambdas[2] * ln
                opt.zero_grad()
                total.backward()
                opt.step()
        metrics = evaluate(m, eval_loader, tail_memory)
        results[label] = metrics
        print(f'  DCI={metrics["dci"]:.4f} cos_sim={metrics["cos_sim"]:.4f}')
    return results


def per_relation_evaluate(model, test_ds, tail_memory):
    from torch.utils.data import DataLoader
    from experiments.analogy.dataset import collate_analogy

    rel_triples = {rel: [] for rel in RELATION_TYPES}
    for i, (h, rel, t) in enumerate(test_ds.triples):
        rel_triples[rel].append(i)

    rel_results = {}
    print('\n--- Per-Relation Evaluation ---')
    for rel_name, indices in rel_triples.items():
        if len(indices) == 0:
            continue
        subset = torch.utils.data.Subset(test_ds, indices)
        loader = DataLoader(
            subset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_analogy, drop_last=False,
        )
        metrics = evaluate(model, loader, tail_memory)
        rel_results[rel_name] = {
            'count': len(indices),
            'dci': round(metrics['dci'], 4),
            'cos_sim': round(metrics['cos_sim'], 4),
            'pla': round(metrics['pla'], 4),
            'novelty': round(metrics['novelty'], 4),
            'top1_acc': round(metrics['top1_acc'], 4),
            'top3_acc': round(metrics['top3_acc'], 4),
            'distinct_k': round(metrics['distinct_k'], 4),
        }
        print(f'  {rel_name:12s} (n={len(indices):5d}): '
              f'DCI={metrics["dci"]:.4f} '
              f'Top-3={metrics["top3_acc"]:.4f} '
              f'Dist-K={metrics["distinct_k"]:.4f}')
    return rel_results


def main():
    train_loader, test_loader, embeddings, vocab, train_ds, test_ds = \
        get_dataloaders(BATCH_SIZE)

    os.makedirs(RESULT_DIR, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, 'evaluate.log')
    tee = setup_logger(log_path)

    model = AnalogyCBDP(embeddings=embeddings).to(DEVICE)
    ckpt = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if not os.path.exists(ckpt):
        print('ERROR: No CBDP checkpoint. Run stage1 and stage2 first.')
        return
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print('Loaded CBDP model.')

    stats = model.parameter_stats()
    print(f'Probe: {stats["probe"]:,} params ({stats["probe_ratio"]:.2f}% of backbone)')

    tail_memory = build_tail_memory(model, train_loader)

    print('\n' + '=' * 60)
    print('Phase 3: WordNet Analogy Evaluation')
    print('=' * 60)

    print('\n--- Train Pairs ---')
    train_metrics = evaluate(model, train_loader, tail_memory)
    print(f'Backbone: cos={train_metrics["bb_cos_sim"]:.4f} '
          f'pla={train_metrics["bb_pla"]:.4f} '
          f'nov={train_metrics["bb_novelty"]:.4f} '
          f'DCI={train_metrics["bb_dci"]:.4f}')
    print(f'CBDP:     cos={train_metrics["cos_sim"]:.4f} '
          f'pla={train_metrics["pla"]:.4f} '
          f'nov={train_metrics["novelty"]:.4f} '
          f'DCI={train_metrics["dci"]:.4f}')
    print(f'Top-1 Acc={train_metrics["top1_acc"]:.4f} '
          f'Top-3 Acc={train_metrics["top3_acc"]:.4f} '
          f'Distinct-K={train_metrics["distinct_k"]:.4f}')

    print('\n--- Test Pairs ---')
    test_metrics = evaluate(model, test_loader, tail_memory)
    print(f'Backbone: cos={test_metrics["bb_cos_sim"]:.4f} '
          f'pla={test_metrics["bb_pla"]:.4f} '
          f'nov={test_metrics["bb_novelty"]:.4f} '
          f'DCI={test_metrics["bb_dci"]:.4f}')
    print(f'CBDP:     cos={test_metrics["cos_sim"]:.4f} '
          f'pla={test_metrics["pla"]:.4f} '
          f'nov={test_metrics["novelty"]:.4f} '
          f'DCI={test_metrics["dci"]:.4f}')
    print(f'Top-1 Acc={test_metrics["top1_acc"]:.4f} '
          f'Top-3 Acc={test_metrics["top3_acc"]:.4f} '
          f'Distinct-K={test_metrics["distinct_k"]:.4f}')

    dci_gap = abs(test_metrics["dci"] - train_metrics["dci"])
    print(f'DCI gap (train - test): {dci_gap:.4f}')

    print('\n--- Random Baseline ---')
    rand_results = random_baseline(model, test_loader, tail_memory)

    print('\n--- Ablation (test pairs) ---')
    abl_results = ablation_experiment(
        model, train_loader, test_loader, tail_memory, embeddings
    )

    print('\n--- Per-Relation Breakdown ---')
    rel_results = per_relation_evaluate(model, test_ds, tail_memory)

    all_results = {
        'train': train_metrics,
        'test': test_metrics,
        'dci_gap': dci_gap,
        'random_baseline': rand_results,
        'ablation': abl_results,
        'per_relation': rel_results,
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    result_path = os.path.join(RESULT_DIR, 'phase3_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f'\nResults saved to {result_path}')
    tee.close()
    return all_results


if __name__ == '__main__':
    main()
