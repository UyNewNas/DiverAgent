import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.analogy.config import DEVICE, K_OUTPUTS
from experiments.analogy.dataset import get_dataloaders
from experiments.analogy.backbone import AnalogyCBDP
from experiments.analogy.evaluate import evaluate, build_tail_memory

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def nearest_words(embeddings, vocab, query_emb, top_k=5):
    inv_vocab = {i: w for w, i in vocab.items()}
    query_n = F.normalize(query_emb.unsqueeze(0), dim=-1)
    emb_n = F.normalize(embeddings, dim=-1)
    sim = torch.mm(query_n, emb_n.t()).squeeze(0)
    top_sim, top_idx = sim.topk(top_k + 1)
    results = []
    for i in range(top_k + 1):
        idx = top_idx[i].item()
        word = inv_vocab[idx]
        results.append((word, top_sim[i].item()))
    return results[:top_k]


def visualize():
    train_loader, test_loader, embeddings, vocab, train_ds, test_ds = \
        get_dataloaders(256)

    model = AnalogyCBDP(embeddings=embeddings).to(DEVICE)
    ckpt = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if not os.path.exists(ckpt):
        print('ERROR: No CBDP checkpoint found.')
        return
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    tail_memory = build_tail_memory(model, train_loader)
    inv_vocab = {i: w for w, i in vocab.items()}
    embeddings = embeddings.to(DEVICE)

    sample_cases = []
    with torch.no_grad():
        for head_idx, rel_idx, tail_idx, head_words, tail_words in test_loader:
            head_idx = head_idx.to(DEVICE)
            rel_idx = rel_idx.to(DEVICE)
            tail_idx = tail_idx.to(DEVICE)

            rel_vec, head_emb, rel_emb = model.backbone.encode_relation(
                head_idx, rel_idx
            )
            true_tail = model.backbone.word_embed(tail_idx)
            outputs_k, z_k, _ = model.forward_divergent(head_emb, rel_vec, rel_emb)

            rel_names = [test_ds.triples[i][1] for i in range(
                len(sample_cases), min(len(sample_cases) + 10, len(test_ds))
            )]

            for b in range(min(5, head_idx.size(0))):
                hw = head_words[b]
                tw = tail_words[b]
                rn = rel_names[b] if b < len(rel_names) else 'unknown'

                true_t = true_tail[b]
                outs = outputs_k[b]

                cos_vs_true = F.cosine_similarity(outs, true_t.unsqueeze(0)).cpu().tolist()
                best_k = int(np.argmax(cos_vs_true))
                dci_val = None

                case = {
                    'head': hw,
                    'relation': rn,
                    'true_tail': tw,
                    'cos_to_true': [round(c, 4) for c in cos_vs_true],
                    'best_k': best_k,
                    'candidates': [],
                }

                for k in range(K_OUTPUTS):
                    nw = nearest_words(embeddings, vocab, outs[k], top_k=3)
                    case['candidates'].append({
                        'k': k,
                        'cos_to_true': round(cos_vs_true[k], 4),
                        'nearest_words': [(w, round(s, 4)) for w, s in nw],
                    })

                sample_cases.append(case)
                if len(sample_cases) >= 10:
                    break
            if len(sample_cases) >= 10:
                break

    print('\n' + '=' * 80)
    print('Phase 3: Analogy Visualization')
    print('=' * 80)

    for i, case in enumerate(sample_cases):
        print(f'\n--- Case {i+1}: {case["head"]} --[{case["relation"]}]--> ? ---')
        print(f'  True tail: {case["true_tail"]}')
        print(f'  Best matching K: k={case["best_k"]}')
        print(f'  Cos similarity to true tail per output:')
        for k, c in enumerate(case['cos_to_true']):
            marker = ' <<<' if k == case['best_k'] else ''
            print(f'    k={k}: cos={c:.4f}{marker}')
        print(f'  Nearest words for each candidate:')
        for cand in case['candidates']:
            words_str = ', '.join([f'{w}({s:.3f})' for w, s in cand['nearest_words']])
            print(f'    k={cand["k"]} (cos={cand["cos_to_true"]:.4f}): {words_str}')

    metrics = evaluate(model, test_loader, tail_memory)
    dci_scores = []
    with torch.no_grad():
        for head_idx, rel_idx, tail_idx, _, _ in test_loader:
            head_idx = head_idx.to(DEVICE)
            rel_idx = rel_idx.to(DEVICE)
            tail_idx = tail_idx.to(DEVICE)
            rel_vec, head_emb, rel_emb = model.backbone.encode_relation(
                head_idx, rel_idx
            )
            true_tail = model.backbone.word_embed(tail_idx)
            outputs_k, z_k, _ = model.forward_divergent(head_emb, rel_vec, rel_emb)

            cos_v = cosine_pairwise_sim(outputs_k)
            pla_v = (1.0 - F.mse_loss(outputs_k, true_tail.unsqueeze(1).expand_as(outputs_k),
                                      reduction='none').mean(dim=-1).min(dim=1)[0].mean().item())
            mem_n = F.normalize(tail_memory.to(DEVICE), dim=-1)
            out_n = F.normalize(outputs_k.view(-1, outputs_k.size(-1)), dim=-1)
            sim_mem = torch.mm(out_n, mem_n.t())
            nov_v = (1.0 - sim_mem.max(dim=1)[0].mean()).item()
            div_q = max(0.0, min(1.0, 1.0 - abs(cos_v - 0.3)))
            dci = (pla_v * div_q * nov_v) ** (1 / 3) if pla_v > 0 and nov_v > 0 else 0.0
            dci_scores.append(dci)

    dci_scores = np.array(dci_scores)
    worst_idx = np.argsort(dci_scores)[:10]

    print('\n' + '=' * 80)
    print('Failure Case Analysis (Lowest DCI)')
    print('=' * 80)

    fail_cases = []
    for idx in worst_idx:
        head_w = test_ds.triples[idx][0]
        rel_w = test_ds.triples[idx][1]
        tail_w = test_ds.triples[idx][2]
        fail_cases.append({
            'head': head_w,
            'relation': rel_w,
            'true_tail': tail_w,
            'dci': round(dci_scores[idx], 4),
        })

    for i, fc in enumerate(fail_cases):
        print(f'  {i+1}. {fc["head"]} --[{fc["relation"]}]--> {fc["true_tail"]} '
              f'(DCI={fc["dci"]})')

    # patterns
    rel_counts = {}
    for fc in fail_cases:
        rel = fc['relation']
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    print(f'\n  Failure modes by relation type: {rel_counts}')


def cosine_pairwise_sim(outputs_k):
    B, K, D = outputs_k.shape
    flat_n = F.normalize(outputs_k.view(B * K, D), dim=-1)
    flat_n = flat_n.view(B, K, D)
    sim = torch.bmm(flat_n, flat_n.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pw = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return pw.mean().item()


if __name__ == '__main__':
    visualize()
