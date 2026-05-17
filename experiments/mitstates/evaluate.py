import json, os, sys, random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.mitstates.config import (
    IMAGE_SIZE, BATCH_SIZE, K_OUTPUTS, DEVICE, TAU,
    LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY,
)
from experiments.mitstates.dataset import get_dataloaders
from experiments.mitstates.backbone import (
    MITStatesCBDP, diversity_loss, plausibility_loss,
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')

try:
    import open_clip
    HAS_CLIP = True
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model = clip_model.to(DEVICE).eval()
except Exception:
    HAS_CLIP = False
    print('WARNING: open_clip not available, CLIP metrics will be 0')


def clip_score_image_text(image_tensor, text):
    if not HAS_CLIP:
        return 0.0
    img_224 = F.interpolate(image_tensor, size=(224, 224), mode='bilinear')
    img_norm = F.normalize(img_224, dim=1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)
    img_norm = (img_norm - mean) / std
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_norm)
        text_tok = clip_tokenizer([text]).to(DEVICE)
        text_feat = clip_model.encode_text(text_tok)
        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        sim = (img_feat * text_feat).sum(dim=-1)
    return sim.mean().item()


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


def evaluate(model, loader, image_memory, tau=TAU):
    model.eval()
    metrics = {
        'cos_sim': [], 'pla': [], 'novelty': [], 'dci': [],
        'backbone_cos_sim': [], 'backbone_pla': [], 'backbone_novelty': [], 'backbone_dci': [],
        'clip_combo': [], 'clip_attr': [], 'clip_obj': [],
    }

    for x, attr_idx, obj_idx, attr_name, obj_name in tqdm(loader, desc='Evaluate'):
        x = x.to(DEVICE)

        with torch.no_grad():
            _, obj_emb, attr_emb, _, _, _, _ = model.backbone(
                x, attr_idx.to(DEVICE), obj_idx.to(DEVICE)
            )

            bb_outputs = backbone_baseline(model, obj_emb, k=K_OUTPUTS)
            bb_cos = cosine_diversity_value(bb_outputs)
            bb_pla = plausibility_mse_score(bb_outputs, x)
            bb_nov = compute_novelty_score(bb_outputs, image_memory)

            outputs_k, _, _ = model.forward_divergent(obj_emb, attr_emb)
            pr_cos = cosine_diversity_value(outputs_k)
            pr_pla = plausibility_mse_score(outputs_k, x)
            pr_nov = compute_novelty_score(outputs_k, image_memory)

            best_out = outputs_k[:, 0]
            clip_combo = clip_score_image_text(best_out, f'a {attr_name[0]} {obj_name[0]}')
            clip_attr_val = clip_score_image_text(best_out, attr_name[0])
            clip_obj_val = clip_score_image_text(best_out, obj_name[0])

        div_q_bb = max(0.0, min(1.0, 1.0 - abs(bb_cos - tau)))
        div_q_pr = max(0.0, min(1.0, 1.0 - abs(pr_cos - tau)))
        dci_bb = (bb_pla * div_q_bb * bb_nov) ** (1 / 3)
        dci_pr = (pr_pla * div_q_pr * pr_nov) ** (1 / 3)

        for k, v in [
            ('cos_sim', pr_cos), ('pla', pr_pla), ('novelty', pr_nov), ('dci', dci_pr),
            ('backbone_cos_sim', bb_cos), ('backbone_pla', bb_pla),
            ('backbone_novelty', bb_nov), ('backbone_dci', dci_bb),
            ('clip_combo', clip_combo), ('clip_attr', clip_attr_val), ('clip_obj', clip_obj_val),
        ]:
            metrics[k].append(v)

    return {k: np.mean(v) for k, v in metrics.items()}


def shape_color_retention(model, loader):
    model.eval()
    shape_ret = []
    color_ret = []
    with torch.no_grad():
        for x, attr_idx, obj_idx, _, _ in tqdm(loader, desc='Retention'):
            x = x.to(DEVICE)
            _, obj_emb, attr_emb, _, _, _, _ = model.backbone(
                x, attr_idx.to(DEVICE), obj_idx.to(DEVICE)
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
                shape_ret.append(F.cosine_similarity(obj_orig, obj_dec).item())
                color_ret.append(F.cosine_similarity(attr_orig, attr_dec).item())

    return np.mean(shape_ret) if shape_ret else 0.0, np.mean(color_ret) if color_ret else 0.0


def random_baseline(model, eval_loader, image_memory):
    results = {}
    for scale in [0.05, 0.1, 0.3, 0.5]:
        print(f'\n--- Random baseline: scale={scale} ---')

        def fake_probe_forward(self, obj_emb, attr_emb):
            B, D = obj_emb.shape
            transport = torch.randn(B, K_OUTPUTS, D, device=DEVICE) * scale
            z_k = obj_emb.unsqueeze(1) + transport + attr_emb.unsqueeze(1)
            zeros = [torch.zeros(B * K_OUTPUTS, c, 1, 1, device=DEVICE)
                     for c in [256, 128, 64, 32]]
            return z_k, transport, zeros

        orig = model.probe.forward
        model.probe.forward = fake_probe_forward.__get__(model.probe, type(model.probe))
        metrics = evaluate(model, eval_loader, image_memory)
        model.probe.forward = orig
        results[f'random_{scale}'] = metrics
        print(f'  DCI={metrics["dci"]:.4f} cos_sim={metrics["cos_sim"]:.4f}')
    return results


def ablation_experiment(model, train_loader, eval_loader, image_memory):
    import copy
    configs = [
        ('λ_div=0', (0.0, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY)),
        ('λ_pla=0', (LAMBDA_DIVERSITY, 0.0, LAMBDA_NOVELTY)),
        ('λ_nov=0', (LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, 0.0)),
        ('full', (LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY)),
    ]
    results = {}
    for label, lambdas in configs:
        print(f'\n--- Ablation: {label} ---')
        m = copy.deepcopy(model).to(DEVICE)
        m.freeze_backbone()
        opt = torch.optim.Adam(list(m.probe.parameters()) + list(m.gate.parameters()), lr=1e-3)
        for ep in range(10):
            for x, attr_idx, obj_idx, _, _ in train_loader:
                x = x.to(DEVICE)
                with torch.no_grad():
                    _, obj_e, attr_e, _, _, _, _ = m.backbone(
                        x, attr_idx.to(DEVICE), obj_idx.to(DEVICE)
                    )
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


def main():
    train_loader, test_loader, train_ds, test_ds, num_attrs, num_objs = get_dataloaders(BATCH_SIZE)

    model = MITStatesCBDP(num_objects=num_objs, num_attrs=num_attrs).to(DEVICE)
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
    print('Phase 2b: MIT-States Concept Combination Evaluation')
    print('=' * 60)

    print('\n--- Train Pairs (seen attr+obj) ---')
    train_metrics = evaluate(model, train_loader, image_memory)
    print(f'Backbone: cos={train_metrics["backbone_cos_sim"]:.4f} '
          f'pla={train_metrics["backbone_pla"]:.4f} '
          f'nov={train_metrics["backbone_novelty"]:.4f} '
          f'DCI={train_metrics["backbone_dci"]:.4f}')
    print(f'CBDP:     cos={train_metrics["cos_sim"]:.4f} '
          f'pla={train_metrics["pla"]:.4f} '
          f'nov={train_metrics["novelty"]:.4f} '
          f'DCI={train_metrics["dci"]:.4f}')

    print('\n--- Test Pairs (unseen attr+obj combinations) ---')
    test_metrics = evaluate(model, test_loader, image_memory)
    print(f'Backbone: cos={test_metrics["backbone_cos_sim"]:.4f} '
          f'pla={test_metrics["backbone_pla"]:.4f} '
          f'nov={test_metrics["backbone_novelty"]:.4f} '
          f'DCI={test_metrics["backbone_dci"]:.4f}')
    print(f'CBDP:     cos={test_metrics["cos_sim"]:.4f} '
          f'pla={test_metrics["pla"]:.4f} '
          f'nov={test_metrics["novelty"]:.4f} '
          f'DCI={test_metrics["dci"]:.4f}')

    dci_gap = abs(test_metrics["dci"] - train_metrics["dci"])
    print(f'DCI gap (train - test): {dci_gap:.4f}')

    print('\n--- CLIP Scores (test) ---')
    print(f'CLIP combo: {train_metrics["clip_combo"]:.4f} (train) / '
          f'{test_metrics["clip_combo"]:.4f} (test)')
    print(f'CLIP attr:  {train_metrics["clip_attr"]:.4f} (train) / '
          f'{test_metrics["clip_attr"]:.4f} (test)')
    print(f'CLIP obj:   {train_metrics["clip_obj"]:.4f} (train) / '
          f'{test_metrics["clip_obj"]:.4f} (test)')

    print('\n--- Shape/Color Retention ---')
    train_s_ret, train_c_ret = shape_color_retention(model, train_loader)
    test_s_ret, test_c_ret = shape_color_retention(model, test_loader)
    print(f'Train: obj_ret={train_s_ret:.4f} attr_ret={train_c_ret:.4f}')
    print(f'Test:  obj_ret={test_s_ret:.4f} attr_ret={test_c_ret:.4f}')

    print('\n--- Random Baseline ---')
    rand_results = random_baseline(model, test_loader, image_memory)

    print('\n--- Ablation (test pairs) ---')
    abl_results = ablation_experiment(model, train_loader, test_loader, image_memory)

    all_results = {
        'train': train_metrics,
        'test': test_metrics,
        'dci_gap': dci_gap,
        'retention': {
            'train_obj_ret': train_s_ret,
            'train_attr_ret': train_c_ret,
            'test_obj_ret': test_s_ret,
            'test_attr_ret': test_c_ret,
        },
        'random_baseline': {k: v for k, v in rand_results.items()},
        'ablation': {k: v for k, v in abl_results.items()},
    }
    path = os.path.join(RESULT_DIR, 'phase2b_results.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f'\nResults saved to {path}')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
