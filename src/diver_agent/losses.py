import torch
import torch.nn as nn
import torch.nn.functional as F


def diversity_loss(outputs_k):
    batch, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(batch, K, -1)
    flat = F.normalize(flat, dim=-1)
    sim_matrix = torch.bmm(flat, flat.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pairwise_sim = (sim_matrix * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return (1 - pairwise_sim).mean()


def plausibility_loss(outputs_k, target):
    target_expanded = target.unsqueeze(1).expand_as(outputs_k)
    mse_per_sample = F.mse_loss(outputs_k, target_expanded, reduction='none').view(
        outputs_k.size(0), outputs_k.size(1), -1
    ).mean(dim=-1)
    return mse_per_sample.min(dim=1)[0].mean()


class NoveltyLoss(nn.Module):
    def __init__(self, memory_size=5000, threshold=0.95):
        super().__init__()
        self.memory = None
        self.memory_ptr = 0
        self.memory_size = memory_size
        self.is_full = False
        self.threshold = threshold

    def update_memory(self, embeddings):
        with torch.no_grad():
            if self.memory is None:
                self.memory = torch.zeros(
                    self.memory_size, embeddings.size(1), device=embeddings.device
                )
            batch_size = embeddings.size(0)
            end_idx = min(self.memory_ptr + batch_size, self.memory_size)
            self.memory[self.memory_ptr:end_idx] = embeddings[: end_idx - self.memory_ptr]
            if end_idx < self.memory_ptr + batch_size:
                remaining = batch_size - (end_idx - self.memory_ptr)
                self.memory[:remaining] = embeddings[end_idx - self.memory_ptr :]
                self.is_full = True
            self.memory_ptr = (self.memory_ptr + batch_size) % self.memory_size

    def forward(self, embeddings_k):
        if self.memory is None or (not self.is_full and self.memory_ptr < 10):
            return torch.tensor(0.0, device=embeddings_k.device, requires_grad=True)

        memory_pool = self.memory if self.is_full else self.memory[: self.memory_ptr]
        batch, K, dim = embeddings_k.shape
        flat_query = embeddings_k.view(batch * K, dim)
        flat_query_norm = F.normalize(flat_query, dim=-1)
        memory_norm = F.normalize(memory_pool, dim=-1)

        sim = torch.mm(flat_query_norm, memory_norm.t())
        max_sim, _ = sim.max(dim=1)
        over_threshold = (max_sim > self.threshold).float()
        coverage = over_threshold.view(batch, K).mean(dim=1)
        return coverage.mean()


def divergent_loss(outputs_k, target, embeddings_k, novelty_loss_fn,
                   lambda_diversity=1.0, lambda_plausibility=0.8,
                   lambda_novelty=0.5):
    l_div = diversity_loss(outputs_k)
    l_pla = plausibility_loss(outputs_k, target)
    l_nov = novelty_loss_fn(embeddings_k)

    total = -lambda_diversity * l_div + lambda_plausibility * l_pla + lambda_novelty * l_nov

    return total, {
        'diversity': l_div.item(),
        'plausibility': l_pla.item(),
        'novelty': l_nov.item() if isinstance(l_nov, torch.Tensor) else l_nov,
        'total': total.item(),
    }
