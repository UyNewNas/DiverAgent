import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    EMBED_DIM, HIDDEN_DIM, RELATION_DIM, K_OUTPUTS, NUM_RELATIONS,
    LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY,
    NOVELTY_THRESHOLD, NOVELTY_MEMORY_SIZE,
)


class AnalogyEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, relation_dim=RELATION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + relation_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, relation_dim),
        )

    def forward(self, head_emb, rel_emb):
        x = torch.cat([head_emb, rel_emb], dim=-1)
        return self.net(x)


class AnalogyDecoder(nn.Module):
    def __init__(self, input_dim=RELATION_DIM, hidden_dim=HIDDEN_DIM, output_dim=EMBED_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, injections=None):
        h = self.fc1(z)
        if injections is not None:
            h = h + injections[0]
        h = F.relu(h)
        h = self.fc2(h)
        if injections is not None:
            h = h + injections[1]
        h = F.relu(h)
        h = self.fc3(h)
        if injections is not None:
            h = h + injections[2]
        h = F.relu(h)
        h = self.fc4(h)
        if injections is not None:
            h = h + injections[3]
        h = F.relu(h)
        h = self.fc5(h)
        return h


class AnalogyBackbone(nn.Module):
    def __init__(self, num_relations=NUM_RELATIONS, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, relation_dim=RELATION_DIM,
                 embeddings=None):
        super().__init__()
        self.word_embed = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.relation_embed = nn.Embedding(num_relations, relation_dim)
        self.encoder = AnalogyEncoder(embed_dim, hidden_dim, relation_dim)
        self.decoder = AnalogyDecoder(relation_dim, hidden_dim, embed_dim)
        self.embed_dim = embed_dim
        self.relation_dim = relation_dim

    def forward(self, head_idx, rel_idx):
        head_emb = self.word_embed(head_idx)
        rel_emb = self.relation_embed(rel_idx)
        rel_vec = self.encoder(head_emb, rel_emb)
        tail_pred = self.decoder(rel_vec)
        align_loss = F.mse_loss(rel_vec, rel_emb)
        return tail_pred, head_emb, rel_emb, rel_vec, align_loss

    def encode_relation(self, head_idx, rel_idx):
        head_emb = self.word_embed(head_idx)
        rel_emb = self.relation_embed(rel_idx)
        rel_vec = self.encoder(head_emb, rel_emb)
        return rel_vec, head_emb, rel_emb

    def decode(self, z, injections=None):
        return self.decoder(z, injections)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())


class DirectionalProbe(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, relation_dim=RELATION_DIM,
                 hidden_dim=HIDDEN_DIM, k_outputs=K_OUTPUTS):
        super().__init__()
        self.k = k_outputs
        self.transport_mlp = nn.Sequential(
            nn.Linear(embed_dim + relation_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, relation_dim * k_outputs),
        )
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        self.inj_mlp_0 = nn.Linear(relation_dim, HIDDEN_DIM)
        self.inj_mlp_1 = nn.Linear(relation_dim, HIDDEN_DIM)
        self.inj_mlp_2 = nn.Linear(relation_dim, HIDDEN_DIM)
        self.inj_mlp_3 = nn.Linear(relation_dim, HIDDEN_DIM)

    def forward(self, head_emb, rel_vec, rel_emb):
        combined = torch.cat([head_emb, rel_vec], dim=-1)
        transport = self.transport_mlp(combined)
        transport = transport.view(head_emb.size(0), self.k, -1)
        transport = self.noise_scale * transport
        z_k = rel_vec.unsqueeze(1) + transport
        B, K, D = z_k.shape
        z_flat = z_k.reshape(B * K, D)
        injections = [
            self.inj_mlp_0(z_flat),
            self.inj_mlp_1(z_flat),
            self.inj_mlp_2(z_flat),
            self.inj_mlp_3(z_flat),
        ]
        return z_k, transport, injections

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DivergenceGate(nn.Module):
    def __init__(self, dim=RELATION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def diversity_loss(z_k):
    B, K, D = z_k.shape
    z_n = F.normalize(z_k, dim=-1)
    sim = torch.bmm(z_n, z_n.transpose(1, 2))
    mask = 1 - torch.eye(K, device=z_k.device).unsqueeze(0)
    pw = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return (1 - pw).mean()


def plausibility_loss(outputs_k, target):
    target_e = target.unsqueeze(1).expand_as(outputs_k)
    mse = F.mse_loss(outputs_k, target_e, reduction='none').mean(dim=-1)
    return mse.min(dim=1)[0].mean()


class NoveltyLoss(nn.Module):
    def __init__(self, memory_size=NOVELTY_MEMORY_SIZE, threshold=NOVELTY_THRESHOLD):
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
            batch_sz = embeddings.size(0)
            end_idx = min(self.memory_ptr + batch_sz, self.memory_size)
            self.memory[self.memory_ptr:end_idx] = embeddings[:end_idx - self.memory_ptr]
            if end_idx < self.memory_ptr + batch_sz:
                remaining = batch_sz - (end_idx - self.memory_ptr)
                self.memory[:remaining] = embeddings[end_idx - self.memory_ptr:]
                self.is_full = True
            self.memory_ptr = (self.memory_ptr + batch_sz) % self.memory_size

    def forward(self, embeddings_k):
        if self.memory is None or (not self.is_full and self.memory_ptr < 10):
            return torch.tensor(0.0, device=embeddings_k.device, requires_grad=True)
        mem_pool = self.memory if self.is_full else self.memory[:self.memory_ptr]
        B, K, D = embeddings_k.shape
        flat_q = embeddings_k.view(B * K, D)
        flat_q_n = F.normalize(flat_q, dim=-1)
        mem_n = F.normalize(mem_pool, dim=-1)
        sim = torch.mm(flat_q_n, mem_n.t())
        max_sim, _ = sim.max(dim=1)
        over = (max_sim > self.threshold).float()
        coverage = over.view(B, K).mean(dim=1)
        return coverage.mean()


class AnalogyCBDP(nn.Module):
    def __init__(self, num_relations=NUM_RELATIONS, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, relation_dim=RELATION_DIM,
                 k_outputs=K_OUTPUTS, embeddings=None):
        super().__init__()
        self.backbone = AnalogyBackbone(num_relations, embed_dim, hidden_dim,
                                        relation_dim, embeddings)
        self.probe = DirectionalProbe(embed_dim, relation_dim, hidden_dim, k_outputs)
        self.gate = DivergenceGate(relation_dim)
        self.novelty_loss = NoveltyLoss()
        self.relation_dim = relation_dim
        self.k_outputs = k_outputs

    def forward_convergent(self, head_idx, rel_idx):
        return self.backbone(head_idx, rel_idx)

    def forward_divergent(self, head_emb, rel_vec, rel_emb):
        z_k, transport, injections = self.probe(head_emb, rel_vec, rel_emb)
        B, K, D = z_k.shape
        z_flat = z_k.reshape(B * K, D)
        outputs_flat = self.backbone.decode(z_flat, injections)
        outputs_k = outputs_flat.view(B, K, EMBED_DIM)
        return outputs_k, z_k, transport

    def compute_divergent_loss(self, outputs_k, target, z_k):
        l_div = diversity_loss(z_k)
        l_pla = plausibility_loss(outputs_k, target)
        l_nov = self.novelty_loss(outputs_k)
        total = (LAMBDA_DIVERSITY * (l_div - 0.7) ** 2
                 + LAMBDA_PLAUSIBILITY * l_pla
                 + LAMBDA_NOVELTY * l_nov)
        return total, {
            'diversity': l_div.item(),
            'plausibility': l_pla.item(),
            'novelty': l_nov.item() if isinstance(l_nov, torch.Tensor) else l_nov,
            'total': total.item(),
        }

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def parameter_stats(self):
        bb = self.backbone.count_all_parameters()
        bb_train = self.backbone.count_parameters()
        pr = self.probe.count_parameters()
        ga = self.gate.count_parameters()
        return {
            'backbone': bb, 'probe': pr, 'gate': ga,
            'backbone_trainable': bb_train,
            'total': bb + pr + ga,
            'probe_ratio': pr / bb * 100 if bb > 0 else 0,
        }
