import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    IMAGE_SIZE, LATENT_DIM, OBJECT_DIM, ATTR_DIM, K_OUTPUTS,
    LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY,
    NUM_SHAPES, NUM_COLORS,
)


class SharedEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(512 * 4 * 4, feature_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ObjectProjector(nn.Module):
    def __init__(self, feature_dim=256, object_dim=OBJECT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, object_dim * 2),
            nn.ReLU(True),
            nn.Linear(object_dim * 2, object_dim),
        )

    def forward(self, feature):
        return self.net(feature)


class AttributeProjector(nn.Module):
    def __init__(self, feature_dim=256, attr_dim=ATTR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, attr_dim * 2),
            nn.ReLU(True),
            nn.Linear(attr_dim * 2, attr_dim),
        )

    def forward(self, feature):
        return self.net(feature)


class ConditionalDecoder(nn.Module):
    def __init__(self, in_dim=OBJECT_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.deconv0 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn0 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def forward(self, z, injections=None):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        if injections is not None:
            h = h + injections[0]
        h = self.deconv0(h)
        if injections is not None:
            h = h + injections[1]
        h = F.relu(self.bn0(h), inplace=True)
        h = self.deconv1(h)
        if injections is not None:
            h = h + injections[2]
        h = F.relu(self.bn1(h), inplace=True)
        h = self.deconv2(h)
        if injections is not None:
            h = h + injections[3]
        h = F.relu(self.bn2(h), inplace=True)
        h = self.deconv3(h)
        return torch.sigmoid(h)


class DSpritesBackbone(nn.Module):
    def __init__(self, feature_dim=256, object_dim=OBJECT_DIM, attr_dim=ATTR_DIM):
        super().__init__()
        self.encoder = SharedEncoder(feature_dim)
        self.object_proj = ObjectProjector(feature_dim, object_dim)
        self.attr_proj = AttributeProjector(feature_dim, attr_dim)
        self.object_embed = nn.Embedding(NUM_SHAPES, object_dim)
        self.attr_embed = nn.Embedding(NUM_COLORS, attr_dim)
        self.decoder = ConditionalDecoder(object_dim)
        self.object_dim = object_dim
        self.attr_dim = attr_dim

    def encode(self, x):
        feature = self.encoder(x)
        obj_emb = self.object_proj(feature)
        attr_emb = self.attr_proj(feature)
        obj_anchor = self.object_embed.weight
        attr_anchor = self.attr_embed.weight
        return obj_emb, attr_emb, obj_anchor, attr_anchor

    def decode(self, z, injections=None):
        return self.decoder(z, injections)

    def forward(self, x, shape_idx, color_idx):
        B = x.size(0)
        feature = self.encoder(x)
        obj_emb = self.object_proj(feature)
        attr_emb = self.attr_proj(feature)
        obj_anchor = self.object_embed(shape_idx)
        attr_anchor = self.attr_embed(color_idx)
        obj_loss = F.mse_loss(obj_emb, obj_anchor)
        attr_loss = F.mse_loss(attr_emb, attr_anchor)
        recon = self.decode(obj_emb)
        return recon, obj_emb, attr_emb, obj_anchor, attr_anchor, obj_loss, attr_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DirectionalProbe(nn.Module):
    def __init__(self, object_dim=OBJECT_DIM, attr_dim=ATTR_DIM,
                 k_outputs=K_OUTPUTS, hidden_dim=128):
        super().__init__()
        self.k = k_outputs
        self.transport_mlp = nn.Sequential(
            nn.Linear(object_dim + attr_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, object_dim * k_outputs),
        )
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        self.inj_mlp_0 = nn.Linear(object_dim, 256)
        self.inj_mlp_1 = nn.Linear(object_dim, 128)
        self.inj_mlp_2 = nn.Linear(object_dim, 64)
        self.inj_mlp_3 = nn.Linear(object_dim, 32)

    def forward(self, obj_emb, attr_emb):
        combined = torch.cat([obj_emb, attr_emb], dim=-1)
        transport = self.transport_mlp(combined)
        transport = transport.view(obj_emb.size(0), self.k, -1)
        transport = self.noise_scale * transport
        z_k = obj_emb.unsqueeze(1) + transport + attr_emb.unsqueeze(1)
        B, K, D = z_k.shape
        z_flat = z_k.reshape(B * K, D)
        injections = [
            self.inj_mlp_0(z_flat).view(B * K, 256, 1, 1),
            self.inj_mlp_1(z_flat).view(B * K, 128, 1, 1),
            self.inj_mlp_2(z_flat).view(B * K, 64, 1, 1),
            self.inj_mlp_3(z_flat).view(B * K, 32, 1, 1),
        ]
        return z_k, transport, injections

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DivergenceGate(nn.Module):
    def __init__(self, dim=OBJECT_DIM):
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


def diversity_loss(outputs_k):
    B, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(B, K, -1)
    flat_n = F.normalize(flat, dim=-1)
    sim = torch.bmm(flat_n, flat_n.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pw = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return (1 - pw).mean()


def plausibility_loss(outputs_k, target):
    target_expanded = target.unsqueeze(1).expand_as(outputs_k)
    mse = F.mse_loss(outputs_k, target_expanded, reduction='none').view(
        outputs_k.size(0), outputs_k.size(1), -1
    ).mean(dim=-1)
    return mse.min(dim=1)[0].mean()


class NoveltyLoss(nn.Module):
    def __init__(self, memory_size=3000, threshold=0.95):
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


class DSpritesCBDP(nn.Module):
    def __init__(self, feature_dim=256, object_dim=OBJECT_DIM, attr_dim=ATTR_DIM,
                 k_outputs=K_OUTPUTS):
        super().__init__()
        self.backbone = DSpritesBackbone(feature_dim, object_dim, attr_dim)
        self.probe = DirectionalProbe(object_dim, attr_dim, k_outputs)
        self.gate = DivergenceGate(object_dim)
        self.novelty_loss = NoveltyLoss()
        self.object_dim = object_dim
        self.k_outputs = k_outputs

    def forward_convergent(self, x, shape_idx, color_idx):
        return self.backbone(x, shape_idx, color_idx)

    def forward_divergent(self, obj_emb, attr_emb):
        z_k, transport, injections = self.probe(obj_emb, attr_emb)
        B, K, D = z_k.shape
        z_flat = z_k.reshape(B * K, D)
        outputs_flat = self.backbone.decode(z_flat, injections)
        C = 3
        outputs_k = outputs_flat.view(B, K, C, IMAGE_SIZE, IMAGE_SIZE)
        return outputs_k, z_k, transport

    def compute_divergent_loss(self, outputs_k, target, z_k):
        l_div = diversity_loss(outputs_k)
        l_pla = plausibility_loss(outputs_k, target)
        l_nov = self.novelty_loss(z_k)
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
        bb = self.backbone.count_parameters()
        pr = self.probe.count_parameters()
        ga = self.gate.count_parameters()
        return {
            'backbone': bb, 'probe': pr, 'gate': ga,
            'total': bb + pr + ga,
            'probe_ratio': pr / bb * 100,
        }
