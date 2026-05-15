import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import IMAGE_SIZE, LATENT_DIM, NOISE_DIM, K_OUTPUTS
from .config import LAMBDA_DIVERSITY, LAMBDA_PLAUSIBILITY, LAMBDA_NOVELTY


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
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


class SetAggregator(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=LATENT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, latent_dim),
        )

    def forward(self, features):
        pooled = features.mean(dim=1)
        return self.mlp(pooled)


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.deconv0 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn0 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, 4, 2, 1)

    def forward(self, class_emb, injections=None):
        h = self.fc(class_emb)
        h = h.view(h.size(0), 512, 4, 4)

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


class OmniglotBackbone(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=LATENT_DIM, noise_dim=NOISE_DIM):
        super().__init__()
        self.encoder = ImageEncoder(feature_dim)
        self.aggregator = SetAggregator(feature_dim, latent_dim)
        self.decoder = ConditionalDecoder(latent_dim)
        self.noise_dim = noise_dim

    def encode_set(self, support_images):
        B, K, C, H, W = support_images.shape
        flat = support_images.view(B * K, C, H, W)
        features = self.encoder(flat)
        features = features.view(B, K, -1)
        class_emb = self.aggregator(features)
        return class_emb

    def decode(self, class_emb, injections=None):
        return self.decoder(class_emb, injections)

    def forward(self, support_images, target_images):
        class_emb = self.encode_set(support_images)
        generated = self.decode(class_emb)
        return generated, class_emb

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DivergentProbe(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS, hidden_dim=128):
        super().__init__()
        self.k = k_outputs
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim * k_outputs),
        )
        self.noise_scale = nn.Parameter(torch.tensor(0.1))
        self.inj_mlp_0 = nn.Linear(latent_dim, 512)
        self.inj_mlp_1 = nn.Linear(latent_dim, 256)
        self.inj_mlp_2 = nn.Linear(latent_dim, 128)
        self.inj_mlp_3 = nn.Linear(latent_dim, 64)

    def forward(self, class_emb):
        perturbations = self.mlp(class_emb)
        perturbations = perturbations.view(class_emb.size(0), self.k, self.latent_dim)
        eps = self.noise_scale * perturbations
        z_perturbed = class_emb.unsqueeze(1) + eps

        B, K, D = z_perturbed.shape
        z_flat = z_perturbed.reshape(B * K, D)
        injections = [
            self.inj_mlp_0(z_flat).view(B * K, 512, 1, 1),
            self.inj_mlp_1(z_flat).view(B * K, 256, 1, 1),
            self.inj_mlp_2(z_flat).view(B * K, 128, 1, 1),
            self.inj_mlp_3(z_flat).view(B * K, 64, 1, 1),
        ]
        return z_perturbed, injections, eps

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DivergenceGate(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, class_emb):
        return self.net(class_emb)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def diversity_loss(outputs_k):
    B, K, C, H, W = outputs_k.shape
    flat = outputs_k.view(B, K, -1)
    flat_norm = F.normalize(flat, dim=-1)
    sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))
    mask = 1 - torch.eye(K, device=outputs_k.device).unsqueeze(0)
    pairwise = (sim * mask).sum(dim=(1, 2)) / (K * (K - 1))
    return (1 - pairwise).mean()


def plausibility_loss(outputs_k, target):
    target_expanded = target.unsqueeze(1).expand_as(outputs_k)
    mse = F.mse_loss(outputs_k, target_expanded, reduction='none').view(
        outputs_k.size(0), outputs_k.size(1), -1
    ).mean(dim=-1)
    return mse.min(dim=1)[0].mean()


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
            self.memory[self.memory_ptr:end_idx] = embeddings[:end_idx - self.memory_ptr]
            if end_idx < self.memory_ptr + batch_size:
                remaining = batch_size - (end_idx - self.memory_ptr)
                self.memory[:remaining] = embeddings[end_idx - self.memory_ptr:]
                self.is_full = True
            self.memory_ptr = (self.memory_ptr + batch_size) % self.memory_size

    def forward(self, embeddings_k):
        if self.memory is None or (not self.is_full and self.memory_ptr < 10):
            return torch.tensor(0.0, device=embeddings_k.device, requires_grad=True)

        memory_pool = self.memory if self.is_full else self.memory[:self.memory_ptr]
        B, K, D = embeddings_k.shape
        flat_query = embeddings_k.view(B * K, D)
        flat_norm = F.normalize(flat_query, dim=-1)
        mem_norm = F.normalize(memory_pool, dim=-1)
        sim = torch.mm(flat_norm, mem_norm.t())
        max_sim, _ = sim.max(dim=1)
        over = (max_sim > self.threshold).float()
        coverage = over.view(B, K).mean(dim=1)
        return coverage.mean()


class OmniglotCBDP(nn.Module):
    def __init__(self, feature_dim=256, latent_dim=LATENT_DIM,
                 noise_dim=NOISE_DIM, k_outputs=K_OUTPUTS):
        super().__init__()
        self.backbone = OmniglotBackbone(feature_dim, latent_dim, noise_dim)
        self.probe = DivergentProbe(latent_dim, k_outputs)
        self.gate = DivergenceGate(latent_dim)
        self.novelty_loss = NoveltyLoss()
        self.noise_dim = noise_dim

    def forward_convergent(self, support_images, target_images):
        return self.backbone(support_images, target_images)

    def forward_divergent(self, class_emb):
        z_perturbed, injections, eps = self.probe(class_emb)
        B, K, _ = z_perturbed.shape
        z_flat = z_perturbed.reshape(B * K, -1)
        outputs_flat = self.backbone.decode(z_flat, injections)
        _, C, H, W = outputs_flat.shape
        outputs_k = outputs_flat.view(B, K, C, H, W)
        return outputs_k, z_perturbed, eps

    def compute_divergent_loss(self, outputs_k, target, z_perturbed):
        l_div = diversity_loss(outputs_k)
        l_pla = plausibility_loss(outputs_k, target)
        l_nov = self.novelty_loss(z_perturbed)

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
            'total': bb + pr + ga, 'probe_ratio': pr / bb * 100,
        }
