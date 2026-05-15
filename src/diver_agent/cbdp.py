import torch
import torch.nn as nn

from .backbone import AutoencoderBackbone
from .probe import DivergentProbe
from .gate import DivergenceGate
from .losses import divergent_loss, NoveltyLoss


class CBDP(nn.Module):
    def __init__(self, latent_dim=32, k_outputs=4, probe_hidden=48):
        super().__init__()
        self.backbone = AutoencoderBackbone(latent_dim=latent_dim)
        self.probe = DivergentProbe(latent_dim=latent_dim, hidden_dim=probe_hidden, k_outputs=k_outputs)
        self.gate = DivergenceGate(latent_dim=latent_dim)
        self.novelty_loss = NoveltyLoss()
        self.latent_dim = latent_dim
        self.k_outputs = k_outputs

    def forward_convergent(self, x):
        return self.backbone(x)

    def forward_divergent(self, x):
        z = self.backbone.encode(x)
        z_perturbed, epsilons = self.probe(z)
        batch, K, _ = z_perturbed.shape
        z_flat = z_perturbed.view(batch * K, self.latent_dim)
        outputs_flat = self.backbone.decode(z_flat)
        _, C, H, W = outputs_flat.shape
        outputs_k = outputs_flat.view(batch, K, C, H, W)
        return outputs_k, z_perturbed, epsilons

    def forward(self, x, use_divergence=False):
        x_recon, z = self.backbone(x)
        if not use_divergence:
            return x_recon, z, None
        outputs_k, z_perturbed, epsilons = self.forward_divergent(x)
        return x_recon, z, outputs_k

    def compute_divergent_loss(self, outputs_k, target, z_perturbed):
        return divergent_loss(
            outputs_k, target, z_perturbed, self.novelty_loss,
            lambda_diversity=0.15, lambda_plausibility=0.8, lambda_novelty=0.5,
        )

    def update_memory(self, embeddings):
        self.novelty_loss.update_memory(embeddings)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_probe(self):
        for param in self.probe.parameters():
            param.requires_grad = False

    def unfreeze_probe(self):
        for param in self.probe.parameters():
            param.requires_grad = True

    def parameter_stats(self):
        backbone_params = self.backbone.count_parameters()
        probe_params = self.probe.count_parameters()
        gate_params = self.gate.count_parameters()
        total = backbone_params + probe_params + gate_params
        return {
            'backbone': backbone_params,
            'probe': probe_params,
            'gate': gate_params,
            'total': total,
            'probe_ratio': probe_params / backbone_params * 100,
        }
