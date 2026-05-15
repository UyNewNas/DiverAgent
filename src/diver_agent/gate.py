import torch
import torch.nn as nn


class DivergenceGate(nn.Module):
    def __init__(self, latent_dim=32, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.scorer = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        score = self.scorer(z).squeeze(-1)
        activated = score > self.threshold
        return activated, score

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
