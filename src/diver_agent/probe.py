import torch
import torch.nn as nn


class DivergentProbe(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=48, k_outputs=4, noise_std=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.k_outputs = k_outputs

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim * k_outputs)
        self.relu = nn.ReLU(inplace=True)
        self.noise_std = noise_std

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        batch_size = z.size(0)
        h = self.relu(self.fc1(z))
        epsilons = self.fc2(h).view(batch_size, self.k_outputs, self.latent_dim)
        epsilons = epsilons + torch.randn_like(epsilons) * self.noise_std
        z_perturbed = z.unsqueeze(1) + epsilons
        return z_perturbed, epsilons

    def forward_single(self, z):
        z_perturbed, _ = self.forward(z.unsqueeze(0))
        return z_perturbed.squeeze(0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
