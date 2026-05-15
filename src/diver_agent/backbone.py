import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=32, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.relu(self.fc(z))
        x = x.view(x.size(0), 128, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x


class AutoencoderBackbone(nn.Module):
    def __init__(self, latent_dim=32, input_channels=1):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, input_channels)
        self.decoder = ConvDecoder(latent_dim, input_channels)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
