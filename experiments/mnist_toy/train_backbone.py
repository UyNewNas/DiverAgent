import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from diver_agent.backbone import AutoencoderBackbone

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
LATENT_DIM = 32
EPOCHS = 30
LR = 1e-3
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_loader, test_loader = get_mnist_loaders()

    model = AutoencoderBackbone(latent_dim=LATENT_DIM).to(DEVICE)
    print(f'Backbone parameters: {model.count_parameters():,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for x, _ in pbar:
            x = x.to(DEVICE)
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(DEVICE)
                x_recon, _ = model(x)
                test_loss += criterion(x_recon, x).item() * x.size(0)
        test_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}')

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'backbone_best.pt'))

    print(f'Training complete. Best test loss: {best_loss:.6f}')
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'backbone_final.pt'))


if __name__ == '__main__':
    main()
