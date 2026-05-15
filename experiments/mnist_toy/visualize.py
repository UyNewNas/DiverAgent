import torch
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from diver_agent.cbdp import CBDP

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 32
K_OUTPUTS = 4
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = CBDP(latent_dim=LATENT_DIM, k_outputs=K_OUTPUTS).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, 'cbdp_full.pt')
    if not os.path.exists(ckpt_path):
        print('No trained model found.')
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    for digit in range(10):
        indices = [i for i, (_, y) in enumerate(test_dataset) if y == digit]
        indices = random.sample(indices, min(8, len(indices)))
        subset = Subset(test_dataset, indices)
        loader = DataLoader(subset, batch_size=8, shuffle=False)

        x_batch, _ = next(iter(loader))
        x_batch = x_batch.to(DEVICE)

        with torch.no_grad():
            x_recon, _ = model.forward_convergent(x_batch)
            outputs_k, _, _ = model.forward_divergent(x_batch)

        grid = []
        for i in range(min(8, x_batch.size(0))):
            grid.append(x_batch[i])
            grid.append(x_recon[i])
            for k in range(K_OUTPUTS):
                grid.append(outputs_k[i, k])

        grid_tensor = torch.stack(grid)
        save_image(
            grid_tensor,
            os.path.join(RESULTS_DIR, f'digit_{digit}_comparison.png'),
            nrow=2 + K_OUTPUTS,
            pad_value=1.0,
        )
        print(f'Saved comparison for digit {digit}')

    print('All visualizations saved.')


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    main()
