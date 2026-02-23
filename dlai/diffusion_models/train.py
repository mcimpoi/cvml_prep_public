import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import ContextUnet
from data_utilities import SpritesDataset, transform


def make_noise_schedule(timesteps, beta1, beta2, device):
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    return ab_t


def perturb_input(x, timestep, noise, ab_t):
    return (
        ab_t.sqrt()[timestep, None, None, None] * x
        + (1 - ab_t.sqrt()[timestep, None, None, None]) * noise
    )


def train_model(
    sprites_path: str,
    labels_path: str,
    save_dir: str = ".",
    n_epochs: int = 32,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    timesteps: int = 500,
    beta1: float = 1e-4,
    beta2: float = 0.02,
    n_features: int = 64,
    n_context_features: int = 8,
    img_dim: int = 16,
    save_every: int = 4,
    num_workers: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    ab_t = make_noise_schedule(timesteps, beta1, beta2, device)

    # TODO: add in_channels as a parameter to train_model()
    model = ContextUnet(
        in_channels=3,
        n_features=n_features,
        n_context_features=n_context_features,
        img_dim=img_dim,
    ).to(device)

    model.train()
    # TODO: model.compile() is wrong — nn.Module has no .compile() method.
    #       use torch.compile(model) instead.
    model = torch.compile(model)  # for faster training in PyTorch 2.0+

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TODO: add os.makedirs(save_dir, exist_ok=True) here so save_dir is
    #       created automatically if it doesn't exist (e.g. in Colab)

    dataset = SpritesDataset(sprites_path, labels_path, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    for ep in range(n_epochs):
        optim.param_groups[0]["lr"] = learning_rate * (1 - ep / n_epochs)

        pbar = tqdm(dataloader, desc=f"Epoch {ep + 1}/{n_epochs}")
        # TODO: use context labels for conditional training instead of discarding them
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            x_perturbed = perturb_input(x, t, noise, ab_t)

            predicted_noise = model(x_perturbed, t / timesteps)

            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        if (ep + 1) % save_every == 0 or (ep + 1) == n_epochs:
            path = f"{save_dir}/ddpm_epoch_{ep + 1}.pth"
            torch.save(model.state_dict(), path)
            print(f"Saved checkpoint: {path}")

    return model


if __name__ == "__main__":
    train_model(
        sprites_path="/data/sprites/sprites_1788_16x16.npy",
        labels_path="/data/sprites/sprite_labels_nc_1788_16x16.npy",
    )
