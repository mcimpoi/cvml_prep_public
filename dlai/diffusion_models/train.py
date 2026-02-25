import os
import click

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from diffusers import UNet2DModel

from model import ContextUnet
from data_utilities import SpritesDataset, transform


def make_noise_schedule(
    timesteps: int, beta1: float, beta2: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1.0
    return a_t, b_t, ab_t


def perturb_input(x, timestep, noise, ab_t):
    return (
        ab_t.sqrt()[timestep, None, None, None] * x
        + (1 - ab_t[timestep, None, None, None]) * noise
    )


def get_dataloader(sprites_path, labels_path, batch_size, num_workers):
    dataset = SpritesDataset(sprites_path, labels_path, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataloader


def train_model(
    sprites_path: str,
    labels_path: str,
    save_dir: str = ".",
    n_epochs: int = 32,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    timesteps: int = 500,
    beta1: float = 1e-4,
    beta2: float = 0.02,
    n_features: int = 64,
    n_channels: int = 3,
    n_context_features: int = 5,
    img_dim: int = 16,
    save_every: int = 4,
    num_workers: int = 0,
    model_type: str = "context_unet",
    compile_model: bool = True,
    use_context: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataloader = get_dataloader(sprites_path, labels_path, batch_size, num_workers)

    _, _, ab_t = make_noise_schedule(timesteps, beta1, beta2, device)

    # TODO: add in_channels as a parameter to train_model()
    if model_type == "context_unet":
        model = ContextUnet(
            in_channels=n_channels,
            n_features=n_features,
            n_context_features=n_context_features,
            img_dim=img_dim,
        ).to(device)
    elif model_type == "unet2d":
        model = UNet2DModel(
            sample_size=img_dim,  # the target image resolution
            in_channels=n_channels,  # the number of input channels, 3 for RGB images
            out_channels=n_channels,  # the number of output channels
            layers_per_block=2,  # how many layers to use per UNet block
            block_out_channels=(
                n_features,
                n_features * 2,
                n_features * 4,
            ),  # More channels -> more capacity
            downsample_padding=1,
            down_block_types=(
                "DownBlock2D",  # a normal downsampling block
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",  # a normal upsampling block
            ),
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.train()
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # for faster training in PyTorch 2.0+

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TODO: add os.makedirs(save_dir, exist_ok=True) here so save_dir is
    #       created automatically if it doesn't exist (e.g. in Colab)
    os.makedirs(save_dir, exist_ok=True)

    losses = []
    model_name = f"{model_type}_{'c' if compile_model else 'nc'}_{'ctx' if use_context else 'nctx'}"
    for ep in range(n_epochs):
        optim.param_groups[0]["lr"] = learning_rate * (1 - ep / n_epochs)

        pbar = tqdm(dataloader, desc=f"Epoch {ep + 1}/{n_epochs}")
        epoch_losses = []
        # TODO: use context labels for conditional training instead of discarding them
        for x, labels in pbar:
            optim.zero_grad()
            x = x.to(device)
            if use_context:
                context = labels.to(device).float()
                context_mask = torch.bernoulli(
                    torch.full((context.shape[0],), 0.9, device=device)
                )
                context = context * context_mask.unsqueeze(-1)
            else:
                context = None

            t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            x_perturbed = perturb_input(x, t, noise, ab_t)

            if model_type == "unet2d":
                predicted_noise = model(x_perturbed, t).sample
            else:
                predicted_noise = model(x_perturbed, t / timesteps, context=context)

            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        epoch_mean = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_mean)

        if (ep + 1) % save_every == 0 or (ep + 1) == n_epochs:
            path = f"{save_dir}/{model_name}/epoch_{(ep + 1):03d}.pth"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)
            print(f"Saved checkpoint: {path}")

    plot_path = f"{save_dir}/{model_name}/loss.png"
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot: {plot_path}")

    return model


def test_load_and_perturb(
    sprites_path, labels_path, timesteps=500, n_steps=10, beta1=1e-4, beta2=0.02
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, ab_t = make_noise_schedule(timesteps, beta1, beta2, device)

    n_images = 3
    dataset = SpritesDataset(sprites_path, labels_path, transform=transform)
    step_indices = torch.linspace(1, timesteps, n_steps, dtype=torch.long)

    _, axs = plt.subplots(
        n_images, n_steps + 1, figsize=(2 * (n_steps + 1), 2 * n_images)
    )

    for row, img_idx in enumerate(range(n_images)):
        x, _ = dataset[img_idx]
        x = x.to(device)
        noise = torch.randn_like(x)

        axs[row, 0].imshow((x.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
        axs[row, 0].set_title("t=0")
        axs[row, 0].axis("off")

        for i, t in enumerate(step_indices):
            x_t = perturb_input(x, t, noise, ab_t)
            axs[row, i + 1].imshow((x_t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
            axs[row, i + 1].set_title(f"t={t.item()}")
            axs[row, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("debug_noising.png", bbox_inches="tight")
    plt.close()
    print("Saved debug_noising.png")


@click.command()
@click.argument("mode", type=click.Choice(["train", "debug"]), default="train")
@click.option("--sprites", default="/data/sprites/sprites_1788_16x16.npy")
@click.option("--labels", default="/data/sprites/sprite_labels_nc_1788_16x16.npy")
@click.option(
    "--model-type",
    default="context_unet",
    type=click.Choice(["context_unet", "unet2d"]),
)
@click.option("--save-dir", default="./checkpoints")
def main(mode, sprites, labels, model_type, save_dir):
    if mode == "debug":
        test_load_and_perturb(sprites, labels)
    else:
        train_model(
            sprites_path=sprites,
            labels_path=labels,
            model_type=model_type,
            save_dir=save_dir,
            batch_size=256,
            n_epochs=3,
            save_every=1,
            use_context=True,
        )


if __name__ == "__main__":
    main()
