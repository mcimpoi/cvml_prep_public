import torch
import torchinfo
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)

SUMMARY_BATCH_SIZE = 16

def print_model_summary(model_name, model, device, img_dim, n_channels, n_context_features=5):
    x_dummy = torch.zeros(SUMMARY_BATCH_SIZE, n_channels, img_dim, img_dim, device=device)
    t_dummy = torch.zeros(SUMMARY_BATCH_SIZE, device=device).long()
    
    if model_name == "unet2d":
        ctx_dummy = torch.zeros(SUMMARY_BATCH_SIZE, 1, n_context_features, device=device)
        input_data = (x_dummy, t_dummy, ctx_dummy)
    elif model_name == "context_unet":
        input_data = (x_dummy, t_dummy)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    try:
        torchinfo.summary(model, input_data=input_data)
    except Exception as e:
        log.warning(f"Could not print model summary: {e}")

def plot_loss(losses, save_dir, model_name):
    plot_path = f"{save_dir}/{model_name}/loss.png"
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved loss plot: {plot_path}")