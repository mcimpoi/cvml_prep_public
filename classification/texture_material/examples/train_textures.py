# for collab:
#  - install deps (huggingface)
#    !pip install -q transformers datasets
#  - fix tqdm new line:
#    from functools import partial
#    from tqdm import tqdm
#    tqdm = partial(tqdm, position=0, leave=True)

import os
from tqdm import tqdm
from typing import Dict, Optional
from datasets import load_dataset, DatasetDict
from transformers import (
    ConvNextFeatureExtractor,
    AutoModelForImageClassification,
)
from torch.optim.lr_scheduler import MultiStepLR

import torch
import copy

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RESNET_50_BATCH_SIZE = 32
RESNET_50_MODEL_NAME = "microsoft/resnet-50"


CONVNEXT_ATTO_1K_MODEL_NAME = "facebook/convnextv2-atto-1k-224"
CONVNEXT_ATTO_1K_BATCH_SIZE = 256  # 512 for frozen model.

CONVNEXT_V2_NANO_22K_MODEL_NAME = "facebook/convnextv2-nano-22k-224"
CONVNEXT_V2_NANO_22K_BATCH_SIZE = 16  # 24 fits on 4GB GPU

CONVNEXT_V2_NANO_1K_MODEL_NAME = "facebook/convnextv2-nano-1k-224"
CONVNEXT_V2_NANO_1K_BATCH_SIZE = 16  # 24 fits on 4GB GPU


CONVNEXT_V2_TINY_1K_MODEL_NAME = "facebook/convnextv2-tiny-1k-224"
CONVNEXT_V2_TINY_1K_BATCH_SIZE = 16  # 24 fits on 4GB GPU


MODEL_NAME = CONVNEXT_ATTO_1K_MODEL_NAME
MODEL_BATCH_SIZE = CONVNEXT_ATTO_1K_BATCH_SIZE


feature_extractor = ConvNextFeatureExtractor.from_pretrained(MODEL_NAME)

normalize = Normalize(
    mean=feature_extractor.image_mean, std=feature_extractor.image_std
)


train_transform = Compose(
    [
        RandomResizedCrop(
            (
                feature_extractor.size["shortest_edge"],
                feature_extractor.size["shortest_edge"],
            )
        ),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(30),
        ToTensor(),
        normalize,
    ]
)

test_transform = Compose(
    [
        Resize(
            size=(
                feature_extractor.size["shortest_edge"],
                feature_extractor.size["shortest_edge"],
            )
        ),
        ToTensor(),
        normalize,
    ]
)


def save_model(
    model: torch.nn.Module, checkpoint_path: str, epoch: int, loss, optimizer
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )


def train_transforms(examples: Dict):
    examples["pixel_values"] = [
        train_transform(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


def test_transforms(examples):
    examples["pixel_values"] = [
        test_transform(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


def freeze_conv_next(model: torch.nn.Module) -> torch.nn.Module:
    for param in model.convnextv2.parameters():
        param.requires_grad = False
    return model


def collate_fn(examples):
    pixel_values = torch.stack(
        [example["pixel_values"] for example in examples]
    )

    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def eval_model(model, dataset, split="validation", device=None):
    if device is None:
        device = model.device
    model.eval()

    test_total = 0
    test_correct = 0

    processed_val_dataset = dataset[split].with_transform(test_transforms)

    val_dataloader = torch.utils.data.DataLoader(
        processed_val_dataset,
        batch_size=MODEL_BATCH_SIZE // 2,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(pixel_values=batch["pixel_values"])
        predicted = outputs.logits.argmax(-1)

        test_total += batch["labels"].shape[0]
        test_correct += (predicted == batch["labels"]).sum().item()

    test_acc = test_correct / test_total

    logging.info(f"{split} acc: {test_acc:.2f}")


def plot_losses():
    pass


def training_loop(
    dataset: DatasetDict,
    num_epochs: int = 100,
    eval_every: int = 10,
    model_checkpoint_path: Optional[str] = None,
    validation_split_name: str = "dev",
    checkpoint_dir: Optional[str] = None,
    learning_rate: float = 1e-5,
):
    processed_train_dataset = dataset["train"].with_transform(train_transforms)
    labels = dataset["train"].features["label"].names
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    dataloader = torch.utils.data.DataLoader(
        processed_train_dataset,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"running on device: {device}")
    model = freeze_conv_next(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    total, correct = 0, 0
    losses, accuracies = {}, {}
    model_copy = None

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total = 0
        correct = 0
        total_loss = 0
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"], labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()
            total += batch["labels"].shape[0]
            predicted = outputs.logits.argmax(-1)

            correct += (predicted == batch["labels"]).sum().item()

        accuracy = correct / total
        accuracies[epoch] = accuracy
        losses[epoch] = total_loss

        print(f"Loss: {losses[epoch]:.3f} Accuracy {accuracies[epoch]:.2f}")
        if validation_split_name in dataset and (epoch + 1) % eval_every == 0:
            model_copy = copy.deepcopy(model)
            eval_model(model_copy, dataset, validation_split_name, device)
        if checkpoint_dir is not None:
            save_model(
                model,
                checkpoint_path=f"{checkpoint_dir}/model_{epoch:06d}.pt",
                epoch=epoch,
                loss=losses[epoch],
                optimizer=optimizer,
            )

    eval_model(copy.deepcopy(model), dataset, "test", device)


if __name__ == "__main__":
    dataset_mapping = {
        "dtd": "mcimpoi/dtd_split_1",
        "fmd": "mcimpoi/fmd_materials",
        "minc2k5": "mcimpoi/minc-2500_split_1",
    }

    dataset_key = "minc2k5"
    # ln -s to local folder
    checkpoint_dir = f"/checkpoints/{dataset_key}/{MODEL_NAME}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = load_dataset(dataset_mapping[dataset_key])

    training_loop(
        dataset,
        num_epochs=10,
        validation_split_name="validation",
        checkpoint_dir=checkpoint_dir,
        learning_rate=1e-2,
    )
