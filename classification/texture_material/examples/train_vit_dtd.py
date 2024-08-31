import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import DTD
from torchvision.transforms import transforms
from torchvision.models.vision_transformer import vit_b_32, ViT_B_32_Weights

from torchinfo import summary


def main():
    TRAIN_BATCH_SIZE = 32
    TEST_VAL_BATCH_SIZE = 64
    IMAGE_SIZE = 224
    # dataloaders
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = DTD(root="data", split="train", transform=transform, download=True)
    val_dataset = DTD(root="data", split="val", transform=transform, download=True)
    test_dataset = DTD(root="data", split="test", transform=transform, download=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=TEST_VAL_BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=TEST_VAL_BATCH_SIZE, shuffle=True
    )

    print(
        f"Train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    # model
    model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    model = model.cuda()

    summary(model, input_size=(TRAIN_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

    # train
    for epoch in range(10):
        model.train()
        for images, labels in train_dataloader:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for images, labels in val_dataloader:
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Epoch: {epoch}, Val Accuracy: {correct / total:.4f}")

        # test
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for images, labels in test_dataloader:
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Epoch: {epoch}, Test Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
