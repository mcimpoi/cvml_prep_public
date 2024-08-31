import time
import tempfile
import torch
import os


def train_model(
    model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes
):
    since = time.time()
    with tempfile.TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)

        best_acc = 0.0

        for epoch in range(num_epochs):
            for phase in ("train", "val"):
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_correct = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_correct += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_correct.double() / dataset_sizes[phase]

                print(
                    f"Epoch: {epoch + 1} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
                )

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        model.load_state_dict(torch.load(best_model_params_path))
    return model
