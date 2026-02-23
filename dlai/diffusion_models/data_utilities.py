import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SpritesDataset(Dataset):
    def __init__(
        self, sprites_fname: str, labels_fname: str, transform=None, null_context=False
    ):
        self.sprites = np.load(sprites_fname)
        self.labels = np.load(labels_fname)
        self.transform = transform
        self.null_context = null_context

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        if self.transform:
            image = self.transform(self.sprites[idx])
        else:
            image = self.sprites[idx]

        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = torch.tensor(self.labels[idx]).to(torch.int64)

        return image, label

    def get_shapes(self):
        return self.sprites.shape, self.labels.shape


transform = transforms.Compose(
    [
        transforms.ToTensor(),  # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,)),  # range [-1,1]
    ]
)
