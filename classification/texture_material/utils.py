import matplotlib.pyplot as plt
from typing import List


def show_batch(image_batch, label_batch, class_labels: List[str]) -> None:
    plt.figure(figsize=(10, 10))
    for n in range(25):
        _ = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n].numpy().astype("uint8"))
        plt.title(class_labels[label_batch[n]])
        plt.axis("off")
