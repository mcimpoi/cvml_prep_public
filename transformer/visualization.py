# From https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html
import torch
from typing import Optional, List
import matplotlib.pyplot as plt


def show_heatmaps(
    matrices: torch.Tensor,
    xlabel: str,
    ylabel: str,
    titles: Optional[List[str]] = None,
    figsize=(2.5, 2.5),
    cmap="Reds",
) -> None:
    n_rows, n_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    heatmap = None
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            heatmap = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == n_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    if heatmap:
        fig.colorbar(heatmap, ax=axes, shrink=0.6)
