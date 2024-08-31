# Implementing Vision Transformer (https://arxiv.org/abs/2010.11929)
# with some inspiration from:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# but using the transformer implementation from AnnotatedTransformer.

import torch.nn as nn
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ViT(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        transformer: Optional[nn.Module] = None,
    ) -> None:
        self._transformer = (transformer if transformer is not None
                             else nn.Transformer())
        self._image_size = (image_size if isinstance(image_size, tuple)
                            else (image_size, image_size))
        self._patch_size = (patch_size if isinstance(patch_size, tuple)
                            else (patch_size, patch_size))
        
        num_patches = (self._image_size[0] // self._patch_size[0]) * \
                      (self._image_size[1] // self._patch_size[1])

        self._positional_embedding = nn.Embedding(
            1,
            embedding_dim=num_patches + 1,)

    def forward(self, x):
        patches = x.unfold(2, self._patch_size, self._patch_size).unfold(
            3, self._patch_size, self._patch_size)
        logger.info(f"patches.shape: {patches.shape}")

