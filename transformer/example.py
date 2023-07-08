#
import torch
import torch.nn as nn
import logging

from transformer.model import Transformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def init_model_xavier(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.int8)
    return mask == 0


if __name__ == "__main__":
    model: Transformer = Transformer(
        n_blocks=2,
        d_model=128,
        num_heads=8,
        src_vocab_size=11,
        tgt_vocab_size=11,
        dropout_prob=0.1,
    )

    model = init_model_xavier(model)

    model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = model.encode(src, src_mask)
    logger.debug(f"Memory shape: {memory.shape}")
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()  # same as .data[0]
        ys = torch.cat([ys, torch.tensor([[next_word]]).type_as(src)], dim=1)

    print(f"Example untrained model prediction: {ys}")
