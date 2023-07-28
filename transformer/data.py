import torch
from typing import Optional, Generator


class Batch:
    def __init__(
        self, src: torch.Tensor, tgt: Optional[torch.Tensor], pad: int
    ) -> None:
        self.src = src
        # unsqueeze adds one dim; [4, 10] --> [4, 1, 10]
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.int8)
    return mask == 0


def generate_synthetic_data(
    vocab_size: int,
    batch_size: int,
    n_batches: int,
) -> Generator[Batch, None, None]:
    for i in range(n_batches):
        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
