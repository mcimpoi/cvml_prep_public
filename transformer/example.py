#
import torch
import torch.nn as nn
import logging

from typing import Optional

from transformer.model import Transformer
from transformer.training import run_epoch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

DEFAULT_VOCAB_SIZE = 11


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


class Batch:
    def __init__(self, src: torch.Tensor, tgt: Optional[torch.Tensor], pad: int):
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


def generate_synthetic_data(vocab_size: int, batch_size: int, n_batches: int):
    for i in range(n_batches):
        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


def example_inference(decode_steps: int = 10) -> None:
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

    fwd_out = model.forward(src, src, src_mask, src_mask)
    print(f"fwd_out shape: {fwd_out.shape}")
    print(f"src shape: {src.shape} src_mask shape: {src_mask.shape}")

    memory = model.encode(src, src_mask)
    logger.debug(f"Memory shape: {memory.shape}")
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(decode_steps):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()  # same as .data[0]
        ys = torch.cat([ys, torch.tensor([[next_word]]).type_as(src)], dim=1)

    print(f"Example untrained model prediction: {ys}")


def example_simple_model_train():
    VOCAB_SIZE = 11
    model: Transformer = Transformer(
        n_blocks=2,
        d_model=64,
        num_heads=8,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dropout_prob=0.1,
    )
    model = init_model_xavier(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        # TODO: compare with lthe learning rate formula from paper / blog
        lr_lambda=lambda step: 0.95**step + 1,
    )

    batch_size = 10

    for epoch in range(n_epochs):
        model.train()
        run_epoch(
            data_iter=generate_synthetic_data(
                vocab_size=VOCAB_SIZE,
                batch_size=batch_size,
                n_batches=20,
            ),
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            mode="train",
            accum_iter=1,
        )


# TODO: greedy decode
# run eval model


def get_model(
    src_vocab_size: int = DEFAULT_VOCAB_SIZE,
    tgt_vocab_size: int = DEFAULT_VOCAB_SIZE,
    n_blocks: int = 2,
    d_model: int = 64,
    n_heads: int = 8,
) -> Transformer:
    model: Transformer = Transformer(
        n_blocks=n_blocks,
        d_model=d_model,
        num_heads=n_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout_prob=0.1,
    )
    model = init_model_xavier(model)
    return model


def test_data_iter(model: Transformer, data_iter=None):
    batch = next(data_iter)

    logger.debug(f"Batch src: {batch.src.shape} Batch tgt: {batch.tgt.shape}")
    logger.debug(
        f"Batch src_mask: {batch.src_mask.shape} Batch tgt_mask: {batch.tgt_mask.shape}"
    )

    # out = model.forward(
    #     batch.src,
    #     batch.tgt,
    #     batch.src_mask,
    #     batch.tgt_mask,
    # )
    # print(f"out shape: {out.shape}")


if __name__ == "__main__":
    # example_inference()

    model = get_model(11, 11, n_blocks=2, d_model=512)

    from model_reference import make_model, data_gen

    model_ref = make_model(11, 11, N=2)

    data_iter = generate_synthetic_data(11, 2, 5)
    data_iter_ref = data_gen(11, 2, 5)

    test_data_iter(model, data_iter)
    test_data_iter(model_ref, data_iter_ref)

    model.train()
    model_ref.train()

    data_iter = generate_synthetic_data(11, 2, 1)
    batch = next(data_iter)

    # res = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
    res = model.encode(batch.src, batch.src_mask)
    logger.debug(f"res shape: {res.shape}")
