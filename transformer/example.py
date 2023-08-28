import torch
import torch.nn as nn
import logging

from transformer.model import LabelSmoothing, Transformer
from transformer.training import run_epoch

from data import generate_synthetic_data, subsequent_mask

import model_reference as ref


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(level=logging.INFO)

DEFAULT_VOCAB_SIZE = 11


def init_model_xavier(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return model


def learning_rate_func(
    step: int,
    model_size: int,
    factor: float,
    warmup: int,
):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# TODO: This is copy paste. To understand what it does.
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


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
    logger.info(f"fwd_out shape: {fwd_out.shape}")
    logger.info(f"src shape: {src.shape} src_mask shape: {src_mask.shape}")

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


def example_simple_model_train(
    n_epochs: int = 10,
    batch_size: int = 80,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
):
    model = get_model(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        n_blocks=2,
        d_model=512,
    )

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.5,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        # TODO: compare with lthe learning rate formula from paper / blog
        lr_lambda=lambda step: learning_rate_func(
            step,
            model_size=model.d_model,
            factor=1.0,
            warmup=400,
        ),
    )

    label_smoothing = LabelSmoothing(vocab_size, padding_idx=0, smoothing=0.0)

    for epoch in range(n_epochs):
        model.train()
        train_loss, _ = run_epoch(
            data_iter=generate_synthetic_data(
                vocab_size=vocab_size,
                batch_size=batch_size,
                n_batches=20,
            ),
            model=model,
            criterion=nn.KLDivLoss(reduction="sum"),
            label_smoothing=label_smoothing,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            mode="train",
            accum_iter=1,
        )

        model.eval()
        eval_loss, _ = run_epoch(
            data_iter=generate_synthetic_data(
                vocab_size=vocab_size,
                batch_size=batch_size,
                n_batches=5,
            ),
            model=model,
            criterion=nn.KLDivLoss(reduction="sum"),
            label_smoothing=label_smoothing,
            optimizer=None,
            scheduler=None,
            mode="eval",
        )
        logger.info(
            (
                f"Epoch: {epoch} Train loss: {train_loss:.3f}"
                f" Eval loss: {eval_loss:.3f}"
            )
        )

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    decoded_output = greedy_decode(
        model,
        src,
        src_mask,
        max_len=max_len,
        start_symbol=0,
    )
    print(f"Greedy-decoded output: {decoded_output}")


def get_model(
    src_vocab_size: int = DEFAULT_VOCAB_SIZE,
    tgt_vocab_size: int = DEFAULT_VOCAB_SIZE,
    n_blocks: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    n_heads: int = 8,
) -> Transformer:
    model: Transformer = Transformer(
        n_blocks=n_blocks,
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout_prob=0.1,
    )
    model = init_model_xavier(model)
    return model


def test_data_iter(model: Transformer, data_iter=None):
    batch = next(data_iter)

    logger.debug(
        (
            f" Batch src: {batch.src.shape} Mask: {batch.src_mask.shape}\n"
            f" Batch tgt: {batch.tgt.shape} Mask: {batch.tgt_mask.shape}"
        )
    )

    out = model.forward(
        batch.src,
        batch.tgt,
        batch.src_mask,
        batch.tgt_mask,
    )
    logger.debug(f"out shape: {out.shape}")


if __name__ == "__main__":
    # example_inference()

    model = get_model(11, 11, n_blocks=2)
    model_ref = ref.make_model(11, 11, N=2, d_model=512)

    model.eval()
    model_ref.eval()

    data_iter = generate_synthetic_data(11, 2, 5)
    data_iter_ref = ref.data_gen(11, 2, 5)

    test_data_iter(model, data_iter)
    test_data_iter(model_ref, data_iter_ref)

    print("Reference model")
    ref.example_simple_model(n_epochs=10)

    print("My model")
    example_simple_model_train(n_epochs=10, batch_size=80, vocab_size=11)
