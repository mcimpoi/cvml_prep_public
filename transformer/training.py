# Following http://nlp.seas.harvard.edu/annotated-transformer/ - training loop
import logging
import torch
import torch.nn as nn
import torch.utils.data
import time

from transformer.model import Transformer

from typing import Generator, Tuple, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainState:
    def __init__(self):
        self.step: int = 0
        self.accum_step: int = 0
        self.samples: int = 0
        self.tokens: int = 0


def run_epoch(
    data_iter: Generator,
    model: Transformer,
    criterion: nn.Module,  # loss function
    label_smoothing: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    mode: str = "train",
    accum_iter: int = 1,
    train_state=TrainState(),  # TODO: do we need it?
    log_freq: int = 50,
) -> Tuple[float, TrainState]:
    """
    Standard Training and Logging Function
    """
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    start_time = time.time()

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src,
            batch.tgt,
            batch.src_mask,
            batch.tgt_mask,
        )
        generator_out = model.generator(out)
        logger.debug(
            f"Generator out shape: {generator_out.shape}, tgt_y shape: {batch.tgt_y.shape}"
        )

        generator_out_dim = generator_out.size(-1)
        loss = criterion(
            generator_out.contiguous().view(-1, generator_out_dim),
            label_smoothing(
                generator_out.contiguous().view(-1, generator_out_dim),
                batch.tgt_y.contiguous().view(-1),
            ),
        )

        loss_node = loss / batch.n_tokens

        if mode.startswith("train"):
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.n_tokens

            if i % accum_iter == 0:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()  # set_to_none=True by default
                n_accum += 1
                train_state.accum_step += 1
            if scheduler is not None:
                scheduler.step()

        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens

        if i % log_freq == 1 and mode.startswith("train"):
            elapsed = time.time() - start_time
            if optimizer is None:
                logger.warning("Optimizer should not be None in train mode")
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch Step: {i:06d} | Accummulation step {n_accum:03d} | "
                f"Loss: {loss_node.item():.2f} |"
                f" Tokens per Sec: {tokens / elapsed:.2f} Lr: {lr:.6f}"
            )
            tokens = 0

    return total_loss, train_state
