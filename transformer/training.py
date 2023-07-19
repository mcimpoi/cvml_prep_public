# Following http://nlp.seas.harvard.edu/annotated-transformer/ - training loop
import logging
import torch
import torch.nn as nn
import torch.utils.data

from transformer.model import Transformer

from typing import Generator, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TrainState:
    def __init__(self):
        self.step: int = 0
        self.accum_step: int = 0
        self.samples: int = 0
        self.tokens: int = 0


def run_epoch(
    data_iter: Generator,
    model: Transformer,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    mode: str = "train",
    accum_iter: int = 1,
    train_state=TrainState(),  # TODO: do we need it?
) -> Tuple[float, TrainState]:
    """
    Standard Training and Logging Function
    """
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src,
            batch.tgt,
            batch.src_mask,
            batch.tgt_mask,
        )
        loss = criterion(out, batch.tgt_y)
        loss_node = loss / batch.n_tokens

        if mode.startswith("train"):
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.n_tokens

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()  # set_to_none=True by default
                scheduler.step()
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens

    return total_loss, train_state
