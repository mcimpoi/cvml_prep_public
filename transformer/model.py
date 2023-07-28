# Following "The Annotated Transformer"
# http://nlp.seas.harvard.edu/annotated-transformer/

import math
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_D_FF: int = 2048
DEFAULT_ATTN_HEADS: int = 8


# EncoderDecoder in Annotated Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        n_blocks: int,
        d_model: int,
        d_ff: int = DEFAULT_D_FF,
        num_heads: int = DEFAULT_ATTN_HEADS,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(n_blocks, d_model, num_heads, d_ff)
        self.decoder = Decoder(n_blocks, d_model, num_heads, d_ff)
        self.src_embed = nn.Sequential(
            nn.Embedding(src_vocab_size, d_model),
            PositionalEncoding(d_model, dropout_prob),
        )
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.generator = Generator(d_model, tgt_vocab_size)  # TODO here

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout_prob)
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        logger.debug(
            f"Enc::forward::\n  x: {x.shape}, src_mask:"
            f" {src_mask.shape if src_mask is not None else None}"
        )
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout_prob)
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        out = self.norm(x)
        logger.debug(f"Decoder::forward:: out: {out.shape}")
        return out


class EncoderLayer(nn.Module):
    """Encoder Layer - Figure 1 left"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = DEFAULT_D_FF,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout_prob)
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout_prob,
        )
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout_prob) for _ in range(2)]
        )

    def forward(self, x, mask=None):
        logger.debug(
            f"EncoderLayer::forward\n x: {x.shape}"
            + f"mask: {mask.shape if mask is not None else None}"
        )
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = DEFAULT_D_FF,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, dropout_prob)
        self.src_attn = MultiHeadedAttention(num_heads, d_model, dropout_prob)
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout_prob,
        )
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout_prob) for _ in range(3)]
        )

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](
            x,
            lambda x: self.src_attn(
                x,
                memory,
                memory,
                src_mask,
            ),
        )
        return self.sublayer[2](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, sublayer):
        # Sec. 3.1. - output of each sublayer is LayerNorm(x + Sublayer(x))
        # Annotated transformer uses:
        #  return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """TODO"""

    def __init__(self, d_model: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()
        # in the paper, d_model: 512, d_ff: 2048
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # max(0, xW_1 + b_1)W_2 + b_2
        # i.e. relu(xW_1 + b_1) * W_2 + b_2
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout_prob: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        # div_term = 10000^(2i / d_model)
        # take log --> 2i * log(10000) / d_model--> 2i comes from arange
        # when we exponentiate, we get 10000^(2i / d_model).
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # (d_model / 2)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # https://discuss.pytorch.org/t/what-does-register-buffer-do/121091
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.proj(x), dim=-1)


# From Sec 3.2.2
# In this work we employ h = 8 parallel attention layers, or heads.
# For each of these we use dk = dv = dmodel/h = 64.
# Due to the reduced dimension of each head, the total computational cost
# is similar to that of single-head attention with full dimensionality.
class DotProductAttention(nn.Module):
    attention_weights = None

    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        dim = query.shape[-1]
        scores = torch.bmm(query, key.transpose(-2, -1)) / dim**0.5
        self.attention_weights = scores.softmax(dim=-1)
        return (
            torch.bmm(self.dropout(self.attention_weights), value),
            self.attention_weights,
        )


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout_prob: float = 0.1,
    ) -> None:
        """Initialize Multi-Head Attention

        Args:
            num_heads (int): number of heads
            d_model (int): dimension of model
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # assumes dq = dk = dv = d_model / num_heads
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, n_queries, d_model)

        # this seems important!
        # same mask is applied to all heads

        if mask is not None:
            mask = mask.unsqueeze(1)

        logger.debug(f"MultiHeadedAttention:: query: {query.shape}")
        Q = (
            self.Wq(query)
            .view(query.shape[0], -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.Wk(key)
            .view(key.shape[0], -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.Wv(value)
            .view(value.shape[0], -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        logger.debug(
            f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, "
            + f"mask: {mask.shape if mask is not None else None}"
        )
        x, _ = self.attention(Q, K, V, mask, self.dropout_prob)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )

        out = self.Wo(x)
        logger.debug(f"out: {out.shape}")
        return out

    def attention(self, query, key, value, mask=None, dropout_prob=None):
        # query: (batch_size, vocab_size
        # key: (batch_size, n_keys, d_model)
        # value: (batch_size, n_keys, d_model)
        # ?? mask: (batch_size, n_queries, n_keys)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k**0.5
        logger.debug(
            f"attn::\n query: {query.shape}, key: {key.shape}, value: {value.shape}"
        )
        logger.debug(f"scores: {scores.shape}")
        logger.debug(f"mask: {mask.shape if mask is not None else None}")
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout_prob is not None:
            p_attn = nn.Dropout(dropout_prob)(p_attn)
        return torch.matmul(p_attn, value), p_attn


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # logger.debug(f"x: {x.size()}, target: {target.size()}")
        assert x.size(1) == self.size
        true_dist = torch.ones_like(x) * self.smoothing / (self.size - 2)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.true_dist
