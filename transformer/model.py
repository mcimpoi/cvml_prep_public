# Reference:
# http://nlp.seas.harvard.edu/annotated-transformer/#prelims
#

import torch
import torch.nn as nn

from typing import Tuple
import logging

logger = logging.getLogger(__name__)

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

    def __init__(self, num_heads: int, d_model: int, dropout_prob: float = 0.1) -> None:
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
        self.dropout = nn.Dropout(dropout_prob)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, n_queries, d_model)
        Q = self.Wq(query).view(query.shape[0], -1, self.num_heads, self.d_k)
        K = self.Wk(key).view(key.shape[0], -1, self.num_heads, self.d_k)
        V = self.Wv(value).view(value.shape[0], -1, self.num_heads, self.d_k)

        x, _ = self.attention(Q, K, V, mask=mask)
        x = x.view(x.shape[0], -1, self.num_heads * self.d_k)

        return self.Wo(x)

    def attention(self, query, key, value, mask=None, dropout_prob=None):
        # query: (batch_size, vocab_size
        # key: (batch_size, n_keys, d_model)
        # value: (batch_size, n_keys, d_model)
        # ?? mask: (batch_size, n_queries, n_keys)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout_prob is not None:
            p_attn = nn.Dropout(dropout_prob)(p_attn)
        return torch.matmul(p_attn, value), p_attn
