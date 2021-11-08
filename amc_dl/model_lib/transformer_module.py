import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    print(subsequent_mask)
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot Product Attention.
    query, key, value: (*, L, d_k)
    mask: (*, L(q), L(k)). Masked entries are indicated by 0.
        can be broadcastable.
    dropout: a function
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linears_in = [self.linear_q, self.linear_k, self.linear_v]

        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else dropout

    def forward(self, query, key, value, mask=None, regular=True,
                linear_v=None):
        """
        q, k, v: (bs, L, d_model)
        mask: (bs, L, L) or (1, L, L)
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if regular:
            query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears_in, (query, key, value))]
        else:
            query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip([self.linears_in[0], self.linears_in[1], linear_v],
                                 (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linear_out(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, non_linear=None, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout
        self.non_linear = nn.ReLU() if non_linear is None else non_linear

    def forward(self, x):
        if self.dropout is None:
            return self.w_2(self.non_linear(self.w_1(x)))
        else:
            return self.w_2(self.dropout(self.non_linear(self.w_1(x))))


class LayerNorm(nn.Module):

    """ it is recommended to use pytorch Layer norm"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.dropout is None:
            return self.norm(x + sublayer(x))
        else:
            return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, h, d_model, d_ff, non_linear, dropout=0.1,
                 attn_dropout=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, attn_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, non_linear,
                                                    dropout)
        self.sublayer_attn = (SublayerConnection(d_model, dropout))
        self.sublayer_ff = (SublayerConnection(d_model, dropout))

    def forward(self, x, mask):
        x = self.sublayer_attn(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_ff(x, self.feed_forward)


class TransformerEncoder(nn.Module):

    def __init__(self, N, h, d_model, d_ff, non_linear, dropout=0.1,
                 attn_dropout=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(h, d_model, d_ff, non_linear, dropout, attn_dropout)
             for _ in range(N)])
        self.d_model = d_model
        self.N = N

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == '__main__':
    # example code
    enc = TransformerEncoder(N=4, h=3, d_model=12, d_ff=9, non_linear=None,
                             dropout=0.1, attn_dropout=None)
    x = torch.rand(2, 4, 12)
    x = enc(x, None)
    print(x.size())
