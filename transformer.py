import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Implementation of Generalized Relative Positional Encoding (RPE)

The implementation uses array indexing techniques as a fast implementation
of RPE. Here, 
- eq. (10) & eq. (11) in MuseBERT paper is computed using distributive law.
- Direct application of eq. (10) and eq. (11) will be extremely slow.
  Since there are only a few (4) relations and their embedding is used in a 
  huge (nearly 100 * 100) matrix, we compute the embeddings of these relation,
  and avoid expanding the embedding into a huge matrix. This makes the model
  to have a reasonable running time (approx. twice the time w/o RPE).
"""


def query_multiply_rel_pos(query, emb, relmat):
    """
    Compute a part of *query-value product* in generalized RPE.
    (See eq. (10) in the MuseBERT paper.)
    Specifically,
      - $ h_i * W^Q * sum_a (Emb_a^K(r_{ij}^a)) $
    Here
      - b for batch size, h for n_head, vs for vocabulary size.
      - dtype is torch.float unless specified.

    :param query: (b, h, Lq, d)
    :param emb: (vs, d) or (h, vs, d) if different for each head.
        It should include null embedding (usually all zeros at emb[0])
    :param relmat: (b, Lq, Lk)
        dtype: LongTensor
    :return: the score of shape (b, h, Lq, Lk)
    """

    b, h, Lq, d = query.size()
    vs = emb.size(-2)

    rel_pos_scores = \
        torch.matmul(query, emb.transpose(-2, -1)) / math.sqrt(d)

    # reshape to 1-d array: (b * h * Lq * vocab_size,)
    rel_pos_scores = rel_pos_scores.reshape(-1)

    # create posmat: index selection of rel_pos_scores
    # posmat: should be (b, h, Lq, Lk), starting from (b * h * Lq, 1)
    posmat = torch.arange(0, b * h * Lq * vs, vs).to(device).unsqueeze(-1)

    # reshape and transpose posmat -> (h, b, Lq, 1)
    posmat = posmat.reshape(b, h, Lq, 1).transpose_(0, 1)

    # the above operation makes the below addition broadcastable
    # posmat: (h, b, Lq, 1), relmat: (b, Lq, Lk)
    # hence, pos_mat: (h, b, Lq, Lk)
    posmat = posmat + relmat

    # transpose posmat to return dim
    posmat = posmat.transpose_(0, 1)  # (b, h, Lq, Lk)

    # return the relative pos score by indexing
    return rel_pos_scores[posmat]


def attention(query, key, value, rel_scores=None,
              mask=None, dropout=None):
    """
    Compute a part of *attention weight application* and *query-value product*
    in generalized RPE.
    (See eq. (10) - (11) in the MuseBERT paper.)
    Specifically,
    - We use distributive law on eq. (11). The function computes the
      first term:
      $ sum_j (alpha_{ij} * h_j * W^V) $
    - alpha_{ij} is the softmax of output e_{ij} of eq. (10).
    - We use distributive law on eq. (10). The function computes
      $ e_{ij} = h_iW^Q / (h_jW^K)^T  / sqrt{d_z}  + rel_scores $,
      where rel_scores is computed in `query_multiply_rel_pos`.
    Here,
      - b for batch size, h for n_head, vs for vocabulary size.
      - dtype is torch.float unless specified.
    :param query: (b, h, Lq, d)
    :param key: (b, h, Lk, d)
    :param value: (b, h, Lk, d)
    :param rel_scores: (b, h, Lq, Lk) (value-relation embeddings)
    :param mask: (*, Lq, Lk)
        Masked positions are 0, Unmasked are 1. Bool Tensor.
    :param dropout: a dropout function or None.
    :return: (b, h, Lk, d_v)
    """

    d_k = query.size(-1)

    # scores: (b, h, Lq, Lk)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if rel_scores is not None:
        scores += rel_scores

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # p_attn: (b, h, Lq, Lk)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def compute_bool_rel_mat(rel_mat, vs, ignore_zero=True):
    """
    Compute boolean matrix of rel_mat. The function is called in
    `compute_rel_attn_value`.

    :param rel_mat: (bs, Lq, Lk)
    :param vs: vocabulary size, int
    :param ignore_zero: boolean. Whether to treat 0 as a padding relation.
    :return: (b, Lq, vs, Lk) or (b, Lq, vs - 1, Lk) if ignore_zero is True.
        float tensor
    """
    start_ind = 1 if ignore_zero else 0
    return torch.stack([rel_mat == i
                        for i in range(start_ind, vs)], -2).float()


def compute_rel_attn_value(p_attn, rel_mat, emb, ignore_zero=True):
    """
    Compute a part of *attention weight application* and *query-value product*
    in generalized RPE.
    (See eq. (10) - (11) in the MuseBERT paper.)
    Specifically,
    - We use distributive law on eq. (11). The function computes the
      second term:
      $ sum_j (alpha_{ij} * sum_a Emb_a^K(r_{ij}^a)) $
    Here,
      - b for batch size, h for n_head, vs for vocabulary size.
      - dtype is torch.float unless specified.

    :param p_attn: (b, d, L_q, L_k)
    :param rel_mat: (b, Lq, Lk)
    :param emb: (h, vs, d)
    :param ignore_zero: bool. Whether to exclude the first vocab.
    :return: (b, h, Lq, d)
    """

    vs = emb.size(-2)

    # bool_relmat: (b, Lq, vs - 1, Lk), dtype: torch.float
    bool_relmat = compute_bool_rel_mat(rel_mat, vs, ignore_zero=ignore_zero)

    # p_attn: -> (b, d, Lq, 1, 1, Lk)
    # bool_relmat: -> (b, 1, L_q, vs - 1, L_k, 1)
    # acmlt_p_attn: (b, d, Lq, vs - 1, 1, 1) -> (b, d, Lq, vs - 1)
    acmlt_p_attn = \
        torch.matmul(p_attn.unsqueeze(-2).unsqueeze(-2),
                     bool_relmat.unsqueeze(1).unsqueeze(-1)
                     ).squeeze(-1).squeeze(-1)

    # acc_p_attn: -> (b, h, Lq, 1, vs - 1)
    # emb: -> (1, h, 1, vs, d)
    # rel_scores: (b, h, Lq, 1, d) -> (b, h, Lq, d)

    start_ind = 1 if ignore_zero else 0
    rel_values = \
        torch.matmul(acmlt_p_attn.unsqueeze(-2),
                     emb[:, start_ind:].unsqueeze(0).unsqueeze(-3)
                     ).squeeze(-2)
    return rel_values


"""
Sub-modules of transformer
Code is copied and *modified* from 
https://github.com/harvardnlp/annotated-transformer:
- added generalized RPE
- re-write parts to make it consistent with the original Transformer paper.
"""


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, relation_vocab_sizes=None,
                 dropout=0.1):
        """
        relation_vacab_sizes: vocab size of each considered relation
        (S in the MuseBERT paper). If None, no GRPE applied.
        """

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.relation_vocab_sizes = relation_vocab_sizes
        self.ignore_zero = True

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linears_in = nn.ModuleList([self.linear_q,
                                         self.linear_k,
                                         self.linear_v])

        self.attn = None
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        # in our implementation, embeddings are different for each head.
        if relation_vocab_sizes is not None:
            self.rel_score_emb = \
                nn.ModuleList([nn.Embedding(vcs, self.d_k,
                                            padding_idx=0)
                               for vcs in relation_vocab_sizes
                               for _ in range(h)])

            self.rel_value_emb = \
                nn.ModuleList([nn.Embedding(vcs, self.d_k,
                                            padding_idx=0)
                               for vcs in relation_vocab_sizes
                               for _ in range(h)])

    @property
    def rel_score_lists(self):
        """
        Returns a list of length=len(self.relation_vocab_sizes)
        Sizes of each element (self.h, vocab_sizes[i], self.d_model)
        """
        return \
            [torch.stack(
                [self.rel_score_emb[i * self.h + j](
                    torch.arange(0, vcs).to(device))
                 for j in range(self.h)],
                0)
                for i, vcs in enumerate(self.relation_vocab_sizes)] \
            if self.relation_vocab_sizes is not None else None

    @property
    def rel_value_lists(self):
        """
        Returns a list of length=len(self.relation_vocab_sizes)
        Sizes of each element (self.h, vocab_sizes[i], self.d_model)
        """
        return \
            [torch.stack(
                [self.rel_value_emb[i * self.h + j](
                    torch.arange(0, vcs).to(device))
                 for j in range(self.h)],
                0) for i, vcs in enumerate(self.relation_vocab_sizes)] \
            if self.relation_vocab_sizes is not None else None

    def relative_position_score(self, query, rel_mat):
        rel_score = sum([query_multiply_rel_pos(query, rel_sc, rel_mat[:, i])
                         for i, rel_sc in enumerate(self.rel_score_lists)])
        return rel_score

    def relative_position_value(self, rel_mat):
        return \
            sum([compute_rel_attn_value(self.attn, rel_mat[:, i],
                                        emb, ignore_zero=self.ignore_zero)
                 for i, emb in enumerate(self.rel_value_lists)])

    def relative_positional_attention(self, query, key, value, mask, rel_mat):
        """Generalized RPE applied here"""

        rel_score = self.relative_position_score(query, rel_mat)

        # first term of eq. (11) under distributive law.
        x, self.attn = attention(query, key, value, rel_scores=rel_score,
                                 mask=mask, dropout=self.dropout)

        # added with second term of eq. (11).
        x += self.relative_position_value(rel_mat)
        return x

    def vanilla_attention(self, query, key, value, mask):
        x, self.attn = attention(query, key, value, rel_scores=None,
                                 mask=mask, dropout=self.dropout)
        return x

    def attention(self, query, key, value, mask, rel_mat):
        if rel_mat is None or self.relation_vocab_sizes is None:
            return self.vanilla_attention(query, key, value, mask)
        else:
            return self.relative_positional_attention(query, key, value,
                                                      mask, rel_mat)

    def forward(self, query, key, value, rel_mat, mask=None):
        """
        q, k, v: (bs, L, d_model)
        mask: (bs, L, L) or (1, L, L)
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        bs = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [ln(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
             for ln, x in zip(self.linears_in, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, mask, rel_mat)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(bs, -1, self.h * self.d_k)
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

    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, h, d_model, d_ff, non_linear, relation_vocab_sizes,
                 dropout=0.1, attn_dropout=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model, relation_vocab_sizes,
                                              attn_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, non_linear,
                                                    dropout)
        self.sublayer_attn = (SublayerConnection(d_model, dropout))
        self.sublayer_ff = (SublayerConnection(d_model, dropout))

    def forward(self, x, rel_mat, mask):
        x = self.sublayer_attn(x, lambda x: self.self_attn(x, x, x,
                                                           rel_mat, mask))
        return self.sublayer_ff(x, self.feed_forward)


class TransformerEncoder(nn.Module):

    def __init__(self, N, h, d_model, d_ff, non_linear, relation_vocab_sizes,
                 dropout=0.1, attn_dropout=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(h, d_model, d_ff, non_linear, relation_vocab_sizes,
                          dropout, attn_dropout)
             for _ in range(N)])
        self.d_model = d_model
        self.N = N

    def forward(self, x, rel_mat, mask):
        for layer in self.layers:
            x = layer(x, rel_mat, mask)
        return x

