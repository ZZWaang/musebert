import torch
from torch import nn
from torch.nn import functional as F


class ManyToOneAttention(nn.Module):

    def __init__(self, input_dim, attn_dim, num_query):
        super(ManyToOneAttention, self).__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.num_query = num_query
        self.key_fc = nn.Linear(input_dim, attn_dim)
        self.value_fc = nn.Linear(input_dim, attn_dim)
        self.queries = nn.Parameter(torch.rand(num_query, attn_dim))

    def forward(self, h):
        # h: (bs, seq_len, input_dim)
        bs = h.size(0)
        key = self.key_fc(h)
        value = self.value_fc(h)
        query = self.queries.unsqueeze(0).repeat(bs, 1, 1)
        attn_weight = F.softmax(torch.bmm(query, key.transpose(1, 2)), dim=-1)
        attn_applied = torch.bmm(attn_weight, value)
        return attn_applied
