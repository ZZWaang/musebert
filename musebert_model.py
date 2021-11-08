from transformer import TransformerEncoder
from amc_dl.torch_plus.module import PytorchModel
import numpy as np
import torch.nn as nn
import torch


class MuseBERT(PytorchModel):

    tfm: TransformerEncoder

    def __init__(self, name, device, tfm, in_dims, out_dims, loss_inds):
        """
        :param name: name of the model, e.g., 'musebert'
        :param device: cpu or cuda
        :param tfm: transformer encoder
        :param in_dims: intput vocab sizes
        :param out_dims: output vocab sizes (distribution sizes)
        :param loss_inds: (which attributes to compute recon loss)
        """

        def compute_lr_inds(dims):
            dims = np.array(dims, dtype=np.int64)
            r_inds = np.cumsum(dims)
            l_inds = np.insert(r_inds[0: -1], 0, 0)
            return tuple(l_inds), tuple(r_inds)

        super(MuseBERT, self).__init__(name, device)

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.n_col = len(in_dims)

        # the left, right endpoints for output distributions of attributes
        self.lout_inds, self.rout_inds = compute_lr_inds(out_dims)

        self.loss_inds = loss_inds

        self.embeddings = \
            nn.ModuleList([nn.Embedding(ind, tfm.d_model)
                           for ind in in_dims])
        self.tfm = tfm
        self.out = nn.Linear(tfm.d_model, self.rout_inds[-1])
        self.lossf = nn.CrossEntropyLoss(reduction='none')

    @property
    def d_model(self):
        return self.tfm.d_model

    @property
    def N(self):
        return self.tfm.N

    @staticmethod
    def _truncate_data(x, length=None, truncate_len=None, mode=0):
        """Apply data truncation in three different modes."""
        truncate_len = truncate_len if truncate_len is not None \
            else (length.max() if length is not None else None)
        if truncate_len is not None:
            if mode == 0:
                return x[:, 0: truncate_len]
            elif mode == 1:
                return x[:, 0: truncate_len, 0: truncate_len]
            elif mode == 2:
                return x[:, :, 0: truncate_len, 0: truncate_len]
            else:
                raise NotImplementedError
        return x

    def _truncate_lists(self, xs, length, modes):
        truncate_len = length.max()
        return tuple(self.__class__._truncate_data(
            x, None, truncate_len, m
        ) for x, m in zip(xs, modes))

    def truncate(self, data_in, mask, rel_mat, data, inds, length):
        """truncate the input data to max(length)."""
        return self._truncate_lists([data_in, mask, rel_mat, data, inds],
                                    length,
                                    modes=[0, 1, 2, 0, 0])

    def onset_pitch_dur_embedding(self, data_in):
        """ Sum up the onset, pitch, dur embeddings """
        return sum([self.embeddings[i](data_in[:, :, i])
                    for i in range(len(self.in_dims))])

    def run(self, data_in, rel_mat, mask):
        """
        batch -> output distribution
        :param data_in: (bs, L, 3) dtype long. Last dim: onset, pitch, dur.
        :param rel_mat: (bs, k, L, L)
        :param mask: (bs, L, L) dtype long
        :return: (bs, L, pitch + dur dims)
        """
        x = self.onset_pitch_dur_embedding(data_in)
        x = self.tfm(x, rel_mat, mask=mask)
        x = self.out(x)
        return x

    def loss_function(self, recon, tgt, inds, beta):
        """compute reconstuction loss on corrupted attributes (inds) only."""

        def atr_loss(recon, tgt, i):
            if i in self.loss_inds:
                l_ind = self.lout_inds[i]
                r_ind = self.rout_inds[i]
                return (self.lossf(recon[:, l_ind: r_ind],
                                   tgt[:, i]) * w).sum()
            else:
                return torch.zeros([]).float().to(self.device)

        # compute weight w: so that each data sample is treated equally.
        # E.g., when bs=2, 1st sample has 1 corrupted tokens and 2nd has 2,
        #  w = [0.5, 0.25, 0.25], not [0.333, 0.333, 0.333].
        counts = inds.long().sum(-1)
        bs = inds.size(0)
        w = torch.cat([torch.tensor([1 / c.float()] * c.long())
                       for c in counts], 0) / bs
        w = w.to(self.device)
        recon = recon[inds]  # (*, outs)
        tgt = tgt[inds]
        losses = [atr_loss(recon, tgt, i) for i in range(self.n_col)]

        # beta controls the weighting for different attributes
        total_loss = sum([ls * b for ls, b in zip(losses, beta)])

        return (total_loss, *losses)

    def loss(self, data, data_in, rel_mat, mask, inds, length, beta):
        data_in, mask, rel_mat, data, inds = \
            self.truncate(data_in, mask, rel_mat, data, inds, length)
        recon = self.run(data_in, rel_mat, mask)
        loss = self.loss_function(recon, data, inds, beta)
        return loss

    def inference(self, data, data_in, rel_mat, mask, inds,
                  length, truncate=True):
        self.eval()
        with torch.no_grad():
            if truncate:
                data_in, mask, rel_mat, data, inds = \
                    self.truncate(data_in, mask, rel_mat, data, inds, length)
            recon = self.run(data_in, rel_mat, mask)
        return recon

    @classmethod
    def init_model(cls, N=12, h=8, d_model=128, d_ff=512, non_linear=None,
                   relation_vocab_sizes=(5, 5, 5, 5),
                   in_dims=(15, 15, 15, 15, 15, 15, 15),
                   out_dims=(9, 7, 7, 3, 12, 5, 8),
                   loss_inds=(1, 3, 4, 5, 6),
                   dropout=0.1):
        """Easier way to initialize a MuseBERT"""
        name = 'musebert'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if non_linear == 'gelu':
            non_linear = nn.GELU()
        tfm = TransformerEncoder(N, h, d_model, d_ff, non_linear=non_linear,
                                 relation_vocab_sizes=relation_vocab_sizes,
                                 dropout=dropout, attn_dropout=None)
        model = cls(name, device, tfm, in_dims, out_dims, loss_inds)
        model.to(device)
        return model

