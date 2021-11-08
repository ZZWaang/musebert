import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from .rnn_blocks import RnnEncoder


class Ec2vaeEncoder(nn.Module):

    def __init__(self, mel_dim=130, chord_dim=12, hidden_dim=2048,
                 zp_dim=128, zr_dim=128):
        super(Ec2vaeEncoder, self).__init__()
        self.mel_dim = mel_dim
        self.chord_dim = chord_dim
        self.input_dim = mel_dim + chord_dim
        self.zp_dim = zp_dim
        self.zr_dim = zr_dim
        self.z_dim = zp_dim + zr_dim
        self.encoder = RnnEncoder(self.input_dim, hidden_dim, self.z_dim)
    
    def forward(self, x, c):
        if c is not None:
            x = torch.cat((x, c), -1)
        mu, var = self.encoder(x)
        dist_p = Normal(mu[:, 0: self.zp_dim], var[:, 0: self.zr_dim])
        dist_r = Normal(mu[:, self.zp_dim: ], var[:, self.zp_dim: ])
        return dist_p, dist_r


class Ec2vaeDecoder(nn.Module):
    
    def __init__(self, zp_dim=128, zr_dim=128, hidden_dim=2048, mel_dim=130,
                 chord_dim=12, rhy_dim=3, n_step=32, device=None):
        super(Ec2vaeDecoder, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        self.device = device
        self.grucell_0 = nn.GRUCell(zr_dim + rhy_dim,
                                    hidden_dim)
        self.grucell_1 = nn.GRUCell(
            zp_dim + mel_dim + rhy_dim + chord_dim, hidden_dim)
        self.grucell_2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.linear_init_0 = nn.Linear(zr_dim, hidden_dim)
        self.linear_out_0 = nn.Linear(hidden_dim, rhy_dim)
        self.linear_init_1 = nn.Linear(zp_dim, hidden_dim)
        self.linear_out_1 = nn.Linear(hidden_dim, mel_dim)
        self.n_step = n_step
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.rhy_dim = rhy_dim
        self.zp_dim = zp_dim
        self.zr_dim = zr_dim
        
    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        arange = arange.to(self.device)
        x[arange, idx] = 1
        return x

    def rhythm_decoder(self, z, rhythm_gt, teacher_force=False, tfr=0.):
        out = torch.zeros((z.size(0), self.rhy_dim))
        out[:, -1] = 1.
        x = []
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        out = out.to(self.device)
        for i in range(self.n_step):
            out = torch.cat([out, z], 1)
            hx = self.grucell_0(out, hx)
            out = F.log_softmax(self.linear_out_0(hx), 1)
            x.append(out)
            if teacher_force:
                p = torch.rand(1).item()
                if p < tfr:
                    out = rhythm_gt[:, i, :]
                else:
                    out = self._sampling(out)
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def final_decoder(self, z, rhythm, condition, x_gt, teacher_force=False,
                      tfr=0.):
        out = torch.zeros((z.size(0), self.mel_dim))
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(z))
        hx[0] = t
        out = out.to(self.device)

        for i in range(self.n_step):
            if condition is not None:
                out = torch.cat([out, rhythm[:, i, :], z,
                                 condition[:, i, :]], 1)
            else:
                out = torch.cat([out, rhythm[:, i, :], z], 1)
            hx[0] = self.grucell_1(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_1(hx[1]), 1)
            x.append(out)
            if teacher_force:
                p = torch.rand(1).item()
                if p < tfr:
                    out = x_gt[:, i, :]
                else:
                    out = self._sampling(out)
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, z1, z2, condition=None, teacher_force=False,
                x_gt=None, rhythm_gt=None, tfr=0.):
        rhythm = self.rhythm_decoder(z2, rhythm_gt, teacher_force, tfr)
        return self.final_decoder(z1, rhythm, condition, x_gt,
                                  teacher_force, tfr)
    
    def get_x_gt(self, x):
        return x
    
    def get_rhythm_gt(self, x):
        rhythm_gt = x[:, :, :-2].sum(-1).unsqueeze(-1)
        rhythm_gt = torch.cat((rhythm_gt, x[:, :, -2:]), -1)
        return rhythm_gt


if __name__ == '__main__':
    x = torch.rand(5, 32, 130)
    c = torch.rand(5, 32, 12)
    enc = Ec2vaeEncoder()
    dec = Ec2vaeDecoder()
    dist1, dist2 = enc(x, c)
    z1 = dist1.rsample()
    z2 = dist2.rsample()
    print(z1.size(), z2.size())
    x_gt = dec.get_x_gt(x)
    rhy_gt = dec.get_rhythm_gt(x)
    print(x_gt.size(), rhy_gt.size())
    out = dec(z1, z2, c, True, x_gt, rhy_gt, 0.5)
    print(out.size())