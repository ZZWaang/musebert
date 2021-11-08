import numpy as np
from typing import Union


"""Functions to encode R_base to R_fac."""


def onset_to_onset_attributes(onset, eo=0):
    """
    Convert an array of onsets (o) to factorized (o_bt, o_sub).
    - eo is a random int in range(0, 4), making the encoding stochastic.
    """

    def o_to_o_bt(o, eo):
        return (o + eo) // 4

    def o_o_bt_to_o_sub(o, o_bt):
        return (o - 4 * o_bt) % 7

    o_beat = o_to_o_bt(onset, eo)

    o_subdiv = o_o_bt_to_o_sub(onset, o_beat)

    return np.stack([o_beat, o_subdiv], -1)


def estimate_ep_by_maxmin(p, w):
    def mfilter(x):
        return min(max(int(x), -6), 6)

    return mfilter(mfilter(60 - (p.max() + p.min()) / 2) + w)


def pitch_to_pitch_attributes(pitch, ep=0):
    """
    Convert an array of pitches (p) to factorized (p_hig, p_reg, p_deg).
    - ep is a random int in range(-6, 7), making the encoding stochastic.
    """

    def compute_pitch_range_indices(p):

        # left range: the low registers
        lr = p < 24

        # right range: the high registers
        rr = p >= 108

        # valid range: the middle registers
        vr = np.logical_not(np.logical_and(lr, rr))

        return vr, lr, rr

    def p_to_p_hig(p, ep, vr, lr, rr):
        p_hig = np.zeros_like(p)
        p_hig[vr] = np.minimum(np.maximum((p[vr] + ep) // 12 - 3, 0), 4)
        p_hig[lr] = 5
        p_hig[rr] = 6
        return p_hig

    def p_p_hig_to_p_reg(p, p_hig, vr, lr, rr):
        p_reg = np.zeros_like(p)
        p_reg[vr] = p[vr] // 12 - p_hig[vr] - 2
        p_reg[lr] = p[lr] // 12
        p_reg[rr] = p[rr] // 12 - 9
        return p_reg

    def p_to_p_deg(p):
        return p % 12

    # Pitches in the low/high/normal registers are encoded differently.
    # See eq. (3) of the paper for more details.
    pitch_ranges = compute_pitch_range_indices(pitch)

    p_highness = p_to_p_hig(pitch, ep, *pitch_ranges)

    p_register = \
        p_p_hig_to_p_reg(pitch, p_highness, *pitch_ranges)

    p_degree = p_to_p_deg(pitch)

    return np.stack([p_highness, p_register, p_degree], -1)


def dur_to_dur_attributes(dur):
    """
    Convert an array of durations (d) to factorized (d_hlf, d_sqv).
    """

    def d_to_d_hlf(d):
        return d // 8

    def d_to_d_sqv(d):
        return d % 8

    d_half = d_to_d_hlf(dur)
    d_semiqvr = d_to_d_sqv(dur)
    return np.stack([d_half, d_semiqvr], -1)


def encode_note_mat_to_atr_mat(nmat,
                               length=None,
                               tgt_pad_length=None,
                               eo=0,
                               ep=0,
                               estimate_ep=False,
                               w=0):
    """
    Convert note matrix (i.e., R_base) to attribute matrix (i.e., R_fac).

    :param nmat: The input X_base note matrix.
    :param length: Number of notes in X_base.
        None if length == nmat.shape[0], i.e., no padding.
    :param tgt_pad_length: number of rows of the output X_fac.
        None if tgt_pad_length == nmat.shape[0]
    :param eo: The random shift noise added to onset.
    :param ep: The random shift noise added to pitch.
    :param estimate_ep: Whether to estimate ep by all pitches.
        - If True, input ep will not be used.
    :param w: The random shift noise added to estimates.
    :return:
      - attribute matrix (X_fac) with zero-padding.
      - ep
    """

    # initialize attribute_matrix to record encoded info from nmat.
    tgt_pad_length = nmat.shape[0] \
        if tgt_pad_length is None else tgt_pad_length
    atr_mat = np.zeros((tgt_pad_length, 7), dtype=np.int64)

    # reassign length if length is None, assuming no padding.
    length = nmat.shape[0] if length is None else length

    # the trivial case returns the all-zero matrix.
    if length == 0:
        return atr_mat, None

    # (o, p, d) in X_base.
    onset, pitch, dur = \
        nmat[0: length, 0], nmat[0: length, 1], nmat[0: length, 2]

    # output row [0: 2]: o_bt, o_sub
    atr_mat[0: length, 0: 2] = onset_to_onset_attributes(onset, eo)

    # estimate ep or use the given one
    ep = estimate_ep_by_maxmin(pitch, w) if estimate_ep else ep
    # output row [2: 5]: p_hig, p_reg, p_deg
    atr_mat[0: length, 2: 5] = pitch_to_pitch_attributes(pitch, ep)

    # output row [5: 7]: d_hlf, d_sqv
    atr_mat[0: length, 5:] = dur_to_dur_attributes(dur)

    return atr_mat, (ep,)


"""Functions to decode R_fac to R_base."""


def onset_attributes_to_onset(o_beat, o_subdiv):
    """
    Convert arrays of o_bt and o_sub to o.
    - See eq. (1) of the paper.
    """

    def o_bt_o_sub_to_o(o_bt, o_sub):

        def convert_sub(x):
            """ o_sub from its 2's complement representation to exact value."""
            return (x + 3) % 7 - 3

        return (4 * o_bt + convert_sub(o_sub)) % 32

    o = o_bt_o_sub_to_o(o_beat, o_subdiv)

    return o


def pitch_attributes_to_pitch(p_highness, p_register, p_degree):
    """
    Convert arrays of p_hig, p_reg and p_deg to p.
    - See eq. (3) of the paper.
    """

    def compute_pitch_register_range_indices(hig):
        lr = hig == 5
        rr = hig == 6
        vr = np.logical_not(np.logical_and(lr, rr))
        return vr, lr, rr

    def p_hig_p_reg_p_deg_to_p(p_hig, p_reg, p_deg, vr, lr, rr):
        p = np.zeros_like(p_hig)
        p[vr] = 24 + 12 * (p_hig[vr] + p_reg[vr]) + p_deg[vr]
        p[lr] = 12 * p_reg[lr] + p_deg[lr]
        p[rr] = 108 + 12 * p_reg[rr] + p_deg[rr]
        return p

    pitch_ranges = compute_pitch_register_range_indices(p_highness)
    p = p_hig_p_reg_p_deg_to_p(p_highness, p_register,
                               p_degree, *pitch_ranges)

    return p


def dur_attributes_to_dur(d_half, d_semiqvr):
    """
    Convert arrays of d_hlf and d_sqv to d.
    - See eq. (2) of the paper.
    """

    def d_hlf_dur_sqv_to_d(d_hlf, d_sqv):
        return 8 * d_hlf + d_sqv

    d = d_hlf_dur_sqv_to_d(d_half, d_semiqvr)

    return d


def decode_atr_mat_to_nmat(atr_mat, length=None, tgt_pad_length=None):
    """
    Convert attribute matrix (i.e., R_fac) to note matrix (i.e., R_base).

    :param atr_mat: The input X_fac attribute matrix.
    :param length: Number of notes in X_fac.
        None if length == atr_mat.shape[0], i.e., no padding.
    :param tgt_pad_length: number of rows of the output X_base.
        None if tgt_pad_length == atr_mat.shape[0]
    :return: note matrix (X_base) with zero-padding.
    """

    # initialize attribute_matrix to record encoded info from nmat.
    tgt_pad_length = atr_mat.shape[0] \
        if tgt_pad_length is None else tgt_pad_length
    nmat = np.zeros((tgt_pad_length, 3), dtype=np.int64)

    # reassign length if length is None, assuming no padding.
    length = atr_mat.shape[0] if length is None else length

    # the trivial case returns the all-zero matrix.
    if length == 0:
        return nmat

    onset_bt, onset_sub, pitch_hig, pitch_reg, pitch_deg, dur_hlf, dur_sqv = \
        (atr_mat[0: length, i] for i in range(atr_mat.shape[1]))
    # pitch_ranges = compute_pitch_register_range_indices(pitch_hig)

    # row 0: onset
    nmat[0: length, 0] = onset_attributes_to_onset(onset_bt, onset_sub)

    # row 1: pitch
    nmat[0: length, 1] = \
        pitch_attributes_to_pitch(pitch_hig, pitch_reg, pitch_deg)

    # row 2: duration
    nmat[0: length, 2] = dur_attributes_to_dur(dur_hlf, dur_sqv)

    return nmat


class Sampler:

    """
    A multinoulli sampler in the following form:
        x = random_sample(x) if x is None else x
    """

    def __init__(self, low, high, dist):
        """
        :param low: lower bound
        :param high: upper bound
        :param dist: the distribution
        :param normalize: whether to normalize the distribution
        """

        self.low = low
        self.high = high
        self.dist = dist

    def sample(self, x=None):
        """ Do a sample from the distribution unless x provided. """
        return x if x is not None \
            else np.random.choice(np.arange(self.low, self.high), p=self.dist)


class NoteAttributeAutoEncoder:

    """
    Encoder-decoder of note_mat <-> atr_mat.
    - Encoder is a stochastic process.
    - Decoder is deterministic.
    - self.fast_mode() returns partial results, the matrix only.
    - self.regular_mode() returns all information, including the matrix,
        length, and a record of randomness.
    """
    eo_dist: Union[Sampler, None]
    ep_dist: Union[Sampler, None]
    w_dist: Union[Sampler, None]

    def __init__(self, eo_dist, ep_dist, w_dist, estimate_ep=True,
                 nmat_pad_length=100, atr_mat_pad_length=100, _fast_mode=True):
        """
        :param eo_dist: Random noise added to onset attribute encode.
        :param ep_dist: Random noise added to pitch attribute encode.
        :param w_dist: Random noise added to ep estimator.
        :param estimate_ep: Whether to compute ep hinted by estimation.
        :param nmat_pad_length: pad length of nmat after decode.
        :param atr_mat_pad_length: pad length of atr_mat after encode.
        :param _fast_mode: controls the mode.
        """

        self.estimate_ep = estimate_ep
        self.eo_dist = eo_dist
        self.ep_dist = ep_dist
        self.w_dist = w_dist
        self.nmat_pad_length = nmat_pad_length
        self.atr_mat_pad_length = atr_mat_pad_length
        self._fast_mode = True

    def fast_mode(self):
        self._fast_mode = True

    def regular_mode(self):
        self._fast_mode = False

    def eo_sampler(self, eo=None):
        if self.eo_dist is None:
            return
        return self.eo_dist.sample(eo)

    def ep_sampler(self, ep=None):
        if self.ep_dist is None:
            return
        return self.ep_dist.sample(ep)

    def w_sampler(self, w=None):
        if self.w_dist is None:
            return
        return self.w_dist.sample(w)

    def _full_encode(self, nmat, length, eo=None, ep=None, w=None):
        eo = self.eo_sampler(eo)
        ep = self.ep_sampler(ep)
        w = self.w_sampler(w)
        atr_mat, ep = encode_note_mat_to_atr_mat(nmat, length,
                                                 self.atr_mat_pad_length,
                                                 eo, ep,
                                                 self.estimate_ep,
                                                 w)
        return atr_mat, length, {'ep': ep, 'eo': eo, 'w': w}

    def _partial_encode(self, nmat, length):
        eo = self.eo_sampler()
        ep = self.ep_sampler()
        w = self.w_sampler()
        atr_mat, _ = encode_note_mat_to_atr_mat(nmat, length,
                                                self.atr_mat_pad_length,
                                                eo, ep,
                                                self.estimate_ep,
                                                w)
        return atr_mat, length

    def encode(self, nmat, length, eo=None, ep=None, w=None):
        if self._fast_mode:
            return self._partial_encode(nmat, length)
        else:
            return self._full_encode(nmat, length, eo, ep, w)

    def _full_decode(self, atr_mat, length):
        return decode_atr_mat_to_nmat(atr_mat, length, self.nmat_pad_length), \
               length

    def _partial_decode(self, atr_mat, length):
        return decode_atr_mat_to_nmat(atr_mat, length, self.nmat_pad_length)

    def decode(self, atr_mat, length):
        if self._fast_mode:
            return self._partial_decode(atr_mat, length)
        else:
            return self._full_decode(atr_mat, length)
