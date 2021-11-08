import numpy as np
import operator
from note_attribute_repr import pitch_attributes_to_pitch, \
    onset_attributes_to_onset


"""Compute a stack of relation matrices """


def compute_relation_mat_from_atr_mat(atr_mat, length):
    """
    Compute relation matrix of binary pitch and onset relations from atr_mat
    (i.e. X_fac in the MuseBERT paper.)
    - S = {o, p, o_bt, p_hig} relations are considered.
    - The element in a relation matrix is an index indicating the relation.
    - The index for relation matrix is hard-coded:
      - 0: pad
      - 1: equal
      - 2: greater
      - 3: less
      - 4: Unknown

    :param atr_mat: The X_fac of size (L, 7).
    :param length: actual length of L (length without padding).
      if None, length = atr_mat.shape[0]
    :param unknown_inds: inds masked with UNK, a bool array.
    :return: relation matrix of size (4, L, L).
        Stacked onset and pitch relations in the order: (o, p, o_bt, p_hig)
    """

    # initialize relation matrix to record pitch and onset relation.
    relation_mat = np.zeros((4, atr_mat.shape[0], atr_mat.shape[0]),
                            dtype=np.int8)

    # reassign length if length is None, assuming no padding.
    length = atr_mat.shape[0] if length is None else length

    # the trivial case returns the all-zero matrix.
    if length == 0:
        return relation_mat

    relation_mat_ = np.zeros((4, length, length),
                             dtype=np.int8)

    def relation(x, f):
        return f(np.expand_dims(x, 0), np.expand_dims(x, -1))

    def eq_relation(x):
        return relation(x, operator.__eq__)

    def gt_relation(x):
        return relation(x, operator.__gt__)

    def lt_relation(x):
        return relation(x, operator.__lt__)

    def eq_gt_lt_relations(x):
        eq_rel = eq_relation(x)
        gt_rel = gt_relation(x)
        lt_rel = lt_relation(x)
        no_rel = np.logical_not(np.logical_or(np.logical_or(eq_rel, gt_rel),
                                              lt_rel))
        return np.stack([no_rel, eq_rel, gt_rel, lt_rel], 0)

    def write_relation_on_mat(stacked_rel, i):
        relation_mat_[i] = stacked_rel.astype(np.int8).argmax(0)

    o_bt, o_sub, p_hig, p_reg, p_deg = \
        (atr_mat[0: length, i] for i in range(0, 5))
    pitch = pitch_attributes_to_pitch(p_hig, p_reg, p_deg)
    onset = onset_attributes_to_onset(o_bt, o_sub)

    # the following *_eqgtlt is a stacked (4, length, length) bool array.
    # each layer is bool mat of no_rel, eq, gt, lt.
    p_eqgtlt = eq_gt_lt_relations(pitch)
    o_eqgtlt = eq_gt_lt_relations(onset)
    p_hig_eqgtlt = eq_gt_lt_relations(p_hig)
    o_bt_eqgtlt = eq_gt_lt_relations(o_bt)

    # write each relation on the relation_mat_
    for i, stk_rel in enumerate([o_eqgtlt, p_eqgtlt,
                                 o_bt_eqgtlt, p_hig_eqgtlt]):
        write_relation_on_mat(stk_rel, i)

    relation_mat[:, 0: length, 0: length] = relation_mat_
    return relation_mat


def corrupt_relation_mat(relation_mat, mask, mask_val):
    """ Corrupt relation matrix by setting masked places to mask_val. """
    corrupted_rel_mat = relation_mat.copy()
    corrupted_rel_mat[mask] = mask_val
    return corrupted_rel_mat


""" Corrupt R_fac and relation matrices"""


class CorrupterTemplate:

    """
    An A=abstract class to handle different types of BERT_like data corruption.
    - fast mode only returns the output together with necessary record of
      randomness.
    - regular mode returns the output together with a complete record of
      randomness.
    """

    def __init__(self, pad_length):
        self.pad_length = pad_length

        self._fast_mode = True

        self.unknown_values = (9, 7, 7, 3, 12, 5, 8)  # masked vocab id

        # list(range(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds))
        # is the vocab sizes of attributes.
        self.lower_bounds = (0, 0, 0, 0, 0, 0, 0)
        self.upper_bounds = (9, 7, 7, 3, 12, 5, 8)

        self.mask_val = 4  # mask_val (vocab_id) for relmat

    def fast_mode(self):
        self._fast_mode = True

    def regular_mode(self):
        self._fast_mode = False

    def corrupt_unknown(self, x, inds, col_id):
        """ Corruption operation: mask """
        if len(inds) == 0:
            return
        x[inds, col_id] = self.unknown_values[col_id]

    def corrupt_assign(self, x, inds, col_id):
        """ Corruption operation: replace """
        n_inds = len(inds)
        if n_inds == 0:
            return
        x[inds, col_id] = np.random.randint(self.lower_bounds[col_id],
                                            self.upper_bounds[col_id],
                                            n_inds)

    def _inds_to_bool_array(self, inds):
        inds_bool = np.zeros(self.pad_length, dtype=np.bool)
        inds_bool[inds] = True
        return inds_bool


class SimpleCorrupter(CorrupterTemplate):

    """
    Corrupter used in MuseBERT pre-training. The operations includes:
    - random select rows (note) of some probability.
    - select columns (attribute) by some probability or by definition.
    - corrupt an attribute with
        - Mask
        - Replace
        - Unchange
    """

    def __init__(self, pad_length, corrupt_col_ids,
                 mask_ratio=0.15, unchange_ratio=0.1, unknown_ratio=0.8,
                 relmat_cpt_ratio=0.5):
        """
        The data is assumed to have shape (L, D). Mask will be applied to
        selected indices in D.
        :param pad_length:
            Integer value of total length of attribute matrix
            (including padding).
        :param corrupt_col_ids: the column ids to corrupt
        :param mask_ratio:
            Ratio to apply a mask.
        :param unchange_ratio:
            In the case of applying mask, the ratio of unchanging the token.
        :param unknown_ratio:
            In the case of applying mask, the ratio of changing it to <MASK>.
        :param relmat_cpt_ratio: the ratio to mask an element in the relmat
            where an entry is "masked" (by either of the three methods).
        """

        super(SimpleCorrupter, self).__init__(pad_length)

        self.corrupt_col_ids = corrupt_col_ids
        self.n_cpt_cols = len(corrupt_col_ids)

        self.mask_ratio = mask_ratio
        self.unchange_ratio = unchange_ratio
        self.unknown_ratio = unknown_ratio
        self.modify_ratio = 1 - unchange_ratio - unknown_ratio

        self.atr_cpt_method_dict = {}
        self.relmat_cpt_ratio = relmat_cpt_ratio

    def clear_dict(self):
        self.atr_cpt_method_dict = {}

    """ Part 1/3: Corrupt X_fac"""
    def _compute_corruption_quantities(self, length):
        """
        Determines total number of corruption: n_pos, with
        - n_modify: number of replacement
        - n_unchanged: number of token unchanged
        - n_mask = n_pos - n_modify - n_unchanged
        """

        n_pos = int(np.ceil(length * self.mask_ratio))
        if n_pos >= 3:
            # We assume there is at least one for each.
            # With some math analysis, we prefer n_unknown to be larger.
            n_modify = min(max(int(n_pos * self.modify_ratio), 1),
                           n_pos - 2)
            n_unchange = min(max(int(n_pos * self.unchange_ratio), 1),
                             n_pos - 2)
            # Implicitly, n_unknown = n_pos - n_modify - n_unchange
        else:
            n_modify = 0
            n_unchange = n_pos - 1
        return n_pos, n_modify, n_unchange

    @staticmethod
    def _select_corruption_inds(length, n_pos):
        """Randomly select inds/rows to corrupt."""
        inds = np.random.choice(np.arange(0, length), n_pos, replace=False)
        return inds

    @staticmethod
    def _assign_corruption_methods(n_modify, n_unchange, inds):
        """Assign one of the three corruption methods to selected rows."""
        inds = np.random.choice(inds, len(inds), replace=False)
        modify_inds = inds[0: n_modify]
        unchange_inds = inds[n_modify: n_modify + n_unchange]
        unknown_inds = inds[n_modify + n_unchange:]
        return unchange_inds, unknown_inds, modify_inds

    def corrupt_operations(self, data, n_modify, n_unchange, inds):
        """
        This function does *in-place* corruption operation to data.
        The returned matrix is only endured with replacement
          (without being masked.)
        """

        modified_only_mat = data.copy()
        for col_id in self.corrupt_col_ids:
            unchange_inds, unknown_inds, modify_inds = \
                self.__class__._assign_corruption_methods(n_modify, n_unchange,
                                                          inds)
            self.corrupt_assign(data, modify_inds, col_id)
            modified_only_mat[:, col_id] = data[:, col_id]
            self.corrupt_unknown(data, unknown_inds, col_id)
            self.atr_cpt_method_dict[col_id] = (unchange_inds,
                                                unknown_inds,
                                                modify_inds)
        return modified_only_mat

    def _corrupt_atr_mat(self, atr_mat, length, return_dict=False, inds=None):
        """
        :param atr_mat: The data to corrupt. The data will be copied before
            corruption.
        :param length: Effective length of the data.
        :param return_dict: whether to return a dict of methods applies to
            masked inds.
        """

        data = atr_mat.copy()

        # the trivial case
        if length == 0:
            inds_array = np.zeros(self.pad_length, dtype=np.bool) \
                if inds is None else inds
            if return_dict:
                return data, inds_array, data, {}
            else:
                return data, inds_array, data

        # determines the number of corruptions
        n_pos, n_modify, n_unchange = \
            self._compute_corruption_quantities(length)

        # randomly select rows to corrupt
        inds = self.__class__._select_corruption_inds(length, n_pos) \
            if inds is None else np.where(inds)[0]

        # data corruption is applied in in-place fashion. The returned mat is
        # a copy of replacement-only intermediate product.
        self.clear_dict()
        modified_only_mat = \
            self.corrupt_operations(data, n_modify, n_unchange, inds)

        inds_array = self._inds_to_bool_array(inds)
        if return_dict:
            return data, length, inds_array, modified_only_mat,\
                self.atr_cpt_method_dict
        else:
            return data, length, inds_array, modified_only_mat

    def corrupt_atr_mat(self, atr_mat, length, inds):
        if self._fast_mode:
            return self._corrupt_atr_mat(atr_mat, length, False, inds)
        else:
            return self._corrupt_atr_mat(atr_mat, length, True, inds)

    """Part 2/3: Corrupt a stack of relation matrices"""

    def _generate_mask(self, relation_mat, length, mask_inds):
        """Generate a symmetrical random mask under self.relmat_cpt_ratio."""
        mask_of_ratio = np.random.rand(*relation_mat.shape)
        mask_of_ratio = (mask_of_ratio + mask_of_ratio.transpose((0, 2, 1))) \
            > 2 * (1 - self.relmat_cpt_ratio)

        mask = np.zeros_like(relation_mat[0], dtype=np.bool)
        mask[mask_inds, 0: length] = True
        mask[0: length, mask_inds] = True
        np.fill_diagonal(mask, False)

        return np.logical_and(mask, mask_of_ratio)

    def _corrupt_rel_mat(self, rel_mat, length, mask_inds, return_mask=False,
                         rel_mask=None):
        """Mask a stack of relation matrices."""
        mask = self._generate_mask(rel_mat, length, mask_inds) \
            if rel_mask is None else rel_mask

        corrupted_relmat = corrupt_relation_mat(rel_mat, mask, self.mask_val)

        if return_mask:
            return corrupted_relmat, mask
        else:
            return (corrupted_relmat,)

    def corrupt_rel_mat(self, rel_mat, length, mask_inds, rel_mask):
        if self._fast_mode:
            return self._corrupt_rel_mat(rel_mat, length, mask_inds,
                                         False, rel_mask)
        else:
            return self._corrupt_rel_mat(rel_mat, length, mask_inds,
                                         True, rel_mask)

    """Part 3/3: Main"""

    def compute_relmat_and_corrupt_atrmat_and_relmat(self, atr_mat, length,
                                                     inds=None, rel_mask=None):
        """
        The entry method called by dataset class.

        The corruption logic:
        - X_fac is corrupted in BERT-like fashion.
        - Relmat is computed and corrupted at the same time, because otherwise,
          once note attributes are replaced, the relation should be recomputed.
          1) We keep a record of intermediate corruption of X_fac where
            replacement is applied but mask is not (i.e., modify_only_mat).
          2) We compute the relation matrices of modify_only_mat as if it is
            the ground truth data.
          3) We generate a random sysmetrical mask to mask the relation
            matrices.
        """

        # BERT-like corruption to X_Fac. The intermediate result is stored in
        # modify_only_mat.
        cpt_atr_mat = self.corrupt_atr_mat(atr_mat, length, inds)
        modify_only_mat = cpt_atr_mat[3]

        # compute relation matrices (replacement considered)
        rel_mat = compute_relation_mat_from_atr_mat(modify_only_mat, length)

        # corrupt relation matrices (with mask only)
        cpt_rel_mat = self.corrupt_rel_mat(rel_mat, length, cpt_atr_mat[2],
                                           rel_mask)
        return (*cpt_atr_mat, *cpt_rel_mat)



