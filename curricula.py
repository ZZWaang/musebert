from note_attribute_repr import NoteAttributeAutoEncoder, Sampler
from note_attribute_corrupter import SimpleCorrupter
from dataset import PolyphonicDataLoaders, PolyphonicDataset, NoteMatrixDataset
from musebert_model import MuseBERT
from curriculum_preset import *


def prepare_data_loaders(atr_autoenc, corrupter, batch_size):
    train_set = PolyphonicDataset(NoteMatrixDataset.get_train_dataset(), -6, 5,
                                  atr_autoenc, corrupter)
    val_set = PolyphonicDataset(NoteMatrixDataset.get_val_dataset(), 0, 0,
                                atr_autoenc, corrupter)
    data_loaders = \
        PolyphonicDataLoaders.get_loaders(batch_size, batch_size, train_set, val_set,
                                          True, False)
    return data_loaders


def prepare_model(loss_inds, relation_vocab_sizes=(5, 5, 5, 5)):
    return MuseBERT.\
        init_model(relation_vocab_sizes=relation_vocab_sizes,
                   loss_inds=loss_inds)


class Curriculum:

    """
    A class to handle four types of (hyper-)parameters
    - autoenc_dict: R_base <-> R_fac conversion
    - corrupter_dict: BERT-like corruption parameters
    - model_dict: MuseBERT parameters
    - train_dict: training parameters and learning rate parameters
    """

    def __init__(self, autoenc_dict, corrupter_dict, model_dict, train_dict):
        self.autoenc_dict = autoenc_dict
        self.corrupter_dict = corrupter_dict
        self.model_dict = model_dict
        self.train_dict = train_dict
        self.consistency_check()
        self.autoenc = NoteAttributeAutoEncoder(**self.autoenc_dict)
        self.corrupter = SimpleCorrupter(**self.corrupter_dict)

    def consistency_check(self):
        assert self.autoenc_dict['nmat_pad_length'] == \
               self.autoenc_dict['atr_mat_pad_length'] == \
               self.corrupter_dict['pad_length']
        assert tuple(np.where(np.array(self.train_dict['beta']) != 0)[0]) == \
               self.model_dict['loss_inds']

    def prepare_data(self):
        # prepare data_loaders
        autoenc = NoteAttributeAutoEncoder(**self.autoenc_dict)
        corrupter = SimpleCorrupter(**self.corrupter_dict)
        return prepare_data_loaders(autoenc, corrupter, self.train_dict['batch_size'])

    def prepare_model(self, device):
        return prepare_model(**self.model_dict).to(device)

    def reset_batch_size(self, new_bs):
        self.train_dict['batch_size'] = new_bs

    @property
    def beta(self):
        return self.train_dict['beta']

    @property
    def lr(self):
        return self.train_dict['lr_dict']

    def __call__(self, device):
        data_loaders = self.prepare_data()
        model = self.prepare_model(device)
        return data_loaders, model


# curriculum for used for pre-training
all_curriculum = Curriculum(default_autoenc_dict,
                            all_corrupter_dict,
                            all_model_dict,
                            all_train_dict)


# curricula for fine-tuning specific attributes
onset_ft_curriculum = Curriculum(default_autoenc_dict,
                                 onset_corrupter_dict,
                                 onset_model_dict,
                                 onset_train_ft_dict)

pitch_ft_curriculum = Curriculum(default_autoenc_dict,
                                 pitch_corrupter_dict,
                                 pitch_model_dict,
                                 pitch_train_ft_dict)

duration_ft_curriculum = Curriculum(default_autoenc_dict,
                                    duration_corrupter_dict,
                                    duration_model_dict,
                                    duration_train_ft_dict)
