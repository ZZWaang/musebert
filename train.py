import torch
from torch import optim
from curricula import all_curriculum, Curriculum

from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, \
    ConstantScheduler, TrainingInterface
from utils import get_linear_schedule_with_warmup
from typing import Union


class TrainMuseBERT(TrainingInterface):

    def _batch_to_inputs(self, batch):
        """Convert a data batch to proper data types."""

        data, data_in, rel_mat, mask, inds, length = batch

        # data: the ground truth X_fac
        data = data.long().to(self.device)

        # data_in: the corrupted X_fac^*
        data_in = data_in.long().to(self.device)

        # rel_mat: the corrupted R_S^*.
        rel_mat = rel_mat.long().to(self.device)

        # MuseBERT mask (masking the paddings)
        mask = mask.char().to(self.device)

        # The corrupted rows.
        inds = inds.bool().to(self.device)

        # number of notes contained in each sample.
        length = length.long().to(self.device)
        return data, data_in, rel_mat, mask, inds, length


def train_musebert(parallel: bool, curriculum: Curriculum,
                   model_path: Union[None, str]=None):
    """
    The main function to train a MuseBERT model.

    :param parallel: whether to use data parallel.
    :param curriculum: input parameters
    :param model_path: None or pre-trained model path.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    readme_fn = __file__

    clip = 1
    parallel = parallel if (torch.cuda.is_available() and
                            torch.cuda.device_count() > 1) else False

    # create data_loaders and initialize model specified by the curriculum.
    data_loaders, model = curriculum(device)

    # load a pre-trained model if necessary.
    if model_path is not None:
        model.load_model(model_path, device)

    # to handle the path to save model parameters, logs etc.
    log_path_mng = LogPathManager(readme_fn)

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=curriculum.lr['lr'])
    schdl_step = len(data_loaders.train_loader) * curriculum.lr['final_epoch']
    scheduler = \
        get_linear_schedule_with_warmup(optimizer,
                                        curriculum.lr['warmup'],
                                        schdl_step,
                                        curriculum.lr['final_lr_factor'])
    optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

    # tensorboard writers
    writer_names = ['loss', 'o_bt', 'o_sub', 'p_hig', 'p_reg',
                    'p_deg', 'd_hlf', 'd_sqv']
    tags = {'loss': None}
    summary_writers = SummaryWriters(writer_names, tags,
                                     log_path_mng.writer_path)

    # keyword training parameters
    beta_scheduler = ConstantScheduler(curriculum.beta)
    params_dic = dict(beta=beta_scheduler)
    param_scheduler = ParameterScheduler(**params_dic)

    # initialize the training interface
    musebert_train = \
        TrainMuseBERT(device, model, parallel, log_path_mng, data_loaders,
                      summary_writers, optimizer_scheduler,
                      param_scheduler, curriculum.lr['n_epoch'])

    # start training
    musebert_train.run()


if __name__ == '__main__':
    # pre-training MuseBERT
    train_musebert(parallel=False, curriculum=all_curriculum)
