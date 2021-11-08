from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, final_lr_factor,
                                    last_epoch=-1):
    """
    Copied and **modified** from: https://github.com/huggingface/transformers

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which
    it increases linearly from 0 to the initial lr set in the optimizer.

    :param optimizer: The optimizer for which to schedule the learning rate.
    :param num_warmup_steps: The number of steps for the warmup phase.
    :param num_training_steps: The total number of training steps.
    :param final_lr_factor: Final lr = initial lr * final_lr_factor
    :param last_epoch: the index of the last epoch when resuming training.
        (defaults to -1)
    :return: `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            final_lr_factor, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def augment_note_matrix(nmat, length, shift):
    """Pitch shift a note matrix in R_base format."""
    aug_nmat = nmat.copy()
    aug_nmat[0: length, 1] += shift
    return aug_nmat
