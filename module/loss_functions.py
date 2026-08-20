# adapted from https://github.com/princeton-vl/RAFT-Stereo/blob/6068c1a26f84f8132de10f60b2bc0ce61568e085/train_stereo.py#L35

import torch


def sequence_loss(preds, gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of predictions """

    n_predictions = len(preds)
    assert n_predictions >= 1
    loss = 0.0

    assert valid.shape == gt.shape
    assert not torch.isinf(gt[valid.bool()]).any()
    # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
    adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))

    for i in range(n_predictions):
        assert not torch.isnan(preds[i]).any() and not torch.isinf(preds[i]).any()
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (preds[i] - gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, gt.shape, preds[i].shape]
        loss += i_weight * i_loss[valid.bool()].mean()
    return loss
