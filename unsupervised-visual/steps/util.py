import math
import pickle
import numpy as np
import torch
import torch.nn as nn


class DotLoss(nn.Module):

    def __init__(self, margin=1.):
        super(DotLoss, self).__init__()
        self.margin = margin

    def forward(self, image_outputs, audio_outputs):
        return dot_product_loss(image_outputs, audio_outputs, margin=self.margin)


def dot_product_loss(image_outputs, audio_outputs, margin=1.):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert (image_outputs.dim() == 2)
    assert (audio_outputs.dim() == 2)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        anchorsim = torch.dot(image_outputs[i], audio_outputs[i])
        Iimpsim = torch.dot(image_outputs[I_imp_ind], audio_outputs[i])
        Aimpsim = torch.dot(image_outputs[i], audio_outputs[A_imp_ind])
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss


def calc_recalls(image_outputs, audio_outputs):
    """
	Computes recall at 1, 5, and 10 given encoded image and audio outputs.
	"""
    S = compute_dotproduct_similarity_matrix(image_outputs, audio_outputs)
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1': A_r1.avg, 'A_r5': A_r5.avg, 'A_r10': A_r10.avg,
               'I_r1': I_r1.avg, 'I_r5': I_r5.avg, 'I_r10': I_r10.avg}
    # 'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls


def compute_dotproduct_similarity_matrix(image_outputs, audio_outputs):
    """
    Assumes image_outputs is a (batchsize, embedding_dim) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert (image_outputs.dim() == 2)
    assert (audio_outputs.dim() == 2)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
        for audio_idx in range(n):
            S[image_idx, audio_idx] = torch.dot(image_outputs[image_idx], audio_outputs[audio_idx])
    return S


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every lr_decay epochs"""
    lr = base_lr * (0.5 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_progress(prog_pkl, quiet=False, epoch_index=-1):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """

    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[epoch_index]

    _print("\nPrevious Progress:")
    msg = "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    print("\nResume training from:")
    print("  epoch = %s" % epoch)
    print("  global_step = %s" % global_step)
    print("  best_epoch = %s" % best_epoch)
    print("  best_acc = %.4f" % best_avg_r10)
    return prog, epoch, global_step, best_epoch, best_avg_r10
