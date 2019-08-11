from __future__ import print_function, division

from include import *
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

class FocalLossWithFC(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, loss_weight=1.0):
        super(FocalLossWithFC, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, 0.25)
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, x, target):
        if self.normalize:
            self.fc.weight.renorm(2, 0, 1e-5).mul(1e5)
        logit = self.fc(x)

        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean().mul_(self.loss_weight)


def binary_cross_entropy(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------
def lovasz_binary(logits, labels, mode='mean', ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    loss = torch.empty(logits.size()[0], 1, device='cuda') if torch.cuda.is_available() else torch.empty(logits.size()[0], 1).cpu()
    for i, (log, lab) in enumerate(zip(logits, labels)):
        loss[i] = lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))

    if mode == 'mean':
        loss = loss.mean()
    elif mode == 'sum':
        loss = loss.sum()
    elif mode == 'individual':
        pass
    else:
        raise ValueError("Invalid loss mode. Please selece one from ['mean', 'sum', 'individual']")

    return loss

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True) # bug
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

# --------------------------- HELPER FUNCTIONS ---------------------------

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# --------------------------- Ring Loss ---------------------------

class SoftmaxLoss(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, loss_weight=1.0):
        super(SoftmaxLoss, self).__init__()
        self.fc = nn.Linear(input_size, int(output_size), bias=False)
        nn.init.kaiming_uniform_(self.fc.weight, 0.25)
        self.weight = self.fc.weight
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, x, y):
        if self.normalize:
            self.fc.weight.renorm(2, 0, 1e-5).mul(1e5)
        prob = F.log_softmax(self.fc(x), dim=1)
        self.prob = prob
        loss = F.nll_loss(prob, y)
        return loss.mul_(self.loss_weight)


class RingLoss(nn.Module):
    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


class AngleSoftmax(nn.Module):
    def __init__(self, input_size, output_size, normalize=True, m=4, lambda_max=1000.0, lambda_min=5.0,
                 power=1.0, gamma=0.1, loss_weight=1.0):
        """
        :param input_size: Input channel size.
        :param output_size: Number of Class.
        :param normalize: Whether do weight normalization.
        :param m: An integer, specifying the margin type, take value of [0,1,2,3,4,5].
        :param lambda_max: Starting value for lambda.
        :param lambda_min: Minimum value for lambda.
        :param power: Decreasing strategy for lambda.
        :param gamma: Decreasing strategy for lambda.
        :param loss_weight: Loss weight for this loss.
        """
        super(AngleSoftmax, self).__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(int(output_size), input_size))
        nn.init.kaiming_uniform_(self.weight, 1.0)
        self.m = m

        self.it = 0
        self.LambdaMin = lambda_min
        self.LambdaMax = lambda_max
        self.gamma = gamma
        self.power = power

    def forward(self, x, y):

        if self.normalize:
            wl = self.weight.pow(2).sum(1).pow(0.5)
            wn = self.weight / wl.view(-1, 1)
            self.weight.data.copy_(wn.data)
        if self.training:
            lamb = max(self.LambdaMin, self.LambdaMax / (1 + self.gamma * self.it)**self.power)
            self.it += 1
            phi_kernel = PhiKernel(self.m, lamb)
            feat = phi_kernel(x, self.weight, y)
            loss = F.nll_loss(F.log_softmax(feat, dim=1), y)
        else:
            feat = x.mm(self.weight.t())
            self.prob = F.log_softmax(feat, dim=1)
            loss = F.nll_loss(self.prob, y)

        return loss.mul_(self.loss_weight)

class PhiKernel(Function):
    def __init__(self, m, lamb):
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
        self.mcoeff_w = [
            lambda x: 0,
            lambda x: 1,
            lambda x: 4 * x,
            lambda x: 12 * x ** 2 - 3,
            lambda x: 32 * x ** 3 - 16 * x,
            lambda x: 80 * x ** 4 - 60 * x ** 2 + 5
        ]
        self.mcoeff_x = [
            lambda x: -1,
            lambda x: 0,
            lambda x: 2 * x ** 2 + 1,
            lambda x: 8 * x ** 3,
            lambda x: 24 * x ** 4 - 8 * x ** 2 - 1,
            lambda x: 64 * x ** 5 - 40 * x ** 3
        ]
        self.m = m
        self.lamb = lamb

    def forward(self, input, weight, label):
        xlen = input.pow(2).sum(1).pow(0.5).view(-1, 1)  # size=B
        wlen = weight.pow(2).sum(1).pow(0.5).view(1, -1)
        cos_theta = (input.mm(weight.t()) / xlen / wlen).clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)

        k = (self.m * cos_theta.acos() / np.pi).floor()
        phi_theta = (-1) ** k * cos_m_theta - 2 * k

        feat = cos_theta * xlen
        phi_theta = phi_theta * xlen

        index = (feat * 0.0).scatter_(1, label.view(-1, 1), 1).byte()
        feat[index] -= feat[index] / (1.0 + self.lamb)
        feat[index] += phi_theta[index] / (1.0 + self.lamb)
        self.save_for_backward(input, weight, label)
        self.cos_theta = (cos_theta * index.float()).sum(dim=1).view(-1, 1)
        self.k = (k * index.float()).sum(dim=1).view(-1, 1)
        self.xlen, self.index = xlen, index
        return feat

    def backward(self, grad_outputs):
        input, weight, label = self.saved_variables
        input, weight, label = input.data, weight.data, label.data
        grad_input = grad_weight = None
        grad_input = grad_outputs.mm(weight) * self.lamb / (1.0 + self.lamb)
        grad_outputs_label = (grad_outputs*self.index.float()).sum(dim=1).view(-1, 1)
        coeff_w = ((-1) ** self.k) * self.mcoeff_w[self.m](self.cos_theta)
        coeff_x = (((-1) ** self.k) * self.mcoeff_x[self.m](self.cos_theta) + 2 * self.k) / self.xlen
        coeff_norm = (coeff_w.pow(2) + coeff_x.pow(2)).pow(0.5)
        grad_input += grad_outputs_label * (coeff_w / coeff_norm) * torch.index_select(weight, 0, label) / (1.0 + self.lamb)
        grad_input -= grad_outputs_label * (coeff_x / coeff_norm) * input / (1.0 + self.lamb)
        grad_input += (grad_outputs * (1 - self.index).float()).mm(weight) / (1.0 + self.lamb)
        grad_weight = grad_outputs.t().mm(input)
        return grad_input, grad_weight, None



# --------------------------- Metric Learning Losses ---------------------------
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.BCEWithLogitsLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.classify_loss(output, labels)
        return loss