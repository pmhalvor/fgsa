from torch import nn
import torch 

from utils import score


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = torch.flatten(input)
    target = torch.flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    union = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / union.clamp(min=epsilon))


def f1_loss(target:torch.Tensor, input:torch.Tensor, is_training=True) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    y_pred = input.argmax(dim=1)
    y_true = target
    
    raise NotImplementedError   # BUG: Does not consider ignore_id

    tp = (y_true * y_pred).sum(-1).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = (2* (precision*recall) / (precision + recall + epsilon)).sum()
    f1.requires_grad = is_training
    return f1


class _AbstractDiceLoss(torch.nn.Module):
    """
    Base class for different implementations of Dice loss.

    Found at: https://github.com/wolny/pytorch-3dunet/blob/eafaa5f830eebfb6dbc4e570d1a4c6b6e25f2a1e/pytorch3dunet/unet3d/losses.py
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # NOTE: change by pmhalvor: added argmax to get same shape
        input = torch.argmax(input, dim=1).float()

        # BUG: No need to normalize since multiclass..?
        # get probabilities from logits
        # input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        average = 1. - torch.mean(per_channel_dice)
        average.requires_grad = True
        return average


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid', ignore_id=-1):
        super().__init__(weight, normalization)
        self.ignore_id = ignore_id

    def dice(self, input, target, weight):
        input[target == self.ignore_id] = 0
        target[target == self.ignore_id] = 0 

        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class F1_Loss(torch.nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.is_training=True
        
    def forward(self, input, target):
        return f1_score(input, target, is_training=self.is_training)

    def train():
        self.is_training = True
        
    def eval():
        self.is_training = False


class RaclF1Loss(torch.nn.Module):
    """ 
    In order to build this as close to f1_score used in RACL, you need to 'warm up' the loss.

    NOTE 
    Basically, feed with polarity beforehand, meaning needs custom backward...

    Skipping for now.
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.train_op = False
        
    def forward(self, input, target):
        raise NotImplementedError  # needs custom backward

        true_sentiment = self.batch_polarity
        predict_sentiment = self.output_polarity
        train_op = self.train_op

        return score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, train_op)

    def set_train_op(b=True):
        self.train_op = b


class MIULoss(torch.nn.Module):    
    """ 
    Multiclass Intersection over Union Loss for batched 1D tensors w/ multiple classes per label
    """
    def __init__(self, smooth=1e-7, ignore_id=-1, label_dim=1):
        super().__init__()
        self.smooth = smooth
        self.sigmoid = torch.nn.Sigmoid()
        self.ignore_id = ignore_id
        self.label_dim = label_dim

    def forward(self, input, target):
        """
        Expects input as logits from multiclass model w/ labels at self.label_dim (default 1)
        """
        return self.iou(prediction=input, target=target, ignore_id=self.ignore_id, label_dim=self.label_dim)

    @staticmethod
    def iou(prediction, target, ignore_id=-1, label_dim=1):
        # ignore indexes
        best_guess = prediction.max(dim=label_dim)  # reduce to single best label estimate
        argmax_pred = best_guess.indices  # only need label index for comparisions
        argmax_pred[target == ignore_id] = 0  # remove ignore ids
        target[target == ignore_id] = 0  # remove ignore ids

        # bool tensor for intersecting prediction-target
        overlap = target == argmax_pred

        # get estimate for prediction certainty 
        certainty = self.sigmoid(best_guess.values)

        # intersecting correct labels w/ estimated certainty
        intersect = (certainty * (overlap).float()).flatten().sum(-1)  # sum instead of area for 1D
        
        # count all labels expected
        expected = (target > 0).float().flatten().sum(-1)  # sum instead of area for 1D

        # count all labels predicted
        predicted = torch.flatten(certainty).sum(-1)  # sum instead of area for 1D

        # area of two spaces, minus the intersecting parts (counted double)
        union = expected + predicted - intersect
        
        # get intersection over union w/ smoothing
        iou = (intersection + self.smooth)/(union + self.smooth)

        return 1. - iou  # difference from 1 since using as loss metric

    