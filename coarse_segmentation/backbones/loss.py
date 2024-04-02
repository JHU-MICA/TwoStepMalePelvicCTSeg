# implements loss functions

import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


from monai.networks import one_hot
from monai.utils import LossReduction

class NR_DiceLoss(_Loss):
    """
    Compute average Noise RObust Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    (reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9109297)
    
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    """

    def __init__(
        self,
        gamma : float = 1.5,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 0,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        """
        Args:
            gamma: exponent of numerator
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.gamma = gamma
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """

        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        numerator = torch.sum(torch.pow(torch.abs(target - input),self.gamma), dim=reduce_axis)

        ground_o = torch.sum(torch.pow(target,2), dim=reduce_axis)
        pred_o = torch.sum(torch.pow(input,2), dim=reduce_axis)

        denominator = ground_o + pred_o


        f: torch.Tensor = (numerator + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f