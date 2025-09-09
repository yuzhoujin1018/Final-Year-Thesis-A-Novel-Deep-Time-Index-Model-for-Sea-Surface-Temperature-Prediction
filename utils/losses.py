from typing import Optional, Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor


def rmse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Computes the Root Mean Square Error (RMSE) between input and target.

    Args:
        input: predicted values
        target: ground truth values
        reduction: 'mean' (default) or 'sum' or 'none'

    Returns:
        RMSE loss
    """
    return torch.sqrt(F.mse_loss(input, target, reduction=reduction))


def get_loss_fn(loss_name: str,
                delta: Optional[float] = 1.0,
                beta: Optional[float] = 1.0) -> Callable:
    return {
        'mse': F.mse_loss,
        'mae': F.l1_loss,
        'huber': partial(F.huber_loss, delta=delta),
        'smooth_l1': partial(F.smooth_l1_loss, beta=beta),
        'rmse': rmse_loss  # 添加RMSE损失函数
    }[loss_name]