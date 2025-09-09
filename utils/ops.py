





from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from einops import reduce
def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_tensor(array: np.ndarray, to_default_device: bool = True) -> Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    return tensor.to(default_device()) if to_default_device else tensor
def _handle_division_errors(result: np.ndarray) -> np.ndarray:
    result[result != result] = 0.0  # NaN处理
    result[result == np.inf] = 0.0  # Inf处理
    return result
def divide_no_nan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = b == 0.0
    b = np.where(mask, 1.0, b)  # 避免除零
    result = a / b
    result[mask] = 0.0  # 除零结果设为0
    return _handle_division_errors(result)
def _get_scaling_factor(x: Tensor) -> Tensor:
    scaling_factor = reduce(torch.abs(x).data, 'b t d -> b 1 d', 'mean')
    return torch.where(scaling_factor == 0.0, 1.0, scaling_factor)
def scale(x: Tensor, scaling_factor: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if scaling_factor is not None:
        return x / scaling_factor, scaling_factor
    scaling_factor = _get_scaling_factor(x)
    return x / scaling_factor, scaling_factor
def descale(forecast: Tensor, scaling_factor: Tensor) -> Tensor:
    return forecast * scaling_factor