# models/DeepTime.py
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat, reduce

from models.modules.inr import INR
from models.modules.regressors import RidgeRegressor
from models.modules.SelfAttention_Family import FullAttention, AttentionLayer  # 新增注意力模块
from utils.masking import TriangularCausalMask  # 新增因果掩码
import gin


@gin.configurable()
def deeptime(datetime_feats: int, layer_size: int, inr_layers: int,
             n_fourier_feats: int, scales: float, n_heads: int = 4,
             attention_dropout: float = 0.1):  # 新增attention_dropout参数
    return DeepTIMe(datetime_feats, layer_size, inr_layers, n_fourier_feats, scales, n_heads, attention_dropout)


class DeepTIMe(nn.Module):
    def __init__(self, datetime_feats: int, layer_size: int, inr_layers: int,
                 n_fourier_feats: int, scales: float, n_heads: int = 4,
                 attention_dropout: float = 0.1):  # 新增attention_dropout参数
        super().__init__()
        # INR模块（保持原有结构）
        self.inr = INR(
            in_feats=datetime_feats + 1,
            layers=inr_layers,
            layer_size=layer_size,
            n_fourier_feats=n_fourier_feats,
            scales=scales
        )

        # 新增注意力层
        self.attention = AttentionLayer(
            FullAttention(
                mask_flag=True,
                output_attention=False,
                d_model=layer_size,
                n_heads=n_heads,
                d_ff=layer_size * 4,  # 前馈层维度扩展
                attention_dropout=attention_dropout  # 传递attention_dropout参数
            ),
            d_model=layer_size,
            n_heads=n_heads
        )

        # 岭回归模块（保持原有结构）
        self.adaptive_weights = RidgeRegressor()

        # 记录参数（新增n_heads和attention_dropout）
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.n_heads = n_heads  # 新增
        self.attention_dropout = attention_dropout  # 新增

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        # 获取时间坐标（保持原有逻辑）
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)

        # 时间特征生成（保持原有逻辑）
        if y_time.shape[-1] != 0:
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)
            time_reprs = self.inr(coords)
        else:
            time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)

        # 新增注意力处理（维度调整 -> 注意力 -> 维度恢复）
        time_reprs = rearrange(time_reprs, 'b t d -> b t d')  # 确保形状为 [batch, time, features]
        attn_output, _ = self.attention(
            queries=time_reprs,
            keys=time_reprs,
            values=time_reprs,
            attn_mask=TriangularCausalMask(batch_size, time_reprs.size(1))  # 因果掩码
        )
        time_reprs = attn_output  # 输出维度保持不变 [batch, time, layer_size]

        # 后续逻辑保持不变
        lookback_reprs = time_reprs[:, :-tgt_horizon_len]
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        w, b = self.adaptive_weights(lookback_reprs, x)
        preds = self.forecast(horizon_reprs, w, b)
        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')