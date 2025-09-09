import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,ema_weight=0.3,d_model=512,n_heads=8,dp_rank=8,dropout=0.05,d_ff=2048):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(attention_dropout)
        self.ema_weight = ema_weight
        self.mlp = nn.Linear(d_model // n_heads, dp_rank)
        self.ff_1 = nn.Sequential(nn.Linear(d_model // n_heads, d_ff, bias=True), nn.GELU(), nn.Dropout(
            dropout), nn.Linear(d_ff, d_model // n_heads, bias=True))
        self.ff_2 = nn.Sequential(nn.Linear(d_model // n_heads, d_ff, bias=True), nn.GELU(), nn.Dropout(
            dropout), nn.Linear(d_ff, d_model // n_heads, bias=True))
        self.norm = nn.LayerNorm(d_model // n_heads)
        self.ema_queries = None

    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum("blhe, blhr->blhe", src, src_dp)
        return src_dp

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(24)
        scale1 = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        dp_values = self.dynamic_projection(values, self.mlp)
        dp_keys = self.dynamic_projection(keys, self.mlp)

        ema_keys = None
        ema_queries = None
        if ema_queries is None:
            ema_queries = torch.nn.Parameter(torch.zeros_like(queries).to(queries.device))
        if ema_keys is None:
            ema_keys = torch.nn.Parameter(torch.zeros_like(keys).to(keys.device))
        ema_queries = self.ema_weight * ema_queries + (1 - self.ema_weight) * queries
        ema_keys = self.ema_weight * ema_keys + (1 - self.ema_weight) * dp_keys

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        scores = torch.einsum("blhe,bshe->bhls", ema_queries, ema_keys)
        scores1 = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        A1 = self.dropout(torch.softmax(scale1 * scores1, dim=-1))


        V = torch.einsum("bhls,bshd->blhd", A, dp_values)
        V1 = torch.einsum("bhls,bshd->blhd", A1, values)

        # src2 = V + V1
        # src = self.ff_1(V) + self.ff_2(V1)
        # src = src + src2
        # src = self.norm(src)

        src1 = self.ff_1(V)
        src2 = self.ff_2(V1)
        src = src1 + src2
        src = self.norm(src)

        if self.output_attention:
            return (src.contiguous(), A)
        else:
            return (src.contiguous(), None)



class FullAttentionema(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,ema_weight=0.3,d_model=512,n_heads=8,dp_rank=8,dropout=0.05,d_ff=2048):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(attention_dropout)
        self.ema_weight = ema_weight
        self.mlp = nn.Linear(d_model // n_heads, dp_rank)
        self.ff_1 = nn.Sequential(nn.Linear(d_model // n_heads, d_ff, bias=True), nn.GELU(), nn.Dropout(
            dropout), nn.Linear(d_ff, d_model // n_heads, bias=True))
        self.ff_2 = nn.Sequential(nn.Linear(d_model // n_heads, d_ff, bias=True), nn.GELU(), nn.Dropout(
            dropout), nn.Linear(d_ff, d_model // n_heads, bias=True))
        self.norm = nn.LayerNorm(d_model // n_heads)
        self.ema_queries = None

    # def dynamic_projection(self, src, mlp):
    #     src_dp = mlp(src)
    #     src_dp = F.softmax(src_dp, dim=-1)
    #     src_dp = torch.einsum("blhe, blhr->blhe", src, src_dp)
    #     return src_dp

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scale1 = self.scale or 1. / sqrt(48)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)



        ema_keys = None
        ema_queries = None
        if ema_queries is None:
            ema_queries = torch.nn.Parameter(torch.zeros_like(queries).to(queries.device))
        if ema_keys is None:
            ema_keys = torch.nn.Parameter(torch.zeros_like(keys).to(keys.device))
        ema_queries = self.ema_weight * ema_queries + (1 - self.ema_weight) * queries
        ema_keys = self.ema_weight * ema_keys + (1 - self.ema_weight) * keys

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        scores = torch.einsum("blhe,bshe->bhls", ema_queries, ema_keys)
        scores1 = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        A1 = self.dropout(torch.softmax(scale1 * scores1, dim=-1))


        V = torch.einsum("bhls,bshd->blhd", A, values)
        V1 = torch.einsum("bhls,bshd->blhd", A1, values)

        # src2 = V + V1
        src = self.ff_1(V) + self.ff_2(V1)
        # src = src + src2
        src = self.norm(src)

        if self.output_attention:
            return (src.contiguous(), A)
        else:
            return (src.contiguous(), None)





class AttentionLayer(nn.Module):
    def __init__(self,attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn