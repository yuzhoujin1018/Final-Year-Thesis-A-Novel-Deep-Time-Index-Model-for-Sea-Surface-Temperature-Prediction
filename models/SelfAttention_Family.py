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


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, queries, keys, sample_k, n_top):
        """
        Probabilistic QK computation for attention mechanism.

        Args:
            queries: Tensor of shape [batch_size, L_Q, d_model].
            keys: Tensor of shape [batch_size, L_K, d_model].
            sample_k: Number of sampled keys (float, will be converted to int).
            n_top: Number of selected top queries.

        Returns:
            scores_top: Top scores after sampling.
            index: Indices of the sampled keys.
        """
        # Get dimensions
        L_Q, L_K = queries.size(1), keys.size(1)

        # Ensure sample_k is an integer and at least 1
        sample_k = max(1, int(sample_k))

        # Randomly sample indices for keys
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q

        # Compute sampled keys based on the indices
        keys_sample = keys.gather(1, index_sample.unsqueeze(-1).expand(-1, -1, keys.size(-1)))

        # Compute attention scores between queries and sampled keys
        scores = torch.matmul(queries, keys_sample.transpose(-2, -1))

        # Select top-n scores for each query
        scores_top, index = torch.topk(scores, n_top, dim=-1)

        return scores_top, index

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


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