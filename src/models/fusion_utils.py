"""
注意力机制和融合工具模块

从原始CMTA项目中提取的核心组件，适配医学影像特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class NystromAttention(nn.Module):
    """
    Nystrom近似注意力机制，用于高效处理长序列
    """
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.residual = residual
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, N, dim] 输入特征序列
        Returns:
            output: [B, N, dim] 注意力输出
        """
        B, N, _ = x.shape

        # Layer normalization
        x_norm = self.norm(x)

        # Linear projections
        q = self.to_q(x_norm)  # [B, N, heads*dim_head]
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        # Reshape for multi-head attention
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # [B, heads, N, dim_head]
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)

        # 使用Nystrom近似处理长序列
        if N > self.num_landmarks:
            # Sample landmarks
            indices = torch.randperm(N, device=x.device)[:self.num_landmarks]
            landmarks = x[:, indices]

            # Compute attention with landmarks approximation
            q_landmarks = q[:, :, indices, :]
            k_landmarks = k[:, :, indices, :]
            v_landmarks = v[:, :, indices, :]

            # Simplified attention computation
            attn = torch.matmul(q, k_landmarks.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)

            output = torch.matmul(attn, v_landmarks)
        else:
            # Standard attention for shorter sequences
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            output = torch.matmul(attn, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.to_out(output)
        output = self.dropout(output)

        # Residual connection
        if self.residual:
            output = output + x

        return output


class TransLayer(nn.Module):
    """
    Transformer编码层，使用Nystrom注意力
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class MultiheadAttention(nn.Module):
    """
    标准多头注意力机制
    """
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Args:
            query, key, value: [seq_len, batch_size, embed_dim]
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.size(1)

        # Linear projection
        qkv = self.in_proj(torch.cat([query, key, value], dim=0))
        q, k, v = torch.split(qkv, query.size(0), dim=0)

        # Reshape for multi-head attention
        q = q.view(query.size(0), batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(key.size(0), batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(value.size(0), batch_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(-1, batch_size, self.embed_dim)
        output = self.out_proj(output)

        return output.transpose(0, 1), attn_weights


def SNN_Block(dim1: int, dim2: int, dropout: float = 0.5) -> nn.Module:
    """
    SNN激活函数块
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.SELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


def MLP_Block(dim1: int, dim2: int, dropout: float = 0.5) -> nn.Module:
    """
    MLP块，包含LayerNorm和Dropout
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.LayerNorm(dim2),
        nn.Dropout(0.3)
    )


def conv1d_Block(dim1: int, dim2: int, dropout: float = 0.5) -> nn.Module:
    """
    1D卷积块
    """
    return nn.Sequential(
        nn.Conv1d(dim1, dim2, 1),
        nn.InstanceNorm1d(dim2),
        nn.Dropout(0.3)
    )


def initialize_weights(module: nn.Module):
    """
    权重初始化函数
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class PPEG(nn.Module):
    """
    Position Encoding Generator
    """
    def __init__(self, dim: int = 512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer(nn.Module):
    """
    Transformer编码器，用于特征序列处理
    """
    def __init__(self, feature_dim: int = 512):
        super(Transformer, self).__init__()
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        # Pad to square
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([features, features[:, :add_length, :]], dim=1)
        else:
            h = features

        # Add class token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # Transformer layers
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)

        return h[:, 0], h[:, 1:]  # Return cls token and patch tokens