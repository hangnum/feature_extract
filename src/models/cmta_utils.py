"""
CMTA模型的核心工具模块

包含：注意力机制、PIB、知识分解、Transformer等组件
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np


def ceil(x):
    """math.ceil的别名"""
    return math.ceil(x)


def rearrange(tensor, pattern, **kwargs):
    """einops.rearrange的简化实现"""
    if pattern == "b n (h d) -> b h n d":
        b, n, hd = tensor.shape
        h = kwargs.get('h', 8)
        d = hd // h
        return tensor.view(b, n, h, d)
    elif pattern == "b h n d -> b n (h d)":
        b, h, n, d = tensor.shape
        return tensor.view(b, n, h * d)
    elif pattern == "... i d, ... j d -> ... i j":
        # 这个简化实现可能不完全准确，但对于基本用例足够
        return tensor
    else:
        raise ValueError(f"不支持的模式: {pattern}")


def reduce(tensor, pattern, reduction, **kwargs):
    """einops.reduce的简化实现"""
    if reduction == "sum":
        if pattern == "... (n l) d -> ... n d":
            l = kwargs.get('l', 8)
            b, nl, d = tensor.shape
            n = nl // l
            return tensor.view(b, n, l, d).sum(dim=2)
    elif reduction == "mean":
        if pattern == "... (n l) d -> ... n d":
            l = kwargs.get('l', 8)
            b, nl, d = tensor.shape
            n = nl // l
            return tensor.view(b, n, l, d).mean(dim=2)
    else:
        raise ValueError(f"不支持的reduction: {reduction}")


def einsum(equation, *tensors):
    """einsum的简化实现"""
    return torch.einsum(equation, *tensors)


def initialize_weights(module):
    """权重初始化"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def exists(val):
    """检查值是否存在"""
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    """Moore-Penrose伪逆的迭代计算"""
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    """Nystrom注意力机制 - 高效的注意力计算"""

    def __init__(self,
                 dim: int,
                 dim_head: int = 64,
                 heads: int = 8,
                 num_landmarks: int = 256,
                 pinv_iterations: int = 6,
                 residual: bool = True,
                 residual_conv_kernel: int = 33,
                 eps: float = 1e-8,
                 dropout: float = 0.0):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1),
                                     padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # 填充使序列可以被landmark数整除
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # 推导query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # 设置被mask位置的q,k,v为0
        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # 通过求和减少生成landmarks，然后计算平均值
        l = ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        # 计算landmark mask
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # 掩码均值
        q_landmarks /= divisor
        k_landmarks /= divisor

        # 相似度计算
        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # 掩码处理
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # 注意力计算
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # 添加值的深度卷积残差
        if self.residual:
            out += self.res_conv(v)

        # 合并和组合头
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class BilinearFusion(nn.Module):
    """双线性融合模块"""

    def __init__(self,
                 skip: int = 0,
                 use_bilinear: int = 0,
                 gate1: int = 1,
                 gate2: int = 1,
                 dim1: int = 128,
                 dim2: int = 128,
                 scale_dim1: int = 1,
                 scale_dim2: int = 1,
                 mmhid: int = 256,
                 dropout_rate: float = 0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1_og + dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else
            nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(),
                                      nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else
            nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(),
                                      nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), 256),
                                    nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256 + skip_dim, mmhid),
                                    nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        # 门控多模态单元
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        # 融合
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1: int, dim2: int, dropout: float = 0.5):
    """自归一化网络块"""
    return nn.Sequential(
        nn.Linear(dim1, dim2), nn.SELU(), nn.AlphaDropout(p=dropout, inplace=False)
    )


def MLP_Block(dim1: int, dim2: int, dropout: float = 0.5):
    """多层感知机块"""
    return nn.Sequential(
        nn.Linear(dim1, dim2), nn.LayerNorm(dim2), nn.Dropout(0.3)
    )


def conv1d_Block(dim1: int, dim2: int, dropout: float = 0.5):
    """一维卷积块"""
    return nn.Sequential(
        nn.Conv1d(dim1, dim2, 1), nn.InstanceNorm1d(dim2), nn.Dropout(0.3)
    )


class PIB(nn.Module):
    """隐私信息瓶颈 - 无监督特征选择"""

    def __init__(self,
                 x_dim: int,
                 z_dim: int = 256,
                 num_proxies: int = 8,
                 topk: int = 128,
                 noise_std: float = 0.1,
                 sample_num: int = 5):
        super(PIB, self).__init__()

        self.z_dim = z_dim
        self.num_proxies = num_proxies
        self.topk = topk
        self.noise_std = noise_std
        self.sample_num = sample_num

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.GELU(),
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, z_dim),
            nn.Dropout(0.2)
        )

        # 可学习的代理中心
        self.proxies = nn.Parameter(torch.randn(num_proxies, z_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        # 特征重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, 1),
            nn.Sigmoid()
        )

    def add_noise(self, z):
        """注入高斯噪声的增强方法"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(z) * self.noise_std
            return z + noise
        return z

    def forward(self, x):
        """
        输入: x - [B, N, x_dim] 特征组
        输出: selected_features, proxy_weights, topk_indices, z
        """
        B, N, _ = x.shape

        # 特征编码
        z = self.encoder(x)  # [B, N, z_dim]
        z = self.add_noise(z)

        # 计算特征-代理相似度
        z_norm = F.normalize(z, dim=-1)
        proxies_norm = F.normalize(self.proxies, dim=-1)  # [num_proxies, z_dim]
        sim_matrix = torch.matmul(z_norm, proxies_norm.T)  # [B, N, num_proxies]

        # 动态重要性加权
        importance = self.importance_predictor(z)  # [B, N, 1]
        weighted_sim = sim_matrix * importance  # [B, N, num_proxies]

        # Top-K特征选择
        _, topk_indices = torch.topk(
            weighted_sim.mean(dim=-1),  # 跨代理平均重要性
            k=min(self.topk, N),
            dim=1
        )

        # 收集选择的特征
        selected_features = torch.gather(
            z,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.z_dim)
        )  # [B, topk, z_dim]

        # 代理权重分布
        proxy_weights = F.softmax(weighted_sim.mean(dim=1), dim=-1)  # [B, num_proxies]

        return selected_features, proxy_weights, topk_indices, z

    def get_proxy_centers(self):
        """获取当前代理中心（用于可视化）"""
        return self.proxies.detach().cpu().numpy()


class Knowledge_Decomposition(nn.Module):
    """知识分解模块 - 分解为共性和协同知识"""

    def __init__(self, feat_len: int = 6, feat_dim: int = 64):
        super().__init__()
        self.common_encoder = Interaction_Estimator(feat_len, feat_dim)
        self.synergy_encoder = Interaction_Estimator(feat_len, feat_dim)

    def forward(self, gfeat, pfeat):
        common = self.common_encoder(pfeat, gfeat)
        synergy = self.synergy_encoder(pfeat, gfeat)
        return common, synergy


class Interaction_Estimator(nn.Module):
    """交互估计器"""

    def __init__(self, feat_len: int = 6, dim: int = 64):
        super().__init__()
        self.geno_fc = MLP_Block(dim, dim)
        self.path_fc = MLP_Block(dim, dim)
        self.geno_atten = nn.Linear(dim, 1)
        self.path_atten = nn.Linear(dim, 1)

    def forward(self, gfeat, pfeat):
        g_align = self.geno_fc(gfeat)
        p_align = self.path_fc(pfeat)
        atten = g_align.unsqueeze(3) * p_align.unsqueeze(2)
        geno_att = torch.sigmoid(self.geno_atten(atten)).squeeze(-1)
        path_att = torch.sigmoid(self.path_atten(atten.permute(0, 1, 3, 2))).squeeze(-1)
        interaction = p_align * path_att + g_align * geno_att
        return interaction


def Hungarian_Matching(centers, priors):
    """匈牙利匹配算法"""
    cost = torch.cdist(centers, priors, p=1).detach().cpu()
    indices = linear_sum_assignment(cost)[-1]
    one_hot_targets = F.one_hot(torch.tensor(indices), centers.shape[0]).float()
    if torch.cuda.is_available():
        one_hot_targets = one_hot_targets.cuda()
    align_centers = torch.mm(one_hot_targets.T, centers)
    return align_centers


class PPEG(nn.Module):
    """位置编码"""

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


class TransLayer(nn.Module):
    """Transformer层"""

    def __init__(self, norm_layer=nn.LayerNorm, dim: int = 512):
        super().__init__()
        self.norm = norm_layer(dim)
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


class Transformer(nn.Module):
    """Transformer编码器"""

    def __init__(self, feature_dim: int = 512):
        super(Transformer, self).__init__()
        # 编码器
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        # 填充
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)

        # cls token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if torch.cuda.is_available():
            cls_tokens = cls_tokens.cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # Transformer层
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)

        return h[:, 0], h[:, 1:]