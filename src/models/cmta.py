"""
CMTA (Cross-Modality Transformer-based Fusion) 融合模型实现

基于transformer和知识分解的多模态医学影像融合方法，
支持病理学和放射学特征的有效融合。

参考文献：Cross-Modality Transformer-based Fusion for Medical Image Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from .cmta_utils import (
    initialize_weights, NystromAttention, BilinearFusion,
    PIB, Knowledge_Decomposition, Hungarian_Matching, MLP_Block,
    SNN_Block, conv1d_Block, Transformer
)
from ..utils.kmeans import kmeans


class Interaction_Estimator(nn.Module):
    """交互估计器 - 模态间交互建模"""

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


class WeightedAttentionFusion(nn.Module):
    """加权注意力融合模块"""

    def __init__(self, in_channels: int = 4, feat_dim: int = 256,
                 hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels

        # 对每个模态做特征提取
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(in_channels)
        ])

        # 注意力计算器
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

        # 融合后再做一次映射
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B, 4, 512] 4个模态，每个512维特征
        return: [B, out_dim]
        """
        B, C, D = x.shape
        assert C == self.in_channels, f"输入模态数 {C} 不等于设置的 {self.in_channels}"

        feature_list = []
        attention_scores = []

        for i in range(C):
            xi = x[:, i, :]  # [B, D]
            fi = self.extractors[i](xi)  # [B, hidden_dim]
            ai = self.attention_net(fi)  # [B, 1]
            feature_list.append(fi)
            attention_scores.append(ai)

        features = torch.stack(feature_list, dim=1)       # [B, 4, hidden_dim]
        attention_raw = torch.cat(attention_scores, dim=1)  # [B, 4]
        attention_weights = F.softmax(attention_raw, dim=1) # [B, 4]

        # 加权融合
        attention_weights = attention_weights.unsqueeze(-1)        # [B, 4, 1]
        fused = torch.sum(features * attention_weights, dim=1)     # [B, hidden_dim]
        out = self.output_layer(fused)                              # [B, out_dim]
        return out


class CMTA(nn.Module):
    """
    CMTA: Cross-Modality Transformer-based Fusion

    核心功能：
    1. PIB (Privacy Information Bottleneck) 特征选择
    2. 知识分解 (Common + Synergy)
    3. 跨模态注意力机制
    4. 队列记忆库学习
    5. 多尺度特征融合
    """

    def __init__(self,
                 n_classes: int = 2,
                 fusion: str = "concat",
                 model_size: str = "small",
                 feat_dim: int = 1024,
                 num_cluster: int = 64,
                 bank_length: int = 16,
                 update_ratio: float = 0.1):
        super(CMTA, self).__init__()

        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.num_cluster = num_cluster
        self.bank_length = bank_length
        self.update_ratio = update_ratio
        self.seed = 1

        # 模型尺寸配置
        self.size_dict = {
            "pathomics": {"small": [1024, 512], "large": [1024, 1024, 512]},
            "radiomics": {"small": [2048, 1024], "large": [1024, 512, 512]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
            "clinical": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }

        # Pathomics嵌入网络
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        fc.append(nn.Linear(3904, 1024))
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        # Radiomics嵌入网络
        hidden = self.size_dict["radiomics"]['small']
        fc = []
        fc.append(nn.Linear(3904, 2048))
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.radiomics_fc = nn.Sequential(*fc)

        # 知识记忆库
        self.path_know_memory = nn.Parameter(
            torch.randn(1, self.num_cluster, 1024), requires_grad=False
        )
        self.path_fc = SNN_Block(dim1=1024, dim2=self.feat_dim)
        self.path_conv = conv1d_Block(self.num_cluster, 1)

        self.radio_know_memory = nn.Parameter(
            torch.randn(1, self.num_cluster, 1024), requires_grad=False
        )
        self.radio_fc = SNN_Block(dim1=1024, dim2=self.feat_dim)
        self.radio_conv = conv1d_Block(self.num_cluster, 1)

        # 跨模态注意力
        self.P_in_R_Att = NystromAttention(dim=1024, heads=1)
        self.R_in_P_Att = NystromAttention(dim=1024, heads=1)

        # 知识分解模块
        self.know_decompose = Knowledge_Decomposition(self.num_cluster, self.feat_dim)

        # PIB特征选择
        self.path_pib = PIB(x_dim=3904, z_dim=1024, num_proxies=8, topk=128)

        # 患者队列库
        self.patient_bank = []
        for _ in range(self.n_classes):
            pbank = torch.empty((0, 4, self.feat_dim))
            self.patient_bank.append(pbank)

        # 分类器
        self.classifier = nn.Linear(self.feat_dim, self.n_classes)

        # Transformer组件
        self.p_transformer = Transformer(self.feat_dim)
        self.c_transformer = Transformer(self.feat_dim)
        self.transformer = Transformer(self.feat_dim)

        # 注意力融合
        self.attentionfusion = WeightedAttentionFusion(self.feat_dim)

        # 初始化权重
        self.apply(initialize_weights)

    def forward(self,
                x_path: torch.Tensor,
                x_radio: torch.Tensor,
                phase: str = 'train',
                label: Optional[torch.Tensor] = None) -> Dict:
        """
        前向传播

        Args:
            x_path: [B, N, D] 病理学特征
            x_radio: [B, N, D] 放射学特征
            phase: 'train' 或 'test'
            label: [B] 样本标签

        Returns:
            Dict包含预测结果和中间特征
        """
        out_dict = {}

        # 1. PIB特征选择（病理学模态）
        selected_features, proxy_weights, topk_indices, z = self.path_pib(x_path)
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.p_transformer(selected_features)
        pathomics_encoder = cls_token_pathomics_encoder.unsqueeze(1)

        # 2. 放射学编码
        radiomics_features = self.radiomics_fc(x_radio.float())
        cls_token_radiomics_encoder, patch_token_radiomics_encoder = self.c_transformer(radiomics_features)
        radiomics_encoder = cls_token_radiomics_encoder.unsqueeze(1)

        # 3. 跨模态注意力
        # 病理学->放射学注意力
        pathomics_in_radiomics, _ = self.P_in_R_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_radiomics_encoder.transpose(1, 0),
            patch_token_radiomics_encoder.transpose(1, 0),
        )
        pathomics_in_radiomics = pathomics_in_radiomics.permute(1, 0, 2)

        # 放射学->病理学注意力
        radiomics_in_pathomics, _ = self.R_in_P_Att(
            patch_token_radiomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )
        radiomics_in_pathomics = radiomics_in_pathomics.permute(1, 0, 2)

        # 4. K-means聚类和知识记忆更新
        cluster_ids_x, path_centers = kmeans(
            X=radiomics_in_pathomics[0],
            num_clusters=self.num_cluster,
            cluster_centers=self.path_know_memory.detach(),
            distance='euclidean',
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            tqdm_flag=False,
            seed=self.seed
        )

        cluster_ids1_x, radion_centers = kmeans(
            X=pathomics_in_radiomics[0],
            num_clusters=self.num_cluster,
            cluster_centers=self.radio_know_memory.detach(),
            distance='euclidean',
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            tqdm_flag=False,
            seed=self.seed
        )

        path_centers = Hungarian_Matching(path_centers, self.path_know_memory[0]).unsqueeze(0)
        radio_centers = Hungarian_Matching(radion_centers, self.radio_know_memory[0]).unsqueeze(0)

        # 5. 训练阶段更新知识记忆
        if phase == 'train':
            self.path_know_memory = (
                self.path_know_memory * (1 - self.update_ratio) + path_centers * self.update_ratio
            ).detach()
            self.radio_know_memory = (
                self.radio_know_memory * (1 - self.update_ratio) + radio_centers * self.update_ratio
            ).detach()

        # 6. 个体知识提取
        radio_indiv = self.radio_conv(self.radio_fc(radio_centers))
        path_indiv = self.path_conv(self.path_fc(path_centers))

        # 7. 知识分解
        sc_components = self.know_decompose(path_indiv, radio_indiv)
        indiv_components = sc_components + (pathomics_encoder,) + (radiomics_encoder,)
        indiv_know = torch.cat(indiv_components, dim=1)

        # 8. 队列库更新
        if phase == 'train' and label is not None:
            indiv_know = indiv_know.view(1, 4, -1)
            if self.patient_bank[int(label)].shape[0] < self.bank_length:
                self.patient_bank[int(label)] = torch.cat(
                    [self.patient_bank[int(label)], indiv_know], dim=0
                )
            else:
                self.patient_bank[int(label)] = torch.cat(
                    [self.patient_bank[int(label)][1:], indiv_know], dim=0
                )

        # 9. 最终融合和分类
        fusion, _ = self.transformer(indiv_know)
        logits = self.classifier(fusion)

        # 输出字典
        out_dict['decompose'] = [indiv_know, [radio_indiv, path_indiv]]
        out_dict['cohort'] = self.patient_bank
        out_dict['logits'] = logits
        out_dict['fusion'] = fusion
        out_dict['path_encoder'] = cls_token_pathomics_encoder
        out_dict['radio_encoder'] = cls_token_radiomics_encoder

        return out_dict