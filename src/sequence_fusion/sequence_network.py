import numpy as np

import torch
import torch.nn as nn

from torch.nn import functional as F
from .util import initialize_weights
from .util import NystromAttention
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention
from .kmeans.kmeans_pytorch import kmeans

from scipy.optimize import linear_sum_assignment


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    
##################### self ################# 

class Interaction_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=64):
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
    

class Specificity_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=64):
        super().__init__()
        self.conv = MLP_Block(dim, dim)
        
    def forward(self, feat):
        feat = self.conv(feat)
        return feat

# class Interaction_Estimator(nn.Module):
#     def __init__(self, feat_len=6, dim=64):
#         super().__init__()
#         self.geno_fc = MLP_Block(dim, dim)
#         self.path_fc = MLP_Block(dim, dim)
#         self.geno_atten = nn.Linear(dim, 1)
#         self.path_atten = nn.Linear(dim, 1)
         
#     def forward(self, gfeat, pfeat):        
#         g_align = self.geno_fc(gfeat)
#         p_align = self.path_fc(pfeat)
#         atten = g_align.unsqueeze(3) * p_align.unsqueeze(2)
#         geno_att = torch.sigmoid(self.geno_atten(atten)).squeeze(-1)
#         path_att = torch.sigmoid(self.path_atten(atten.permute(0, 1, 3, 2))).squeeze(-1)
#         interaction = p_align * path_att + g_align * geno_att
#         return interaction






# 无监督PIB
class PIB(nn.Module):
    def __init__(self, 
                 x_dim,              # 输入特征维度
                 z_dim=256,          # 编码空间维度
                 num_proxies=8,      # 代理数量（自动学习）
                 topk=128,            # 选择的top-k特征
                 noise_std=0.1,      # 高斯噪声标准差
                 sample_num=5):      # 噪声采样次数
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

        # 可学习的代理中心（自动聚类）
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
        输入: 
            x - [B, N, x_dim] 特征组
        输出:
            selected_features - [B, topk, z_dim] 选择的特征
            proxy_weights - [B, num_proxies] 代理权重分布
        """
        B, N, _ = x.shape
        
        # 1. 特征编码
        z = self.encoder(x)  # [B, N, z_dim]
        z = self.add_noise(z)
        
        # 2. 计算特征-代理相似度
        z_norm = F.normalize(z, dim=-1)
        proxies_norm = F.normalize(self.proxies, dim=-1)  # [num_proxies, z_dim]
        sim_matrix = torch.matmul(z_norm, proxies_norm.T)  # [B, N, num_proxies]
        
        # 3. 动态重要性加权
        importance = self.importance_predictor(z)  # [B, N, 1]
        weighted_sim = sim_matrix * importance  # [B, N, num_proxies]
        
        # 4. Top-K特征选择
        _, topk_indices = torch.topk(
            weighted_sim.mean(dim=-1),  # 跨代理平均重要性
            k=min(self.topk, N),
            dim=1
        )
        
        # 5. 收集选择的特征
        selected_features = torch.gather(
            z, 
            1, 
            topk_indices.unsqueeze(-1).expand(-1, -1, self.z_dim)
        )  # [B, topk, z_dim]
        
        # 6. 代理权重分布（可解释性分析）
        proxy_weights = F.softmax(weighted_sim.mean(dim=1), dim=-1)  # [B, num_proxies]
        
        return selected_features, proxy_weights, topk_indices, z

    def get_proxy_centers(self):
        """获取当前代理中心（用于可视化）"""
        return self.proxies.detach().cpu().numpy()















# # 有监督PIB
# class PIB(nn.Module):
#     def __init__(self,
#                  x_dim,              # 输入特征维度
#                  z_dim=1024,          # 编码空间维度
#                  sample_num=10,      # proxy 采样次数
#                  topk=128,           # 注意力选择的 top-k 特征
#                  num_classes=2,      # 二分类任务
#                  seed=1):            # 随机种子
#         super(PIB, self).__init__()

#         # self.beta = beta
#         self.sample_num = sample_num
#         self.topk = topk
#         self.num_classes = num_classes
#         self.seed = seed
#         self.z_dim = z_dim

#         # 编码器：3层 MLP 提取输入特征表示 z
#         self.encoder = nn.Sequential(
#             nn.Linear(x_dim, z_dim * 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(z_dim * 2, z_dim * 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(z_dim * 2, z_dim),
#         )

#         # 解码器：对 z 做分类预测
#         self.decoder_logits = nn.Linear(z_dim, num_classes)

#         # 原型初始化，每个类别一个 proxy，每个 proxy 包含 [mu, sigma]
#         self.proxies = nn.Parameter(torch.empty([num_classes, z_dim * 2]))
#         torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)

#         # 类别映射字典：如 {"0": 0, "1": 1}
#         self.proxies_dict = {str(i): i for i in range(num_classes)}

#     def gaussian_noise(self, samples, K, seed):
#         # 为每个 proxy 采样 K 次，得到带扰动的表示
#         if self.training:
#             return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
#         else:
#             return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K),
#                                 generator=torch.manual_seed(seed)).cuda()

#     def encoder_result(self, x):
#         # 编码器输出
#         return self.encoder(x)

#     def encoder_proxies(self):
#         # 将 proxy 拆分为 mu 和 sigma（注意使用 softplus 确保 sigma 为正）
#         mu_proxy = self.proxies[:, :self.z_dim]
#         sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:])
#         return mu_proxy, sigma_proxy

#     def forward(self, x, phase,y=None,):
#         feature_num = x.shape[1]  # 特征数

#         # Step 1: 编码输入为 z
#         z = self.encoder_result(x)  # [B, L, D]

#         # Step 2: 获取 proxy 的 mu 和 sigma，并采样
#         mu_proxy, sigma_proxy = self.encoder_proxies()
#         eps_proxy = self.gaussian_noise(samples=([self.num_classes, self.sample_num]), K=self.z_dim, seed=self.seed)
#         z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(dim=1) * eps_proxy
#         z_proxy = torch.mean(z_proxy_sample, dim=1)  # [num_classes, D]

#         # Step 3: 计算 z 与 z_proxy 之间的 attention
#         z_norm = F.normalize(z, dim=2)                       # [B, L, D]
#         z_proxy_norm = torch.unsqueeze(F.normalize(z_proxy), dim=0)  # [1, C, D]
#         att = torch.matmul(z_norm, z_proxy_norm.transpose(1, 2))     # [B, L, C]
   

#         if phase == 'train':
#             # 训练阶段：根据 y 获取对应 proxy index
#             proxy_indices = [self.proxies_dict[str(int(y_item))] for y_item in y]
#             proxy_indices = torch.tensor(proxy_indices).long().cuda()

#             # 生成 mask：为每个样本选择它的正确 proxy
#             mask = torch.zeros_like(att, dtype=torch.bool).cuda()
#             mask[torch.arange(att.size(0)), :, proxy_indices] = True

#             att_positive = torch.masked_select(att, mask).view(att.size(0), att.size(1), 1)
#             att_negative = torch.masked_select(att, ~mask).view(att.size(0), att.size(1), -1)

#             if att_positive.shape[1] >= self.topk:
#                 # proxy loss：强化正样本 attention，削弱负样本 attention
#                 att_topk_positive, _ = torch.topk(att_positive.squeeze(2), self.topk, dim=1)
#                 att_topk_negative, _ = torch.topk(att_negative, self.topk, dim=1)
#                 att_positive_mean = torch.mean(att_topk_positive, dim=1)
#                 att_negative_mean = torch.mean(torch.mean(att_topk_negative, dim=1), dim=1)
#                 proxy_loss = -(att_positive_mean - att_negative_mean).mean()
#             else:
#                 # fallback：使用所有 available 的 attention，而不是 topk
#                 att_positive_mean = torch.mean(att_positive.squeeze(2), dim=1)
#                 att_negative_mean = torch.mean(torch.mean(att_negative, dim=1), dim=1)
#                 proxy_loss = -(att_positive_mean - att_negative_mean).mean()

#             # ✅ 明确设置 positive_proxy_idx
#             positive_proxy_idx = proxy_indices.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(1)  # [B, 1, D]
        
#         else:
#             # 推理/验证阶段：只用 attention 获取 proxy
#             att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)   # [B, L * C]
            
   


#             # 推理/验证阶段：只用 attention 获取 proxy
#             att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)   # [B, L * C]

#             if att_unbind_proxy.shape[1] >= self.topk:
#                 _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, self.topk, dim=1)
#             else:
#                 # fallback：使用全部可用 attention 而不是 topk
#                 _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, att_unbind_proxy.shape[1], dim=1)

#             att_topk_proxy_idx = att_topk_proxy_idx % self.num_classes
#             positive_proxy_idx, _ = torch.mode(att_topk_proxy_idx, dim=1)
#             positive_proxy_idx = positive_proxy_idx.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(1)  # [B, 1, D]
#             proxy_loss = None


#         # Step 4: 取出每个样本对应的 proxy mu/sigma
#         mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)
#         sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
#         mu_topk = torch.gather(mu_proxy_repeat, 1, positive_proxy_idx).squeeze(1)
#         sigma_topk = torch.gather(sigma_proxy_repeat, 1, positive_proxy_idx).squeeze(1)


#         # # Step 5: 从注意力中获取与 proxy 最相关的 top-k 特征
#         # att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)  # [B, L * C]
#         # att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
#         # att_topk_idx = att_topk_idx % feature_num  # 映射回 feature 索引
#         # z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(2).repeat(1, 1, self.z_dim))  # [B, topk, D]

#         att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)  # [B, L * C]
#         if att_unbind.shape[1] >= self.topk:
            
#             att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
#             att_topk_idx = att_topk_idx % feature_num  # 映射回 feature 索引
#             z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(2).repeat(1, 1, self.z_dim))  # [B, topk, D]
#         else:

#             att_topk, att_topk_idx = torch.topk(att_unbind, att_unbind.shape[1], dim=1)
#             att_topk_idx = att_topk_idx % feature_num  # 映射回 feature 索引
#             z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(2).repeat(1, 1, self.z_dim))  # [B, topk, D]


#         # att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)  # [B, L * C]
#         # if att_unbind.shape[1] >= self.topk:
#         #     att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
#         # else:
#         #     att_topk, att_topk_idx = torch.topk(att_unbind, att_unbind.shape[1], dim=1)  # 取全部
#         #     # 可选：补零保持 z_topk 的维度固定为 self.topk
#         #     pad_len = self.topk - att_unbind.shape[1]
#         #     att_topk_idx = F.pad(att_topk_idx, (0, pad_len), value=0)  # 用第一个位置0补齐，或其他默认值

#         att_topk_idx = att_topk_idx % feature_num  # 映射回原始特征索引
#         z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(2).repeat(1, 1, self.z_dim))  # [B, topk, D]



#         # Step 6: 用 proxy-sample 集合做分类预测（多个 proxy 平均）
#         decoder_logits_proxy = torch.mean(self.decoder_logits(z_proxy_sample), dim=1)

#         return decoder_logits_proxy, mu_proxy, sigma_proxy, z_topk, mu_topk, sigma_topk, proxy_loss













# # 知识分解模块
# class Knowledge_Decomposition(nn.Module):
#     def __init__(self, feat_len=6, feat_dim=64):
#         super().__init__()
#         self.geno_spec = Specificity_Estimator(feat_len, feat_dim)
#         self.path_spec = Specificity_Estimator(feat_len, feat_dim)
        
#         self.common_encoder = Interaction_Estimator(feat_len, feat_dim)
#         self.synergy_encoder = Interaction_Estimator(feat_len, feat_dim)
        
#     def forward(self, gfeat, pfeat):
#         g_spec = self.geno_spec(gfeat)
#         p_spec = self.path_spec(pfeat)
#         common = self.common_encoder(pfeat, gfeat)
#         synergy = self.synergy_encoder(pfeat, gfeat)
#         return common, synergy, g_spec, p_spec


# 共性和衍生知识
class Knowledge_Decomposition(nn.Module):
    def __init__(self, feat_len=6, feat_dim=64):
        super().__init__()
        # self.geno_spec = Specificity_Estimator(feat_len, feat_dim)
        # self.path_spec = Specificity_Estimator(feat_len, feat_dim)
        
        self.common_encoder = Interaction_Estimator(feat_len, feat_dim)
        self.synergy_encoder = Interaction_Estimator(feat_len, feat_dim)
        
    def forward(self, gfeat, pfeat):
        # g_spec = self.geno_spec(gfeat)
        # p_spec = self.path_spec(pfeat)
        common = self.common_encoder(pfeat, gfeat)
        synergy = self.synergy_encoder(pfeat, gfeat)
        return common, synergy
    

# 匈牙利匹配算法（Hungarian Matching），用于将两组点（centers和priors）进行最优的一对一匹配，并返回匹配后对齐的centers
def Hungarian_Matching(centers, priors):
    cost = torch.cdist(centers, priors, p=1).detach().cpu()
    indices = linear_sum_assignment(cost)[-1]
    one_hot_targets = F.one_hot(torch.tensor(indices), centers.shape[0]).float().cuda()
    align_centers = torch.mm(one_hot_targets.T, centers)
    return align_centers

def SNN_Block(dim1, dim2, dropout=0.5):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.SELU(), nn.AlphaDropout(p=dropout, inplace=False))
    
def MLP_Block(dim1, dim2, dropout=0.5):
    # return nn.Sequential(nn.Linear(dim1, dim2), nn.LayerNorm(dim2), nn.ReLU(), nn.Dropout(0.5))
    return nn.Sequential(nn.Linear(dim1, dim2), nn.LayerNorm(dim2), nn.Dropout(0.3))

def conv1d_Block(dim1, dim2, dropout=0.5):
    # return nn.Sequential(nn.Conv1d(dim1, dim2, 1), nn.InstanceNorm1d(dim2), nn.ReLU(), nn.Dropout(0.5))
    return nn.Sequential(nn.Conv1d(dim1, dim2, 1), nn.InstanceNorm1d(dim2), nn.Dropout(0.3))


    


class PPEG(nn.Module):
    def __init__(self, dim=512):
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
    def __init__(self, feature_dim=512):
        super(Transformer, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6) # 对self.cls_token进行正态分布初始化
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]  #
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]



from utils.options import parse_args

args = parse_args()















class CMTA(nn.Module):
    def __init__(self, n_classes=2, fusion="concat", model_size="small"):
        super(CMTA, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = 1024
        self.num_cluster = 64  # 聚类中心数
        self.bank_length = 16 # 每个类别的bank容量
        self.update_ratio = 0.1  # 记忆库更新率
        self.seed = 1




        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 512], "large": [1024, 1024, 512]},
            "Radiomics": {"small": [2048, 1024], "large": [1024, 512,  512]},
            "genomics" : {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
            "clinical": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        fc.append(nn.Linear(3904, 1024))
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)

        # Radiomics Embedding Network
        hidden = self.size_dict["Radiomics"]['small']
        fc = []
        fc.append(nn.Linear(3904, 2048))
        for idx in range(len(hidden) - 1):

            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.radiomics_fc = nn.Sequential(*fc)



        # Pathomics Embedding Network
        self.path_know_memory = nn.Parameter(torch.randn(1, self.num_cluster, 1024), requires_grad=False).cuda()
        self.update_path = args.update_rat  # 权重更新比例
        self.path_fc = SNN_Block(dim1=1024, dim2=self.feat_dim)
        self.path_conv = conv1d_Block(self.num_cluster, 1)

        # Radiomics Embedding Network
        self.radio_know_memory = nn.Parameter(torch.randn(1, self.num_cluster, 1024), requires_grad=False).cuda()
        self.update_ratio = args.update_rat  # 权重更新比例
        self.radio_fc = SNN_Block(dim1=1024, dim2=self.feat_dim)
        self.radio_conv = conv1d_Block(self.num_cluster, 1)


         # P->R Attention
        self.P_in_R_Att = MultiheadAttention(embed_dim=1024, num_heads=1)
        # R->P Attention
        self.R_in_P_Att = MultiheadAttention(embed_dim=1024, num_heads=1)



        # 2. 知识分解模块
        self.know_decompose = Knowledge_Decomposition(self.num_cluster, self.feat_dim)



        self.path_pib = PIB(x_dim=3904, z_dim=1024, num_proxies=8, topk=128)
        # self.path_pib = PIB(x_dim=3904, z_dim=1024, num_classes=self.n_classes)
        # self.radio_pib = PIB(x_dim=3904, z_dim=1024, num_classes=self.n_classes)



        


        # # Cohort Bank
        # self.patient_bank = []
        # for i in range(self.n_classes):
        #     pbank = nn.Parameter(torch.randn(0, 4, self.feat_dim), requires_grad=False).cuda()
        #     self.patient_bank.append(pbank)
        self.patient_bank = []
        for _ in range(self.n_classes):
            # 每个元素的 shape 是 (0, 4, feat_dim)
            # 表示 bank 中最多存 30 个样本，每个样本含 4 个知识向量
            pbank = torch.empty((0, 4, self.feat_dim)).cuda()
            self.patient_bank.append(pbank)



        self.classifier_1 = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.n_classes)
        )
        
        self.p_transformer = Transformer(self.feat_dim)
        self.c_transformer = Transformer(self.feat_dim)
        
        self.transformer = Transformer(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, self.n_classes)


        self.attentionfusion = WeightedAttentionFusion(self.feat_dim)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.feat_dim, self.feat_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(self.feat_dim // 2, self.n_classes)
        # )


    def forward(self, x_path, x_radio, phase, label,**kwargs):
        out_dict = {}
 



        # # 1. 处理Pathomics模态
        # with torch.no_grad():
        # decoder_logits_proxy, mu_proxy, sigma_proxy, path_z_topk, mu_topk, sigma_topk, proxy_loss = self.path_pib(x_path, phase, label)
        selected_features, proxy_weights, topk_indices, z = self.path_pib(x_path)
        # pathomics_features = self.pathomics_fc(x_path.float())
        # radiomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.p_transformer(selected_features)  # cls token + patch tokens
        

        pathomics_encoder = cls_token_pathomics_encoder.unsqueeze(1)  # [1,1024] -->  [1,1,1024]

        
        # decoder_logits_proxy, mu_proxy, sigma_proxy, radio_z_topk, mu_topk, sigma_topk, proxy_loss = self.radio_pib(x_radio, phase, label)
        radiomics_features = self.radiomics_fc(x_radio.float())
        # radiomics encoder
        cls_token_radiomics_encoder,patch_token_radiomics_encoder = self.c_transformer(radiomics_features)  # cls token + patch tokens

        radiomics_encoder = cls_token_radiomics_encoder.unsqueeze(1)  

            # cross attention
        pathomics_in_radiomics, Att = self.P_in_R_Att(    # 放射学中的病理学特征
        patch_token_pathomics_encoder.transpose(1, 0),
        patch_token_radiomics_encoder.transpose(1, 0),
        patch_token_radiomics_encoder.transpose(1, 0),
    )  
        
        pathomics_in_radiomics = pathomics_in_radiomics.permute(1, 0, 2) 
        # pathomics decoder
        # cls_token_pathomics_decoder, patch_token_pathomics_decoder = self.transformer(
        # pathomics_in_radiomics.transpose(1, 0))  # cls token + patch tokens

        # mean_token_pathomics_decoder = patch_token_pathomics_decoder.mean(dim=1, keepdim=True)

    
    
        # cross attention
        radiomics_in_pathomics, Att = self.R_in_P_Att(    #  病理学中的放射学特征
        patch_token_radiomics_encoder.transpose(1, 0),
        patch_token_pathomics_encoder.transpose(1, 0),
        patch_token_pathomics_encoder.transpose(1, 0),
        )  
        radiomics_in_pathomics = radiomics_in_pathomics.permute(1, 0, 2)  # [16, 1, 512] → [1, 16, 512]
        
       
        

        # radiomics decoder
        # cls_token_radiomics_decoder, patch_token_radiomics_decoder = self.transformer(
        # radiomics_in_pathomics.transpose(1, 0))  

        # mean_token_radiomics_decoder = patch_token_radiomics_decoder.mean(dim=1, keepdim=True)
        
        cluster_ids_x, path_centers = kmeans(X=radiomics_in_pathomics[0], num_clusters=self.num_cluster, cluster_centers=self.path_know_memory.detach(), distance='euclidean', device=torch.device('cuda:0'), tqdm_flag=False, seed=self.seed)
        cluster_ids1_x,  radion_centers = kmeans(X=pathomics_in_radiomics[0], num_clusters=self.num_cluster, cluster_centers=self.radio_know_memory.detach(), distance='euclidean', device=torch.device('cuda:0'), tqdm_flag=False, seed=self.seed)
        path_centers = Hungarian_Matching(path_centers.cuda(), self.path_know_memory[0]).unsqueeze(0)
        radio_centers = Hungarian_Matching(radion_centers.cuda(), self.radio_know_memory[0]).unsqueeze(0)



            
        # print(path_centers.shape)
        if phase == 'train':
            self.path_know_memory = (self.path_know_memory * (1 - self.update_path) + path_centers * self.update_path).detach()
            self.radio_know_memory = (self.radio_know_memory * (1 - self.update_ratio) + radion_centers * self.update_ratio).detach()         
        radio_indiv = self.radio_conv(self.radio_fc(radio_centers))
        path_indiv = self.path_conv(self.path_fc(path_centers))


        


        


        # components = common, synergy, geno_spec, path_spec
        sc_components = self.know_decompose(path_indiv, radio_indiv)
        # indiv_components =  torch.cat([sc_components, mean_token_pathomics_decoder, mean_token_radiomics_decoder], dim=1)
        indiv_components = sc_components + (pathomics_encoder,) + (radiomics_encoder,)

        

        
        indiv_know = torch.cat(indiv_components, dim=1)
        # if phase == 'train':
        #     if self.patient_bank[int(label)].shape[0] < self.bank_length:
        #         self.patient_bank[int(label)] = torch.cat([self.patient_bank[int(label)], indiv_know], dim=0)
        #     else:
        #         self.patient_bank[int(label)] = torch.cat([self.patient_bank[int(label)][1:], indiv_know], dim=0)
        if phase == 'train':
            indiv_know = indiv_know.view(1, 4, -1)  # 加 batch 维度：现在是 (1, 4, feat_dim)

            if self.patient_bank[int(label)].shape[0] < self.bank_length:
                self.patient_bank[int(label)] = torch.cat([self.patient_bank[int(label)], indiv_know], dim=0)
            else:
                self.patient_bank[int(label)] = torch.cat(
                    [self.patient_bank[int(label)][1:], indiv_know], dim=0
            )


        


            
        
        fusion,_ = self.transformer(indiv_know)
        


        # fusion = self.attentionfusion(indiv_know)
        # fuse_hazard = torch.sigmoid(self.classifier(fusion))
        # fuse_S = torch.cumprod(1 - fuse_hazard, dim=1)

        # prediction
        out_dict['decompose'] = [indiv_know, [radio_indiv, path_indiv]]
        out_dict['cohort'] = self.patient_bank
        # print(out_dict['cohort'])
        # if proxy_radio_loss is not None:
        #     proxy_loss = (proxy_path_loss+proxy_radio_loss)/2
        # else:
        #     proxy_loss = 0
        # out_dict['hazards'] = [fuse_hazard]
        # out_dict['S'] = [fuse_S]
        logits = self.classifier(fusion)
        
        # return logits, fusion,  cls_token_pathomics_encoder, cls_token_radiomics_encoder,indiv_components,out_dict,proxy_loss
        return logits, fusion,  cls_token_pathomics_encoder, cls_token_radiomics_encoder,indiv_components,out_dict
    
    #########
    # def load_memory_and_bank(self, memory_path):
    #     """
    #     从文件中加载 path_know_memory、radio_know_memory 和 patient_bank
    #     """
    #     checkpoint = torch.load(memory_path)
    #     self.path_know_memory = checkpoint['path_know_memory'].cuda()
    #     self.radio_know_memory = checkpoint['radio_know_memory'].cuda()
    #     self.patient_bank = [pb.cuda() for pb in checkpoint['patient_bank']]







# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class WeightedAttentionFusion(nn.Module):
    def __init__(self, in_channels=4, feat_dim=256, hidden_dim=128, out_dim=256):
        """
        in_channels: 模态数（如 4 个特征）
        feat_dim: 每个模态的特征长度
        hidden_dim: attention计算的中间维度
        out_dim: 融合后的输出维度
        """
        super(WeightedAttentionFusion, self).__init__()
        self.in_channels = 4

        # 对每个模态做特征提取（共享或独立均可）
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(in_channels)
        ])

        # 注意力计算器（可以更复杂）
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, 1)  # 输出 attention 权重分数
        )

        # 融合后再做一次映射（可选）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B, 4, 512]  4 个模态，每个 512 维特征
        return: [B, out_dim]
        """
        B, C, D = x.shape
        assert C == self.in_channels, f"[Error] 输入模态数 {C} 不等于设置的 {self.in_channels}"

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
