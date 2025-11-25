import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F





def define_loss(args):
    if args.loss == "ce_surv":
        loss = CrossEntropySurvLoss(alpha=0.0)
    elif args.loss == "nll_surv":
        loss = NLLSurvLoss(alpha=0.0)
    elif args.loss == "cox_surv":
        loss = CoxSurvLoss()
    elif args.loss == "nll_surv_kl":
        print('########### ', "nll_surv_kl")
        loss = [NLLSurvLoss(alpha=0.0), KLLoss()]
    elif args.loss == "nll_surv_mse":
        print('########### ', "nll_surv_mse")
        loss = [NLLSurvLoss(alpha=0.0), nn.MSELoss()]
    elif args.loss == "nll_surv_l1":
        # print('########### ', "nll_surv_l1")
        # loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
        print('########### ', "nll_cross_l1")
        loss = [nn.CrossEntropyLoss(), nn.L1Loss()]

    elif args.loss == "nll_surv_focalloss":
        # print('########### ', "nll_surv_l1")
        # loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
        print('########### ', "focalloss")
        loss = [FocalLoss(), nn.L1Loss()]

    elif args.loss == "nll_surv_focalloss1":
        # print('########### ', "nll_surv_l1")
        # loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
        print('########### ', "focalloss")
        loss = [FocalLoss(),  CosineLoss()]

    elif args.loss == "nll_surv_l2":
        # print('########### ', "nll_surv_l1")
        # loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
        print('########### ', "nll_cross_l1")
        # loss = [nn.CrossEntropyLoss(), SinkhornDistance(),ContrastiveLoss(),ContrastiveLoss(),ContrastiveLoss(),ContrastiveLoss()] KLLoss
        loss = [nn.CrossEntropyLoss(), SinkhornDistance(),ContrastiveLoss(),ContrastiveLoss(),ContrastiveLoss(),ContrastiveLoss()] 

    elif args.loss == "nll_surv_cos":
        print('########### ', "nll_surv_cos")
        loss = [NLLSurvLoss(alpha=0.0), CosineLoss()]
    elif args.loss == "nll_surv_ol":
        print('########### ', "nll_surv_ol")
        loss = [NLLSurvLoss(alpha=0.0), OrthogonalLoss(gamma=0.5)]
    elif args.loss == "CohortLoss":
        print('########### ', "nll_surv_ol")
        loss = [nn.CrossEntropyLoss(), CohortLoss()]
    else:
        raise NotImplementedError
    return loss




class CohortLoss():
    def __call__(self, out, label, temperature=2):
        loss = 0
        # pos_feat = []
        # neg_feat = []
        if 'cohort' in out.keys():
            indiv, origs = out['decompose']
            cohort = out['cohort']
            
            mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]]).cuda()
            indiv_know = indiv.view(4, 1, -1)  # (4, 1, feature_dim)
            orig = torch.cat(origs, dim=1).detach()  # (batch_size, feature_dim)
            sim = F.cosine_similarity(indiv_know, orig, dim=-1)
            intra_loss = torch.mean(torch.abs(sim) * (1 - mask) - mask * sim) + 1
            
            pos_feat = []
            neg_feat = []
        
            for feat in cohort:
                if label == 1:
                    pos_feat.append(feat.detach())
                else:
                    neg_feat.append(feat.detach())

            if len(pos_feat) < 1 or len(neg_feat) < 1:
                inter_loss = 0
            else:
                pos_feat = torch.cat(pos_feat, dim=0)
                neg_feat = torch.cat(neg_feat, dim=0)
                
                neg_dis = torch.matmul(indiv_know.squeeze(1), neg_feat.T) / temperature
                pos_dis = torch.matmul(indiv_know.squeeze(1), pos_feat.T) / temperature

                # Contrastive-like loss
                inter_loss = -torch.log(
                    torch.exp(pos_dis).mean() / (torch.exp(pos_dis).mean() + torch.exp(neg_dis).mean() + 1e-10)
                )
            
            loss = intra_loss.mean() + inter_loss
        return loss










class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=1000, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        x = F.softmax(x, dim=-1)  # 对最后一维进行 softmax
        y = F.softmax(y, dim=-1)
        # x = x / x.sum(dim=-1, keepdim=True)
        # y = y / y.sum(dim=-1, keepdim=True)

        # 扩充维度，如果 x 和 y 是二维的，扩展到三维
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 扩展 batch 维度
        if y.dim() == 2:
            y = y.unsqueeze(0)

        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        C = C.cuda()
        n_points = x.shape[-2]
        batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        # mu = torch.empty(batch_size, n_points, dtype=torch.float,
        #                  requires_grad=False).fill_(1.0 / n_points).squeeze()
        # nu = torch.empty(batch_size, n_points, dtype=torch.float,
        #                  requires_grad=False).fill_(1.0 / n_points).squeeze()
        mu = torch.full((batch_size, n_points), 1.0 / n_points, dtype=torch.float, device=x.device)
        nu = torch.full((batch_size, n_points), 1.0 / n_points, dtype=torch.float, device=x.device)

        u = torch.zeros_like(mu)
        u = u.cuda()
        v = torch.zeros_like(nu)
        v = v.cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8).cuda() - self.lse(self.M(C, u, v))).cuda() + u
            v = self.eps * (torch.log(nu + 1e-8).cuda() - self.lse(self.M(C, u, v).transpose(-2, -1))).cuda() + v
            err = (u - u1).abs().sum(-1).mean()
            err = err.cuda()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        MM = self.M(C, U, V)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost
    
    
    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        # u = u.unsqueeze(-1)
        # v = v.unsqueeze(-2)
        # m = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        result = torch.log(torch.exp(A).sum(-1) + 1e-6)
        return result

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1




class ContrastiveLoss(object):
    def __init__(self, epsilon=1e-8, max_queue_length=10):
        self.epsilon = epsilon
        self.max_queue_length = max_queue_length
        self.cohort_bank = []
        self.labels = []
        self.d = SinkhornDistance()
    def initialize_cohort(self, initial_features, initial_labels):
        """
        初始化 cohort_bank 和 labels。
        :param initial_features: 训练集特征向量 (N, embedding_dim) 的 numpy 数组
        :param initial_labels: 训练集特征标签 (N,) 的 numpy 数组
        """
        self.cohort_bank = initial_features.tolist()
        self.labels = initial_labels.tolist()

    def __call__(self, S, current_label,dataset):
        """
        计算对比损失 l_p。
        :param S: 当前患者的特征向量 (embedding_dim,)
        :param current_label: 当前患者的标签
        :return: 对比损失 l_p
        """
        if self.cohort_bank is None or self.labels is None:
            raise ValueError("Cohort bank is not initialized. Call 'initialize_cohort' first.")
        
        # 正样本 (S+): 与当前患者同类的样本
        S_plus_indices = [i for i, label in enumerate(self.labels) if label == current_label]
        
        # 负样本 (S-): 与当前患者不同类的样本
        S_minus_indices = [i for i, label in enumerate(self.labels) if label != current_label]
        
        # 检查是否有正样本或负样本
        if not S_plus_indices or not S_minus_indices:
            print("No positive or negative samples found. Skipping loss calculation.")
            # 更新cohort
            self.update_cohort_bank(S,current_label,dataset)
            return torch.tensor(0.0, device=S.device)
        # # 计算分子：与正样本的距离和
        # positive_distances = np.sum([F.pairwise_distance(S, self.cohort_bank[idx].to(S.device)) for idx in S_plus_indices])
        
        # # 计算分母：与正样本和负样本的距离和
        # negative_distances = np.sum([F.pairwise_distance(S, self.cohort_bank[idx].to(S.device)) for idx in S_minus_indices])

        positive_distances = torch.sum(torch.stack([F.pairwise_distance(S, self.cohort_bank[idx].to(S.device)) for idx in S_plus_indices]))
        negative_distances = torch.sum(torch.stack([F.pairwise_distance(S, self.cohort_bank[idx].to(S.device)) for idx in S_minus_indices]))

        # positive_distances = torch.sum(torch.stack([self.d(S, self.cohort_bank[idx].to(S.device)) for idx in S_plus_indices]))
        # negative_distances = torch.sum(torch.stack([self.d(S, self.cohort_bank[idx].to(S.device)) for idx in S_minus_indices]))


        denominator = positive_distances + negative_distances + self.epsilon
        
        # Contrastive loss: l_p
        # lp = -np.log(positive_distances / denominator)
        lp = -torch.log(positive_distances / denominator)
        # 更新cohort
        self.update_cohort_bank(S,current_label,dataset)

        return lp

    def update_cohort_bank(self, new_features, new_label,dataset):
        """
        更新 cohort bank 队列，使用先进先出的机制。
        :param new_features: 新加入的特征向量, (embedding_dim,)
        :param new_label: 新患者的标签
        """
        if dataset == 'test_data':
            return
        
        if len(self.cohort_bank) >= self.max_queue_length:
            # 如果超过最大长度，移除最早的特征和对应的标签
            self.cohort_bank.pop(0)
            self.labels.pop(0)
            # # 生成一个随机索引，范围在0到当前队列长度-1之间
            # random_index = random.randint(0, len(self.cohort_bank) - 1)
            
            # # 根据随机索引移除特征和对应的标签
            # self.cohort_bank.pop(random_index)
            # self.labels.pop(random_index)        
        # 加入新特征和对应标签
        self.cohort_bank.append(new_features.detach())
        self.labels.append(new_label.item())





class FocalLoss(nn.Module):
    def __init__(self, alpha=0.24, gamma=2, num_classes=2, size_average=True):  # alpha=0.24 表示给0类的权重为0.24
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {},".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma




    def forward(self, inputs, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds_softmax = F.softmax(inputs,dim=-1)
        self.alpha = self.alpha.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1,preds_softmax.size(-1))
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss






def nll_loss(hazards, S, Y, c, alpha=0.3, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss















# class CohortLoss():
#     def __init__(self):
#         self.pos_feat_queue = []
#         self.neg_feat_queue = []

#     def __call__(self, out, label, temperature=2):
#         loss = 0
#         if 'cohort' in out.keys():
#             alpha = 10
#             indiv, origs = out['decompose']
#             cohort = out['cohort']  # 假设 cohort 是元组 (特征张量, 标签张量)

#             # -------- intra loss --------
#             mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]]).cuda()
#             indiv_know = indiv.view(4, 1, -1)  # (4, 1, feature_dim)
#             orig = torch.cat(origs, dim=1).detach()  # (4, feature_dim)
#             sim = F.cosine_similarity(indiv_know, orig, dim=-1)
#             intra_loss = torch.mean(torch.abs(sim) * (1 - mask) - mask * sim) + 1

#             # -------- 更新正负队列 --------
#             for feat in cohort[1]:  # cohort[1] 是一个 batch 的特征（B × D）
#                 if label == 1:
#                     self.pos_feat_queue.append(feat.detach())
#                 else:
#                     self.neg_feat_queue.append(feat.detach())

#             # 可选：限制队列长度（避免爆炸）例如保留最新的N个
#             max_queue_len = 100
#             if len(self.pos_feat_queue) > max_queue_len:
#                 self.pos_feat_queue = self.pos_feat_queue[-max_queue_len:]
#             if len(self.neg_feat_queue) > max_queue_len:
#                 self.neg_feat_queue = self.neg_feat_queue[-max_queue_len:]

#             # -------- inter loss --------
#             if len(self.pos_feat_queue) > 0 and len(self.neg_feat_queue) > 0:
#                 pos_feat = torch.stack(self.pos_feat_queue, dim=0)  # (N_pos, feat_dim)
#                 neg_feat = torch.stack(self.neg_feat_queue, dim=0)  # (N_neg, feat_dim)

#                 pos_dis = torch.matmul(indiv_know.squeeze(1), pos_feat.T) / temperature
#                 neg_dis = torch.matmul(indiv_know.squeeze(1), neg_feat.T) / temperature

#                 inter_loss = -torch.log(
#                     torch.exp(pos_dis).mean(dim=1) /
#                     (torch.exp(pos_dis).mean(dim=1) + torch.exp(neg_dis).mean(dim=1) + 1e-10)
#                 ).mean()
#             else:
#                 inter_loss = 0

#             loss = intra_loss.mean() + inter_loss

#         return loss






# class CohortLoss():
#     def __call__(self, out, label, temperature=2):
#         loss = 0
#         # pos_feat = []
#         # neg_feat = []
#         if 'cohort' in out.keys():
#             alpha = 10
#             indiv, origs = out['decompose']
#             cohort = out['cohort']
            
#             mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]]).cuda()
#             indiv_know = indiv.view(4, 1, -1)  # (4, 1, feature_dim)
#             orig = torch.cat(origs, dim=1).detach()  # (batch_size, feature_dim)
#             sim = F.cosine_similarity(indiv_know, orig, dim=-1)
#             intra_loss = torch.mean(torch.abs(sim) * (1 - mask) - mask * sim) + 1
            
#             # if int(label) == 0:
#             #     return intra_loss.mean()
            
            
#             # # 只有 label == 1 的时候计算 inter_loss
#             # # 假设 cohort是 [(feat, label), (feat, label), ...]
#             pos_feat = []
#             neg_feat = []
        
#             for feat in cohort:
#                 if label == 1:
#                     pos_feat.append(feat.detach())
#                 else:
#                     neg_feat.append(feat.detach())

#             if len(pos_feat) < 1 or len(neg_feat) < 1:
#                 inter_loss = 0
#             else:
#                 pos_feat = torch.cat(pos_feat, dim=0)
#                 neg_feat = torch.cat(neg_feat, dim=0)
                
#                 neg_dis = torch.matmul(indiv_know.squeeze(1), neg_feat.T) / temperature
#                 pos_dis = torch.matmul(indiv_know.squeeze(1), pos_feat.T) / temperature

#                 # Contrastive-like loss
#                 inter_loss = -torch.log(
#                     torch.exp(pos_dis).mean() / (torch.exp(pos_dis).mean() + torch.exp(neg_dis).mean() + 1e-10)
#                 )
            
#             loss = intra_loss.mean() + inter_loss
#         return loss



# class CohortLoss():
#     def __call__(self, out, label, temperature=2):
#         loss = 0
#         if 'cohort' in out.keys():
#             alpha = 10
#             indiv, origs = out['decompose']
#             cohort = out['cohort']
            
#             mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]]).cuda()
#             indiv_know = indiv.view(4, 1, -1) # common, synergy, g_spec, p_spec
#             orig = torch.cat(origs, dim=1).detach() # gene, path
#             sim = F.cosine_similarity(indiv_know, orig, dim=-1)
#             intra_loss = torch.mean(torch.abs(sim) * (1 - mask) - mask * sim) + 1
            

#             if int(label) != 0:
#                 neg_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort) if int(gt['label']) > j], dim=0).detach()
#                 pos_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort) if int(gt['label']) <= j], dim=0).detach()
#             else:
#                 return intra_loss.mean()
                    
#             if neg_feat.shape[0] < 1 or pos_feat.shape[0] < 1:
#                 inter_loss = 0
#             else:
#                 neg_dis = indiv_know.squeeze(1) * neg_feat / temperature
#                 pos_dis = indiv_know.squeeze(1) * pos_feat / temperature
#                 inter_loss = -torch.log(torch.exp(pos_dis).mean() / (torch.exp(pos_dis).mean() + torch.exp(neg_dis).mean() + 1e-10)) * 1
            
#             loss = intra_loss.mean() + inter_loss
#         return loss 

# loss_dict = {'nllsurv': nll_loss(), 'cohort': CohortLoss()}




def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    # reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        return loss_cox


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)


class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        pos_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        neg_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )

        loss = pos_pairs + self.gamma * neg_pairs
        return loss
