# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTLoss(nn.Module):
    """
    【垂直损失 - 有序模式】
    Optimal Transport Loss using 1D Wasserstein Distance (CDF L1).
    适用于有序的图像栈 (Ordered Focal Stack)。
    """
    def __init__(self):
        super(OTLoss, self).__init__()

    def forward(self, probs, soft_gt):
        """
        Args:
            probs: [B, N, H, W] (Softmax后的概率分布)
            soft_gt: [B, N, H, W] (Soft GT 概率分布)
        """
        # 1. 计算累积分布函数 (CDF)
        # 沿着层级维度 (dim=1) 累加
        pred_cdf = torch.cumsum(probs, dim=1)
        gt_cdf = torch.cumsum(soft_gt, dim=1)
        
        # 2. 计算 Wasserstein 距离 (CDF 之间的 L1 距离)
        # 这是 OT Loss 的核心：利用物理距离惩罚错误
        loss = torch.mean(torch.abs(pred_cdf - gt_cdf))
        return loss


class DivergenceLoss(nn.Module):
    """
    【垂直损失 - 无序模式】
    KL Divergence Loss.
    适用于乱序的图像栈 (Unordered/Shuffled Stack) 或消融实验。
    """
    def __init__(self):
        super(DivergenceLoss, self).__init__()

    def forward(self, logits, soft_gt):
        """
        Args:
            logits: [B, N, H, W] (网络输出的 Logits，未经过 Softmax)
            soft_gt: [B, N, H, W] (Soft GT 概率分布)
        """
        # KL 散度要求输入是 Log-Probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # reduction='batchmean' 在数学上更符合 KL 定义
        loss = F.kl_div(log_probs, soft_gt, reduction='batchmean')
        return loss


class SpatialSmoothnessLoss(nn.Module):
    """
    【水平损失 - 空间正则化】
    约束相邻像素的预测分布应保持一致性，防止伪影。
    """
    def __init__(self, mode='ordered'):
        super(SpatialSmoothnessLoss, self).__init__()
        self.mode = mode

    def forward(self, probs):
        """
        Args:
            probs: [B, N, H, W]
        """
        # 根据模式选择计算特征
        if self.mode == 'ordered':
            # 有序模式：约束 CDF 的连续性 (深度连续)
            # 物理含义：相邻像素的深度不应剧烈跳变
            feature = torch.cumsum(probs, dim=1)
        else:
            # 无序模式：约束概率向量的相似性 (分类一致)
            # 物理含义：相邻像素应属于同一类
            feature = probs

        # 计算水平方向梯度 (Right - Current)
        diff_h = torch.abs(feature[:, :, :, :-1] - feature[:, :, :, 1:])
        
        # 计算垂直方向梯度 (Down - Current)
        diff_v = torch.abs(feature[:, :, :-1, :] - feature[:, :, 1:, :])

        #求平均
        loss = torch.mean(diff_h) + torch.mean(diff_v)
        return loss


class FusionLoss(nn.Module):
    """
    【总损失函数封装】
    统一管理垂直损失(OT/KL)和水平损失(Spatial)。
    """
    def __init__(self, mode='ordered', lambda_spatial=0.01):
        """
        Args:
            mode (str): 'ordered' (使用 OT Loss) 或 'unordered' (使用 KL Loss)
            lambda_spatial (float): 空间正则化项的权重。
                                    建议值: 0.01 ~ 0.1。设为 0 则关闭正则化。
        """
        super(FusionLoss, self).__init__()
        self.mode = mode
        self.lambda_spatial = lambda_spatial
        
        # 初始化子损失
        if mode == 'ordered':
            self.main_loss = OTLoss()
            print("🚀 Loss Config: Using [Wasserstein CDF Loss] (Ordered Mode)")
        else:
            self.main_loss = DivergenceLoss()
            print("🧪 Loss Config: Using [KL Divergence Loss] (Unordered Mode)")
            
        if lambda_spatial > 0:
            self.spatial_loss = SpatialSmoothnessLoss(mode=mode)
            print(f"🌊 Loss Config: Spatial Regularization Enabled (lambda={lambda_spatial})")
        else:
            self.spatial_loss = None

    def forward(self, logits, soft_gt):
        """
        Args:
            logits: [B, N, H, W] (网络直接输出)
            soft_gt: [B, N, H, W] (Dataloader 加载的 float32 GT)
        """
        # 1. 计算主损失 (Fidelity)
        if self.mode == 'ordered':
            # OT Loss 需要概率分布 (Softmax后)
            probs = F.softmax(logits, dim=1)
            fidelity_loss = self.main_loss(probs, soft_gt)
        else:
            # KL Loss 需要 logits (内部做 LogSoftmax)
            # 为了接口统一，这里稍微处理一下
            fidelity_loss = self.main_loss(logits, soft_gt)
            # 如果后面要算 spatial loss，需要先算出 probs
            if self.spatial_loss is not None:
                probs = F.softmax(logits, dim=1)

        # 2. 计算空间正则化损失 (Smoothness)
        smoothness_loss = 0.0
        if self.spatial_loss is not None:
            # 无论何种模式，Spatial Loss 都基于 Probs/CDF 计算
            if 'probs' not in locals():
                probs = F.softmax(logits, dim=1)
            smoothness_loss = self.spatial_loss(probs)

        # 3. 总损失
        total_loss = fidelity_loss + self.lambda_spatial * smoothness_loss
        
        return total_loss