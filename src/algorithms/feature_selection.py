"""
特征选择和注意力机制模块
用于优化状态表示，减少冗余信息，提升训练效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import StandardScaler
import copy

class FeatureSelector(nn.Module):
    """特征选择网络模块"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = None,
                 selection_ratio: float = 0.7, use_attention: bool = True):
        """
        初始化特征选择器

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度（如果为None，则基于selection_ratio计算）
            selection_ratio: 特征选择比例
            use_attention: 是否使用注意力机制
        """
        super(FeatureSelector, self).__init__()

        self.input_dim = input_dim
        self.use_attention = use_attention

        if output_dim is None:
            output_dim = max(32, int(input_dim * selection_ratio))

        self.output_dim = output_dim

        # 特征重要性评估网络
        self.feature_importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出每个特征的重要性分数 [0,1]
        )

        # 特征门控网络（如果使用注意力）
        if use_attention:
            self.gate_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        # 特征重构网络
        self.reconstruction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 特征统计记录
        self.feature_stats = {
            'importance_scores': [],
            'selection_mask': None,
            'feature_correlation': None
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征张量 [batch_size, input_dim]

        Returns:
            selected_features: 选择的特征 [batch_size, output_dim]
            importance_scores: 特征重要性分数 [batch_size, input_dim]
            attention_weights: 注意力权重 [batch_size, input_dim]
        """
        batch_size = x.size(0)

        # 计算特征重要性分数
        importance_scores = self.feature_importance_net(x)  # [batch_size, input_dim]

        # 计算注意力权重（如果启用）
        if self.use_attention:
            attention_weights = self.gate_net(x)  # [batch_size, input_dim]
            # 结合重要性和注意力
            combined_weights = importance_scores * attention_weights
        else:
            combined_weights = importance_scores

        # 特征选择和重构
        selected_features = self._select_features(x, combined_weights)

        return selected_features, importance_scores, combined_weights

    def _select_features(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """基于权重选择特征"""
        # 使用top-k选择最重要的特征
        batch_size, num_features = weights.size()

        # 计算每个样本的top-k特征索引
        _, topk_indices = torch.topk(weights, k=self.output_dim, dim=-1)

        # 创建选择掩码
        mask = torch.zeros_like(weights)
        mask.scatter_(-1, topk_indices, 1.0)

        # 应用掩码并重构
        masked_features = x * mask
        selected_features = self.reconstruction_net(masked_features)

        return selected_features

    def update_feature_stats(self, x: torch.Tensor, episode: int):
        """更新特征统计信息"""
        with torch.no_grad():
            _, importance_scores, _ = self.forward(x)

            # 记录重要性分数统计
            self.feature_stats['importance_scores'].append(importance_scores.mean(dim=0).cpu().numpy())

            # 每100个episode计算一次特征相关性
            if episode % 100 == 0 and len(self.feature_stats['importance_scores']) >= 10:
                self._compute_feature_correlation()

    def _compute_feature_correlation(self):
        """计算特征相关性矩阵"""
        if len(self.feature_stats['importance_scores']) < 2:
            return

        # 将重要性分数历史转换为numpy数组
        importance_history = np.array(self.feature_stats['importance_scores'])

        # 计算特征间相关性
        correlation_matrix = np.corrcoef(importance_history.T)

        # 检测高度相关的特征对（相关性 > 0.8）
        high_corr_pairs = []
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if abs(correlation_matrix[i, j]) > 0.8:
                    high_corr_pairs.append((i, j, correlation_matrix[i, j]))

        self.feature_stats['feature_correlation'] = {
            'matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs
        }

    def get_feature_importance_ranking(self) -> List[Tuple[int, float]]:
        """获取特征重要性排名"""
        if not self.feature_stats['importance_scores']:
            return []

        # 计算平均重要性分数
        avg_importance = np.mean(self.feature_stats['importance_scores'], axis=0)

        # 返回排序后的特征索引和重要性分数
        ranking = [(i, float(score)) for i, score in enumerate(avg_importance)]
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking

    def get_redundant_features(self) -> List[Tuple[int, int, float]]:
        """获取冗余特征对（高度相关的特征）"""
        if not self.feature_stats['feature_correlation']:
            return []

        return self.feature_stats['feature_correlation']['high_corr_pairs']

class AttentionFeatureSelector(nn.Module):
    """基于注意力机制的特征选择器"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8,
                 output_dim: int = None, dropout: float = 0.1):
        """
        初始化注意力特征选择器

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            output_dim: 输出特征维度
            dropout: dropout概率
        """
        super(AttentionFeatureSelector, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        if output_dim is None:
            output_dim = max(32, input_dim // 2)

        self.output_dim = output_dim

        # 多头注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 特征聚合网络
        self.aggregation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征张量 [batch_size, input_dim]

        Returns:
            selected_features: 选择的特征 [batch_size, output_dim]
            attention_weights: 注意力权重 [batch_size, input_dim]
        """
        batch_size = x.size(0)

        # 计算门控权重
        gate_weights = self.gate_net(x)  # [batch_size, input_dim]

        # 应用门控
        gated_features = x * gate_weights

        # 自注意力机制
        attn_output, attn_weights = self.attention(
            gated_features.unsqueeze(1),  # [batch_size, 1, input_dim]
            gated_features.unsqueeze(1),
            gated_features.unsqueeze(1)
        )

        # 聚合特征
        attn_output = attn_output.squeeze(1)  # [batch_size, input_dim]
        selected_features = self.aggregation_net(attn_output)

        return selected_features, attn_weights.squeeze(1)

class AdaptiveFeatureSelector(nn.Module):
    """自适应特征选择器，结合统计方法和深度学习"""

    def __init__(self, input_dim: int, config: Dict = None):
        """
        初始化自适应特征选择器

        Args:
            input_dim: 输入特征维度
            config: 配置参数
        """
        super(AdaptiveFeatureSelector, self).__init__()

        self.input_dim = input_dim

        # 默认配置
        if config is None:
            config = {
                'hidden_dim': 128,
                'selection_ratio': 0.7,
                'use_attention': True,
                'update_frequency': 100,
                'correlation_threshold': 0.8
            }

        self.config = config

        # 特征选择网络
        self.feature_selector = FeatureSelector(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            selection_ratio=config['selection_ratio'],
            use_attention=config['use_attention']
        )

        # 统计特征选择器（离线计算）
        self.statistical_selector = StatisticalFeatureSelector(
            correlation_threshold=config['correlation_threshold']
        )

        # 特征选择历史
        self.selection_history = []
        self.episode_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        selected_features, _, _ = self.feature_selector(x)
        return selected_features

    def update_selection_strategy(self, x: torch.Tensor, episode: int):
        """更新特征选择策略"""
        self.episode_counter += 1

        # 更新特征统计
        self.feature_selector.update_feature_stats(x, episode)

        # 定期更新统计特征选择
        if episode % self.config['update_frequency'] == 0:
            self._update_statistical_selection()

        # 记录选择历史
        if episode % 10 == 0:
            importance_ranking = self.feature_selector.get_feature_importance_ranking()
            self.selection_history.append({
                'episode': episode,
                'top_features': importance_ranking[:10],  # 记录最重要的10个特征
                'redundant_features': self.feature_selector.get_redundant_features()
            })

    def _update_statistical_selection(self):
        """更新统计特征选择"""
        # 获取当前特征重要性排名
        importance_ranking = self.feature_selector.get_feature_importance_ranking()

        # 获取冗余特征对
        redundant_pairs = self.feature_selector.get_redundant_features()

        # 更新统计选择器
        self.statistical_selector.update_selection(importance_ranking, redundant_pairs)

        # 可以基于统计信息动态调整网络参数
        self._adapt_network_parameters()

    def _adapt_network_parameters(self):
        """自适应调整网络参数"""
        # 基于特征重要性分布调整网络结构
        importance_ranking = self.feature_selector.get_feature_importance_ranking()

        if importance_ranking:
            # 计算重要性分布的熵
            importances = np.array([score for _, score in importance_ranking])
            importances = importances / np.sum(importances)  # 归一化

            entropy = -np.sum(importances * np.log(importances + 1e-8))

            # 如果特征重要性分布很集中，可能需要减少输出维度
            if entropy < 2.0:  # 经验阈值
                current_output_dim = self.feature_selector.output_dim
                new_output_dim = max(16, current_output_dim - 4)
                self.feature_selector.output_dim = new_output_dim

                # 重新初始化重构网络
                self.feature_selector.reconstruction_net = nn.Sequential(
                    nn.Linear(self.input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, new_output_dim)
                )

    def get_selection_summary(self) -> Dict:
        """获取特征选择总结"""
        return {
            'total_features': self.input_dim,
            'selected_features': self.feature_selector.output_dim,
            'selection_ratio': self.feature_selector.output_dim / self.input_dim,
            'importance_ranking': self.feature_selector.get_feature_importance_ranking(),
            'redundant_features': self.feature_selector.get_redundant_features(),
            'selection_history': self.selection_history[-10:]  # 最近10次记录
        }

class StatisticalFeatureSelector:
    """统计特征选择器，基于相关性和互信息"""

    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        self.selected_features = set()
        self.redundant_features = set()

    def update_selection(self, importance_ranking: List, redundant_pairs: List):
        """更新特征选择"""
        # 基于重要性排名选择特征
        min_importance = 0.1  # 最低重要性阈值

        for feature_idx, importance in importance_ranking:
            if importance >= min_importance:
                self.selected_features.add(feature_idx)
            else:
                break

        # 标记冗余特征
        for idx1, idx2, correlation in redundant_pairs:
            if correlation > self.correlation_threshold:
                # 保留重要性更高的特征
                if importance_ranking[idx1][1] > importance_ranking[idx2][1]:
                    self.redundant_features.add(idx2)
                else:
                    self.redundant_features.add(idx1)

        # 从选择集中移除冗余特征
        self.selected_features -= self.redundant_features

    def get_selected_features(self) -> List[int]:
        """获取选中的特征索引"""
        return sorted(list(self.selected_features))

    def get_redundant_features(self) -> List[int]:
        """获取冗余特征索引"""
        return sorted(list(self.redundant_features))


