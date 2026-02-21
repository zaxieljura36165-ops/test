"""
神经网络定义
用于分层MAPPO算法的Actor和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
from .feature_selection import AdaptiveFeatureSelector, AttentionFeatureSelector

class MLP(nn.Module):
    """多层感知机基类"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list, 
                 activation: str = 'tanh', output_activation: str = None):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        
        # 构建网络层
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_dim))
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _get_activation(self, activation: str):
        """获取激活函数"""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=-1)
        }
        return activations.get(activation, nn.Tanh())
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class HighLevelActor(nn.Module):
    """高层策略网络（Actor）带注意力机制"""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list = [128, 64],
                 use_attention: bool = True, attention_config: Dict = None):
        super(HighLevelActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention

        # 注意力特征选择器
        if use_attention:
            if attention_config is None:
                attention_config = {
                    'hidden_dim': 128,
                    'num_heads': 8,
                    'output_dim': max(64, state_dim // 2)
                }
            self.feature_selector = AttentionFeatureSelector(
                input_dim=state_dim,
                **attention_config
            )
        else:
            self.feature_extractor = MLP(state_dim, hidden_sizes[0], hidden_sizes[1:], 'tanh')

        # 共享特征提取层（注意力输出后）
        selected_feature_dim = attention_config['output_dim'] if use_attention else hidden_sizes[0]
        self.final_feature_extractor = MLP(selected_feature_dim, hidden_sizes[0], hidden_sizes[1:], 'tanh')

        # 任务划分比输出（连续动作）
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_sizes[0], 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出[0,1]
        )

        # 卸载方式输出（离散动作）
        # 修正：根据算法.md，只有V2I和V2V两个选择，α=0时自动为本地处理
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_sizes[0], 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # V2I、V2V（α≈0时自动本地处理）
            nn.Softmax(dim=-1)
        )

        # 注意力统计
        self.attention_stats = {
            'attention_weights_history': [],
            'feature_importance_history': []
        }
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        返回: (alpha, mode_probs, log_probs)
        """
        if self.use_attention:
            # 使用注意力机制选择和聚焦关键特征
            selected_features, attention_weights = self.feature_selector(state)
            features = self.final_feature_extractor(selected_features)

            # 记录注意力统计（可选，用于分析）
            if self.training:
                self._update_attention_stats(attention_weights)
        else:
            features = self.feature_extractor(state)

        # 任务划分比
        alpha = self.alpha_head(features)

        # 卸载方式概率
        mode_probs = self.mode_head(features)

        # 计算对数概率
        log_probs = torch.log(mode_probs + 1e-8)

        return alpha, mode_probs, log_probs

    def _update_attention_stats(self, attention_weights: torch.Tensor):
        """更新注意力统计信息"""
        # 记录注意力权重历史（用于分析特征重要性）
        if len(self.attention_stats['attention_weights_history']) < 1000:  # 限制历史长度
            self.attention_stats['attention_weights_history'].append(
                attention_weights.mean(dim=0).detach().cpu().numpy()
            )

    def get_attention_summary(self) -> Dict:
        """获取注意力机制总结"""
        if not self.use_attention:
            return {'attention_enabled': False}

        # 计算平均注意力权重
        if self.attention_stats['attention_weights_history']:
            avg_weights = np.mean(self.attention_stats['attention_weights_history'], axis=0)

            # 获取最重要的特征
            top_features = np.argsort(avg_weights)[-10:][::-1]  # 最重要的10个特征

            return {
                'attention_enabled': True,
                'avg_attention_weights': avg_weights,
                'top_important_features': top_features.tolist(),
                'attention_entropy': -np.sum(avg_weights * np.log(avg_weights + 1e-8)),
                'feature_redundancy_score': self._compute_feature_redundancy(avg_weights)
            }
        else:
            return {'attention_enabled': True, 'stats_available': False}

    def _compute_feature_redundancy(self, attention_weights: np.ndarray) -> float:
        """计算特征冗余度（注意力分布的熵）"""
        # 归一化注意力权重
        normalized_weights = attention_weights / (np.sum(attention_weights) + 1e-8)

        # 计算熵（熵越高，特征越分散，冗余度越低）
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))

        # 归一化到[0,1]，0表示完全冗余，1表示完全分散
        max_entropy = np.log(len(attention_weights))
        redundancy_score = 1 - (entropy / max_entropy)

        return float(redundancy_score)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作"""
        alpha, mode_probs, log_probs = self.forward(state)
        
        # 采样卸载方式
        mode_dist = torch.distributions.Categorical(mode_probs)
        mode_action = mode_dist.sample()
        mode_log_prob = mode_dist.log_prob(mode_action)
        
        return alpha, mode_action, mode_log_prob

class HighLevelCritic(nn.Module):
    """高层价值网络（Critic）带注意力机制
    
    CTDE范式实现：
    - 训练时：接收全局状态（所有车辆状态拼接，维度 = num_vehicles * state_dim）
    - 执行时：不使用Critic（只有Actor使用局部状态执行）
    
    根据算法.md第28行：
    全局状态 S_t = {s_{1,t}, s_{2,t}, ..., s_{|V|,t}} 包含所有智能体的局部观测
    仅在集中式训练的Critic网络中使用
    """

    def __init__(self, state_dim: int, num_vehicles: int = 10, 
                 hidden_sizes: list = [128, 64],
                 use_attention: bool = True, attention_config: Dict = None):
        super(HighLevelCritic, self).__init__()

        # CTDE: Critic使用全局状态维度
        self.local_state_dim = state_dim  # 单个车辆的状态维度
        self.num_vehicles = num_vehicles
        self.global_state_dim = state_dim * num_vehicles  # 全局状态维度
        self.use_attention = use_attention

        # 注意力特征选择器（处理全局状态）
        if use_attention:
            if attention_config is None:
                attention_config = {
                    'hidden_dim': 256,  # 增大隐藏层以处理更大的全局状态
                    'num_heads': 8,
                    'output_dim': max(128, self.global_state_dim // 4)  # 从全局状态中提取关键特征
                }
            self.feature_selector = AttentionFeatureSelector(
                input_dim=self.global_state_dim,  # 使用全局状态维度
                **attention_config
            )
        else:
            self.feature_extractor = MLP(self.global_state_dim, hidden_sizes[0], hidden_sizes[1:], 'tanh')

        # 价值网络（注意力输出后）
        selected_feature_dim = attention_config['output_dim'] if use_attention else self.global_state_dim
        # 价值网络使用较少的隐藏层以提高稳定性
        value_hidden_sizes = hidden_sizes[:2] if len(hidden_sizes) >= 2 else hidden_sizes
        self.value_net = MLP(selected_feature_dim, 1, value_hidden_sizes, 'tanh')

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值
        
        Args:
            global_state: 全局状态张量 [batch_size, global_state_dim]
                         或 [batch_size, num_vehicles, local_state_dim]
        
        Returns:
            value: 状态价值 [batch_size, 1]
        """
        # 如果输入是三维的（batch, num_vehicles, state_dim），展平为二维
        if len(global_state.shape) == 3:
            batch_size = global_state.shape[0]
            global_state = global_state.view(batch_size, -1)
        
        if self.use_attention:
            # 使用注意力机制选择和聚焦关键特征
            selected_features, _ = self.feature_selector(global_state)
            value = self.value_net(selected_features)
        else:
            value = self.value_net(global_state)

        return value

class LowLevelActor(nn.Module):
    """低层策略网络（Actor）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list = [64, 32]):
        super(LowLevelActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 网络主体
        self.network = MLP(state_dim, action_dim, hidden_sizes, 'tanh')
        
        # 动作标准化
        self.action_std = nn.Parameter(torch.ones(action_dim) * 0.5)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        action_mean = self.forward(state)
        action_std = self.action_std.expand_as(action_mean)
        
        # 创建正态分布
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # 采样动作
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class LowLevelCritic(nn.Module):
    """低层价值网络（Critic）"""
    
    def __init__(self, state_dim: int, hidden_sizes: list = [64, 32]):
        super(LowLevelCritic, self).__init__()
        
        self.state_dim = state_dim
        self.value_net = MLP(state_dim, 1, hidden_sizes, 'tanh')
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值"""
        return self.value_net(state)

class V2ISchedulerActor(LowLevelActor):
    """V2I调度网络"""
    
    def __init__(self, state_dim: int, num_rsu: int, hidden_sizes: list = [64, 32]):
        # 动作维度：目标RSU选择 + 发射功率
        action_dim = num_rsu + 1  # RSU选择 + 功率
        super(V2ISchedulerActor, self).__init__(state_dim, action_dim, hidden_sizes)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        output = self.network(state)
        
        # 分离RSU选择和功率（在特征维度上切片，不是batch维度）
        rsu_logits = output[..., :-1]  # RSU选择logits [batch_size, num_rsu]
        power = torch.sigmoid(output[..., -1:])  # 功率[0,1] [batch_size, 1]
        
        return rsu_logits, power

class V2VSchedulerActor(LowLevelActor):
    """V2V调度网络"""
    
    def __init__(self, state_dim: int, max_neighbors: int, hidden_sizes: list = [64, 32]):
        # 动作维度：目标邻车选择 + 发射功率
        action_dim = max_neighbors + 1  # 邻车选择 + 功率
        super(V2VSchedulerActor, self).__init__(state_dim, action_dim, hidden_sizes)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        output = self.network(state)
        
        # 分离邻车选择和功率（在特征维度上切片，不是batch维度）
        neighbor_logits = output[..., :-1]  # 邻车选择logits [batch_size, max_neighbors]
        power = torch.sigmoid(output[..., -1:])  # 功率[0,1] [batch_size, 1]
        
        return neighbor_logits, power

class LocalComputingActor(LowLevelActor):
    """本地计算资源分配网络（连续动作 - 高斯策略）"""
    
    def __init__(self, state_dim: int, max_freq: float, hidden_sizes: list = [32, 16]):
        # 动作维度：本地计算频率（输出mean和log_std）
        action_dim = 2  # mean + log_std
        super(LocalComputingActor, self).__init__(state_dim, action_dim, hidden_sizes)
        self.max_freq = max_freq
        
        # 限制log_std的范围，避免数值不稳定
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state: torch.Tensor) -> tuple:
        """前向传播
        
        返回：
            mean: 频率均值 [0, 1]
            log_std: 对数标准差
        """
        output = self.network(state)
        
        # 分离均值和log_std
        mean = torch.sigmoid(output[..., 0:1])  # [batch_size, 1], 范围[0, 1]
        log_std = output[..., 1:2]  # [batch_size, 1]
        
        # 限制log_std范围
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std

class HierarchicalActorCritic(nn.Module):
    """分层Actor-Critic网络"""
    
    def __init__(self, config: Dict[str, Any]):
        super(HierarchicalActorCritic, self).__init__()
        
        self.config = config
        
        # 高层网络
        self.high_level_actor = HighLevelActor(
            config['high_level']['state_dim'],
            config['high_level']['action_dim'],
            config['high_level']['hidden_sizes'],
            use_attention=config['high_level'].get('use_attention', False),
            attention_config=config['high_level'].get('attention_config')
        )
        
        # CTDE: Critic使用全局状态（所有车辆状态拼接）
        self.high_level_critic = HighLevelCritic(
            config['high_level']['state_dim'],
            num_vehicles=config['high_level'].get('num_vehicles', 10),
            hidden_sizes=config['high_level']['hidden_sizes'],
            use_attention=config['high_level'].get('use_attention', False),
            attention_config=config['high_level'].get('attention_config')
        )
        
        # 低层网络
        self.v2i_actor = V2ISchedulerActor(
            config['v2i']['state_dim'],
            config['v2i']['num_rsu'],
            config['v2i']['hidden_sizes']
        )
        
        self.v2i_critic = LowLevelCritic(
            config['v2i']['state_dim'],
            config['v2i']['hidden_sizes']
        )
        
        self.v2v_actor = V2VSchedulerActor(
            config['v2v']['state_dim'],
            config['v2v']['max_neighbors'],
            config['v2v']['hidden_sizes']
        )
        
        self.v2v_critic = LowLevelCritic(
            config['v2v']['state_dim'],
            config['v2v']['hidden_sizes']
        )
        
        self.local_actor = LocalComputingActor(
            config['local']['state_dim'],
            config['local']['max_freq'],
            config['local']['hidden_sizes']
        )
        
        self.local_critic = LowLevelCritic(
            config['local']['state_dim'],
            config['local']['hidden_sizes']
        )
    
    def get_high_level_action(self, state: torch.Tensor, training: bool = False, 
                             global_state: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """获取高层动作
        
        Args:
            state: 局部状态（Actor使用）
            training: 是否训练模式
            global_state: 全局状态（Critic使用，仅训练时需要）
        """
        alpha, mode_action, mode_log_prob = self.high_level_actor.sample_action(state)
        
        result = {
            'alpha': alpha,
            'mode_action': mode_action,
            'mode_log_prob': mode_log_prob,
        }
        
        # 只在训练时计算value（Critic使用全局状态）
        if training and global_state is not None:
            value = self.high_level_critic(global_state)
            result['value'] = value
        
        return result
    
    def get_low_level_action(self, state: torch.Tensor, mode: str) -> Dict[str, torch.Tensor]:
        """获取低层动作"""
        if mode == 'V2I':
            rsu_logits, power = self.v2i_actor(state)
            value = self.v2i_critic(state)
            
            # 采样RSU选择
            rsu_dist = torch.distributions.Categorical(logits=rsu_logits)
            rsu_action = rsu_dist.sample()
            rsu_log_prob = rsu_dist.log_prob(rsu_action)
            
            return {
                'rsu_action': rsu_action,
                'rsu_log_prob': rsu_log_prob,
                'power': power,
                'value': value
            }
        
        elif mode == 'V2V':
            neighbor_logits, power = self.v2v_actor(state)
            value = self.v2v_critic(state)
            
            # 采样邻车选择
            neighbor_dist = torch.distributions.Categorical(logits=neighbor_logits)
            neighbor_action = neighbor_dist.sample()
            neighbor_log_prob = neighbor_dist.log_prob(neighbor_action)
            
            return {
                'neighbor_action': neighbor_action,
                'neighbor_log_prob': neighbor_log_prob,
                'power': power,
                'value': value
            }
        
        elif mode == 'local':
            mean, log_std = self.local_actor(state)
            value = self.local_critic(state)
            
            # 采样频率（高斯分布）
            std = log_std.exp()
            normal_dist = torch.distributions.Normal(mean, std)
            freq_action = normal_dist.sample()
            
            # 截断到[0, 1]
            freq_action = torch.clamp(freq_action, 0.0, 1.0)
            
            # 计算log_prob（注意：clamp后需要修正log_prob）
            freq_log_prob = normal_dist.log_prob(freq_action).sum(dim=-1)
            
            return {
                'freq': freq_action.squeeze(-1),
                'freq_log_prob': freq_log_prob,
                'mean': mean.squeeze(-1),
                'log_std': log_std.squeeze(-1),
                'value': value
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_all_values(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取所有网络的价值估计"""
        return {
            'high_level': self.high_level_critic(state),
            'v2i': self.v2i_critic(state),
            'v2v': self.v2v_critic(state),
            'local': self.local_critic(state)
        }
