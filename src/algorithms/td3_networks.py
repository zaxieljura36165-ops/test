"""
TD3网络结构定义
用于分层TD3算法的Actor和Critic网络

特点：
1. 双Q网络（TD3核心）
2. 支持混合动作空间（连续+离散）
3. 层归一化提高训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class LayerNormMLP(nn.Module):
    """带层归一化的MLP"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list,
                 use_layer_norm: bool = True, output_activation: str = None):
        super(LayerNormMLP, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_dim))
        
        if output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TD3Actor(nn.Module):
    """
    TD3 Actor网络
    输出确定性动作（连续值）
    
    对于混合动作空间：
    - 连续动作直接输出
    - 离散动作输出logits，执行时取argmax或添加Gumbel噪声
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list = [256, 256],
                 use_layer_norm: bool = True, action_bounds: Dict = None):
        super(TD3Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds or {}
        
        # 特征提取网络
        self.feature_net = LayerNormMLP(
            state_dim, hidden_sizes[-1], hidden_sizes[:-1],
            use_layer_norm=use_layer_norm
        )
        
        # 动作输出头（不同动作可能有不同的激活函数）
        self.action_head = nn.Linear(hidden_sizes[-1], action_dim)
        
        self._init_final_layer()
    
    def _init_final_layer(self):
        """初始化最后一层（小权重）"""
        nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            action: 原始动作输出 [batch_size, action_dim]
        """
        features = self.feature_net(state)
        action = self.action_head(features)
        return action
    
    def get_action(self, state: torch.Tensor, apply_bounds: bool = True) -> torch.Tensor:
        """
        获取动作（带边界处理）
        
        对于混合动作空间，不同维度使用不同的激活函数：
        - alpha (任务划分比): sigmoid → [0, 1]
        - mode_logits: 原始logits
        - power: sigmoid → [0, 1]
        """
        raw_action = self.forward(state)
        
        if apply_bounds:
            # 对不同维度应用不同的边界
            # 假设动作布局: [alpha, mode_logit, power]
            processed = []
            for i in range(self.action_dim):
                if i == 0:  # alpha - 任务划分比
                    processed.append(torch.sigmoid(raw_action[:, i:i+1]))
                elif i == self.action_dim - 1:  # power - 发射功率
                    processed.append(torch.sigmoid(raw_action[:, i:i+1]))
                else:  # mode_logits - 离散选择的连续松弛
                    processed.append(raw_action[:, i:i+1])
            
            return torch.cat(processed, dim=1)
        
        return raw_action


class TD3Critic(nn.Module):
    """
    TD3 Critic网络（双Q网络）
    
    输入: state + action
    输出: 两个Q值估计
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list = [256, 256],
                 use_layer_norm: bool = True):
        super(TD3Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        input_dim = state_dim + action_dim
        
        # Q1网络
        self.q1_net = LayerNormMLP(
            input_dim, 1, hidden_sizes,
            use_layer_norm=use_layer_norm
        )
        
        # Q2网络
        self.q2_net = LayerNormMLP(
            input_dim, 1, hidden_sizes,
            use_layer_norm=use_layer_norm
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            q1, q2: 两个Q值估计
        """
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回Q1值（用于策略更新）"""
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)


class HighLevelTD3Actor(TD3Actor):
    """
    高层TD3 Actor
    
    输出:
    - alpha: 任务划分比 [0, 1]
    - mode_logits: 卸载模式logits [V2I, V2V] (2维)
    - base_power: 基础功率 [0, 1]
    
    总输出维度: 4
    """
    
    def __init__(self, state_dim: int, hidden_sizes: list = [256, 256, 128],
                 use_layer_norm: bool = True):
        # 高层动作: [alpha, v2i_logit, v2v_logit, power]
        super(HighLevelTD3Actor, self).__init__(
            state_dim, 4, hidden_sizes, use_layer_norm
        )
    
    def get_action(self, state: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        获取高层动作
        
        Args:
            state: 状态
            temperature: Gumbel-Softmax温度（训练时使用）
        
        Returns:
            Dict包含alpha, mode_probs, mode_action, power
        """
        raw_action = self.forward(state)
        
        # 解析各部分
        alpha = torch.sigmoid(raw_action[:, 0:1])  # [0, 1]
        mode_logits = raw_action[:, 1:3]           # [V2I, V2V] logits
        power = torch.sigmoid(raw_action[:, 3:4]) # [0, 1]
        
        # 模式选择（Gumbel-Softmax或argmax）
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        if self.training and temperature > 0:
            # 训练时使用Gumbel-Softmax进行可微分采样
            mode_action = F.gumbel_softmax(mode_logits, tau=temperature, hard=True)
        else:
            # 推理时使用argmax
            mode_action = torch.zeros_like(mode_probs)
            mode_action.scatter_(1, mode_probs.argmax(dim=-1, keepdim=True), 1.0)
        
        return {
            'alpha': alpha,
            'mode_logits': mode_logits,
            'mode_probs': mode_probs,
            'mode_action': mode_action,  # one-hot
            'mode_idx': mode_probs.argmax(dim=-1),  # 0=V2I, 1=V2V
            'power': power,
            'raw_action': raw_action  # 用于存储和Critic计算
        }


class HighLevelTD3Critic(TD3Critic):
    """
    高层TD3 Critic（CTDE范式）
    
    修正：使用局部状态与单车动作匹配，避免状态-动作维度不一致
    """
    
    def __init__(self, local_state_dim: int, num_vehicles: int,
                 action_dim: int = 4, hidden_sizes: list = [256, 256],
                 use_layer_norm: bool = True):
        global_state_dim = local_state_dim
        super(HighLevelTD3Critic, self).__init__(
            global_state_dim, action_dim, hidden_sizes, use_layer_norm
        )
        self.local_state_dim = local_state_dim
        self.num_vehicles = num_vehicles


class LowLevelTD3Actor(TD3Actor):
    """低层TD3 Actor基类"""
    pass


class V2ISchedulerTD3Actor(LowLevelTD3Actor):
    """
    V2I调度 TD3 Actor
    
    输出:
    - rsu_logits: RSU选择logits (num_rsu维)
    - power: 发射功率 [0, 1]
    """
    
    def __init__(self, state_dim: int, num_rsu: int, hidden_sizes: list = [128, 128],
                 use_layer_norm: bool = True):
        self.num_rsu = num_rsu
        action_dim = num_rsu + 1  # RSU logits + power
        super(V2ISchedulerTD3Actor, self).__init__(
            state_dim, action_dim, hidden_sizes, use_layer_norm
        )
    
    def get_action(self, state: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """获取V2I调度动作"""
        raw_action = self.forward(state)
        
        rsu_logits = raw_action[:, :self.num_rsu]
        power = torch.sigmoid(raw_action[:, self.num_rsu:self.num_rsu+1])
        
        rsu_probs = F.softmax(rsu_logits, dim=-1)
        
        if self.training and temperature > 0:
            rsu_action = F.gumbel_softmax(rsu_logits, tau=temperature, hard=True)
        else:
            rsu_action = torch.zeros_like(rsu_probs)
            rsu_action.scatter_(1, rsu_probs.argmax(dim=-1, keepdim=True), 1.0)
        
        return {
            'rsu_logits': rsu_logits,
            'rsu_probs': rsu_probs,
            'rsu_action': rsu_action,  # one-hot
            'rsu_idx': rsu_probs.argmax(dim=-1),
            'power': power,
            'raw_action': raw_action
        }


class V2VSchedulerTD3Actor(LowLevelTD3Actor):
    """
    V2V调度 TD3 Actor
    
    输出:
    - neighbor_logits: 邻车选择logits (max_neighbors维)
    - power: 发射功率 [0, 1]
    """
    
    def __init__(self, state_dim: int, max_neighbors: int, hidden_sizes: list = [128, 128],
                 use_layer_norm: bool = True):
        self.max_neighbors = max_neighbors
        action_dim = max_neighbors + 1  # neighbor logits + power
        super(V2VSchedulerTD3Actor, self).__init__(
            state_dim, action_dim, hidden_sizes, use_layer_norm
        )
    
    def get_action(self, state: torch.Tensor, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """获取V2V调度动作"""
        raw_action = self.forward(state)
        
        neighbor_logits = raw_action[:, :self.max_neighbors]
        power = torch.sigmoid(raw_action[:, self.max_neighbors:self.max_neighbors+1])
        
        neighbor_probs = F.softmax(neighbor_logits, dim=-1)
        
        if self.training and temperature > 0:
            neighbor_action = F.gumbel_softmax(neighbor_logits, tau=temperature, hard=True)
        else:
            neighbor_action = torch.zeros_like(neighbor_probs)
            neighbor_action.scatter_(1, neighbor_probs.argmax(dim=-1, keepdim=True), 1.0)
        
        return {
            'neighbor_logits': neighbor_logits,
            'neighbor_probs': neighbor_probs,
            'neighbor_action': neighbor_action,  # one-hot
            'neighbor_idx': neighbor_probs.argmax(dim=-1),
            'power': power,
            'raw_action': raw_action
        }


class LocalComputingTD3Actor(LowLevelTD3Actor):
    """
    本地计算 TD3 Actor
    
    输出:
    - freq: 本地计算频率 [0, 1]
    """
    
    def __init__(self, state_dim: int, hidden_sizes: list = [64, 64],
                 use_layer_norm: bool = True):
        super(LocalComputingTD3Actor, self).__init__(
            state_dim, 1, hidden_sizes, use_layer_norm
        )
    
    def get_action(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """获取本地计算动作"""
        raw_action = self.forward(state)
        freq = torch.sigmoid(raw_action)
        
        return {
            'freq': freq,
            'raw_action': raw_action
        }


class HierarchicalTD3Networks(nn.Module):
    """
    分层TD3网络集合
    包含高层和所有低层的Actor-Critic网络
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(HierarchicalTD3Networks, self).__init__()
        
        self.config = config
        
        # 高层网络
        self.high_actor = HighLevelTD3Actor(
            config['high_level']['state_dim'],
            config['high_level']['hidden_sizes'],
            config['high_level'].get('use_layer_norm', True)
        )
        
        self.high_critic = HighLevelTD3Critic(
            config['high_level']['state_dim'],
            config['high_level'].get('num_vehicles', 10),
            action_dim=4,
            hidden_sizes=config['high_level']['hidden_sizes'][:2],
            use_layer_norm=config['high_level'].get('use_layer_norm', True)
        )
        
        # 高层目标网络
        self.high_actor_target = HighLevelTD3Actor(
            config['high_level']['state_dim'],
            config['high_level']['hidden_sizes'],
            config['high_level'].get('use_layer_norm', True)
        )
        
        self.high_critic_target = HighLevelTD3Critic(
            config['high_level']['state_dim'],
            config['high_level'].get('num_vehicles', 10),
            action_dim=4,
            hidden_sizes=config['high_level']['hidden_sizes'][:2],
            use_layer_norm=config['high_level'].get('use_layer_norm', True)
        )
        
        # 低层网络 - V2I
        self.v2i_actor = V2ISchedulerTD3Actor(
            config['v2i']['state_dim'],
            config['v2i']['num_rsu'],
            config['v2i']['hidden_sizes']
        )
        self.v2i_critic = TD3Critic(
            config['v2i']['state_dim'],
            config['v2i']['num_rsu'] + 1,
            config['v2i']['hidden_sizes']
        )
        self.v2i_actor_target = V2ISchedulerTD3Actor(
            config['v2i']['state_dim'],
            config['v2i']['num_rsu'],
            config['v2i']['hidden_sizes']
        )
        self.v2i_critic_target = TD3Critic(
            config['v2i']['state_dim'],
            config['v2i']['num_rsu'] + 1,
            config['v2i']['hidden_sizes']
        )
        
        # 低层网络 - V2V
        self.v2v_actor = V2VSchedulerTD3Actor(
            config['v2v']['state_dim'],
            config['v2v']['max_neighbors'],
            config['v2v']['hidden_sizes']
        )
        self.v2v_critic = TD3Critic(
            config['v2v']['state_dim'],
            config['v2v']['max_neighbors'] + 1,
            config['v2v']['hidden_sizes']
        )
        self.v2v_actor_target = V2VSchedulerTD3Actor(
            config['v2v']['state_dim'],
            config['v2v']['max_neighbors'],
            config['v2v']['hidden_sizes']
        )
        self.v2v_critic_target = TD3Critic(
            config['v2v']['state_dim'],
            config['v2v']['max_neighbors'] + 1,
            config['v2v']['hidden_sizes']
        )
        
        # 低层网络 - Local
        self.local_actor = LocalComputingTD3Actor(
            config['local']['state_dim'],
            config['local']['hidden_sizes']
        )
        self.local_critic = TD3Critic(
            config['local']['state_dim'],
            1,
            config['local']['hidden_sizes']
        )
        self.local_actor_target = LocalComputingTD3Actor(
            config['local']['state_dim'],
            config['local']['hidden_sizes']
        )
        self.local_critic_target = TD3Critic(
            config['local']['state_dim'],
            1,
            config['local']['hidden_sizes']
        )
        
        # 初始化目标网络
        self._init_target_networks()
    
    def _init_target_networks(self):
        """初始化目标网络（硬拷贝）"""
        self.high_actor_target.load_state_dict(self.high_actor.state_dict())
        self.high_critic_target.load_state_dict(self.high_critic.state_dict())
        
        self.v2i_actor_target.load_state_dict(self.v2i_actor.state_dict())
        self.v2i_critic_target.load_state_dict(self.v2i_critic.state_dict())
        
        self.v2v_actor_target.load_state_dict(self.v2v_actor.state_dict())
        self.v2v_critic_target.load_state_dict(self.v2v_critic.state_dict())
        
        self.local_actor_target.load_state_dict(self.local_actor.state_dict())
        self.local_critic_target.load_state_dict(self.local_critic.state_dict())
    
    def soft_update(self, tau: float):
        """软更新所有目标网络"""
        self._soft_update_network(self.high_actor, self.high_actor_target, tau)
        self._soft_update_network(self.high_critic, self.high_critic_target, tau)
        
        self._soft_update_network(self.v2i_actor, self.v2i_actor_target, tau)
        self._soft_update_network(self.v2i_critic, self.v2i_critic_target, tau)
        
        self._soft_update_network(self.v2v_actor, self.v2v_actor_target, tau)
        self._soft_update_network(self.v2v_critic, self.v2v_critic_target, tau)
        
        self._soft_update_network(self.local_actor, self.local_actor_target, tau)
        self._soft_update_network(self.local_critic, self.local_critic_target, tau)
    
    def _soft_update_network(self, source: nn.Module, target: nn.Module, tau: float):
        """单个网络的软更新"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

