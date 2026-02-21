"""
MADDPG网络结构
包括Actor和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class MADDPGActor(nn.Module):
    """分层 MADDPG Actor网络（分布式执行）
    
    输入：局部状态
    输出：混合动作空间（连续+离散），采用分层结构：
    - 高层：alpha + mode (V2I/V2V)
    - 低层：根据 mode 选择 target 与 power
    - 本地：frequency
    """
    
    def __init__(self, state_dim: int, hidden_sizes: list = [256, 128]):
        super(MADDPGActor, self).__init__()
        
        self.state_dim = state_dim
        
        # 共享特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )
        
        # === 高层动作头 ===
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # V2I, V2V
        )
        
        # === 低层动作头 ===
        # V2I 目标与功率
        self.v2i_target_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 5 RSU + 1 备用节点
        )
        self.v2i_power_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # V2V 目标与功率
        self.v2v_target_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 5 邻车 + 1 备用节点
        )
        self.v2v_power_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 本地计算频率
        self.frequency_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, temperature: float = 1.0, 
                hard: bool = False) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            state: [batch, state_dim]
            temperature: Gumbel-Softmax温度
            hard: 是否使用hard Gumbel-Softmax
        
        Returns:
            actions: {
                'alpha': [batch, 1],
                'mode': [batch, 2] (Gumbel-Softmax),
                'target': [batch, 6] (Gumbel-Softmax),
                'power': [batch, 1],
                'frequency': [batch, 1]
            }
        """
        features = self.feature_net(state)
        
        # === 高层 ===
        alpha = self.alpha_head(features)
        mode_logits = self.mode_head(features)
        mode = F.gumbel_softmax(mode_logits, tau=temperature, hard=hard, dim=-1)
        
        # === 低层 ===
        v2i_logits = self.v2i_target_head(features)
        v2i_target = F.gumbel_softmax(v2i_logits, tau=temperature, hard=hard, dim=-1)
        v2i_power = self.v2i_power_head(features)
        
        v2v_logits = self.v2v_target_head(features)
        v2v_target = F.gumbel_softmax(v2v_logits, tau=temperature, hard=hard, dim=-1)
        v2v_power = self.v2v_power_head(features)
        
        # 本地频率
        frequency = self.frequency_head(features)
        
        # === 分层融合 ===
        if hard:
            mode_idx = torch.argmax(mode, dim=-1)
            target = torch.where(
                mode_idx.unsqueeze(-1) == 0, v2i_target, v2v_target
            )
            power = torch.where(
                mode_idx.unsqueeze(-1) == 0, v2i_power, v2v_power
            )
        else:
            mode_probs = F.softmax(mode_logits, dim=-1)
            target = mode_probs[:, 0:1] * F.softmax(v2i_logits, dim=-1) + \
                     mode_probs[:, 1:2] * F.softmax(v2v_logits, dim=-1)
            power = mode_probs[:, 0:1] * v2i_power + mode_probs[:, 1:2] * v2v_power
        
        return {
            'alpha': alpha,
            'mode': mode,
            'target': target,
            'power': power,
            'frequency': frequency
        }


class MADDPGCritic(nn.Module):
    """MADDPG Critic网络（集中式训练）
    
    输入：全局状态 + 所有智能体的动作
    输出：Q值
    """
    
    def __init__(self, global_state_dim: int, num_agents: int, 
                 action_dims: Dict[str, int], hidden_sizes: list = [512, 256, 128]):
        """
        Args:
            global_state_dim: 全局状态维度
            num_agents: 智能体数量
            action_dims: 动作维度字典 {'alpha': 1, 'mode': 2, 'target': 6, 'power': 1, 'frequency': 1}
        """
        super(MADDPGCritic, self).__init__()
        
        self.num_agents = num_agents
        self.action_dims = action_dims
        
        # 计算总动作维度
        total_action_dim = sum(action_dims.values()) * num_agents
        
        # Q网络
        self.q_net = nn.Sequential(
            nn.Linear(global_state_dim + total_action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1)
        )
    
    def forward(self, global_state: torch.Tensor, 
                actions: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """前向传播
        
        Args:
            global_state: [batch, global_state_dim]
            actions: {agent_id: {'alpha': [batch, 1], 'mode': [batch, 2], ...}}
        
        Returns:
            q_value: [batch, 1]
        """
        # 拼接所有智能体的动作
        action_list = []
        for agent_id in range(self.num_agents):
            agent_actions = actions[agent_id]
            action_list.extend([
                agent_actions['alpha'],
                agent_actions['mode'],
                agent_actions['target'],
                agent_actions['power'],
                agent_actions['frequency']
            ])
        
        all_actions = torch.cat(action_list, dim=-1)
        
        # 拼接全局状态和所有动作
        q_input = torch.cat([global_state, all_actions], dim=-1)
        
        q_value = self.q_net(q_input)
        return q_value


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    """软更新目标网络
    
    target = tau * source + (1 - tau) * target
    
    Args:
        source: 源网络
        target: 目标网络
        tau: 更新系数
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(source: nn.Module, target: nn.Module):
    """硬更新目标网络
    
    target = source
    """
    target.load_state_dict(source.state_dict())

