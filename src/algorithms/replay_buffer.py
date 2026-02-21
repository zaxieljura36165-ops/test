"""
多智能体经验回放缓冲区
用于MADDPG等off-policy算法
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from collections import deque
import random


class MAReplayBuffer:
    """多智能体经验回放缓冲区"""
    
    def __init__(self, buffer_size: int, num_agents: int):
        """
        Args:
            buffer_size: 缓冲区最大容量
            num_agents: 智能体数量
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, experience: Dict[str, Any]):
        """添加经验
        
        Args:
            experience: {
                'states': Dict[agent_id, state_tensor],  # 局部状态
                'global_state': tensor,  # 全局状态
                'actions': Dict[agent_id, action_dict],  # 动作
                'rewards': Dict[agent_id, float],  # 奖励
                'next_states': Dict[agent_id, state_tensor],  # 下一状态
                'next_global_state': tensor,  # 下一全局状态
                'done': bool  # 是否结束
            }
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """随机采样一批经验
        
        Returns:
            batch: {
                'states': Dict[agent_id, tensor[batch, state_dim]],
                'global_states': tensor[batch, global_state_dim],
                'actions': Dict[agent_id, Dict[action_name, tensor[batch, ...]]],
                'rewards': Dict[agent_id, tensor[batch]],
                'next_states': Dict[agent_id, tensor[batch, state_dim]],
                'next_global_states': tensor[batch, global_state_dim],
                'dones': tensor[batch]
            }
        """
        experiences = random.sample(self.buffer, batch_size)
        
        batch = {
            'states': {agent_id: [] for agent_id in range(self.num_agents)},
            'global_states': [],
            'actions': {agent_id: {'alpha': [], 'mode': [], 'target': [], 
                                   'power': [], 'frequency': []} 
                       for agent_id in range(self.num_agents)},
            'rewards': {agent_id: [] for agent_id in range(self.num_agents)},
            'next_states': {agent_id: [] for agent_id in range(self.num_agents)},
            'next_global_states': [],
            'dones': []
        }
        
        for exp in experiences:
            # 状态
            for agent_id in range(self.num_agents):
                if agent_id in exp['states']:
                    batch['states'][agent_id].append(exp['states'][agent_id])
                    batch['next_states'][agent_id].append(exp['next_states'][agent_id])
                    batch['rewards'][agent_id].append(exp['rewards'][agent_id])
                else:
                    # 没有任务的车辆：填充零状态和零奖励
                    state_dim = list(exp['states'].values())[0].shape[0]
                    batch['states'][agent_id].append(torch.zeros(state_dim))
                    batch['next_states'][agent_id].append(torch.zeros(state_dim))
                    batch['rewards'][agent_id].append(0.0)
            
            batch['global_states'].append(exp['global_state'])
            batch['next_global_states'].append(exp['next_global_state'])
            batch['dones'].append(float(exp['done']))
            
            # 动作
            for agent_id in range(self.num_agents):
                if agent_id in exp['actions']:
                    action = exp['actions'][agent_id]
                    batch['actions'][agent_id]['alpha'].append(action.get('alpha', 0.0))
                    batch['actions'][agent_id]['mode'].append(action.get('mode_idx', 0))
                    batch['actions'][agent_id]['target'].append(action.get('target', 0))
                    batch['actions'][agent_id]['power'].append(action.get('power', 0.0))
                    batch['actions'][agent_id]['frequency'].append(action.get('frequency', 0.0))
                else:
                    batch['actions'][agent_id]['alpha'].append(0.0)
                    batch['actions'][agent_id]['mode'].append(0)
                    batch['actions'][agent_id]['target'].append(0)
                    batch['actions'][agent_id]['power'].append(0.0)
                    batch['actions'][agent_id]['frequency'].append(0.0)
        
        # 转换为tensor
        for agent_id in range(self.num_agents):
            batch['states'][agent_id] = torch.stack(batch['states'][agent_id])
            batch['next_states'][agent_id] = torch.stack(batch['next_states'][agent_id])
            batch['rewards'][agent_id] = torch.tensor(batch['rewards'][agent_id], dtype=torch.float32)
            
            for key in batch['actions'][agent_id]:
                batch['actions'][agent_id][key] = torch.tensor(
                    batch['actions'][agent_id][key], dtype=torch.float32
                )
        
        batch['global_states'] = torch.stack(batch['global_states'])
        batch['next_global_states'] = torch.stack(batch['next_global_states'])
        batch['dones'] = torch.tensor(batch['dones'], dtype=torch.float32)
        
        return batch
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

