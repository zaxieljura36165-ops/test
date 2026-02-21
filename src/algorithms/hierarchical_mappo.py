"""
分层MAPPO算法实现
基于系统建模.md中的分层MAPPO框架设计
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import copy

from configs.system_config import SystemConfig
from configs.mappo_config import MAPPOConfig
from src.algorithms.neural_networks import HierarchicalActorCritic
from src.models.problem_formulation import OptimizationProblem

class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, experience: Dict[str, Any]):
        """添加经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """采样经验"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class HierarchicalMAPPO:
    """分层MAPPO算法"""
    
    def __init__(self, config: MAPPOConfig, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        
        # 网络配置
        self.network_config = self._build_network_config()
        
        # 创建网络
        self.network = HierarchicalActorCritic(self.network_config)
        
        # 优化器
        self.optimizers = self._create_optimizers()
        
        # 经验缓冲区
        self.experience_buffer = ExperienceBuffer(config.BUFFER_CONFIG['buffer_size'])
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        # 设备
        self.device = torch.device(config.DEVICE)
        self.network.to(self.device)
    
    def _build_network_config(self) -> Dict[str, Any]:
        """构建网络配置 - CTDE范式：Critic使用全局状态"""
        return {
            'high_level': {
                'state_dim': self.config.HIGH_LEVEL_CONFIG['state_dim'],
                'action_dim': self.config.HIGH_LEVEL_CONFIG['action_dim'],
                'num_vehicles': self.system_config.NUM_VEHICLES,  # CTDE: 用于计算全局状态维度
                'hidden_sizes': [self.config.HIGH_LEVEL_CONFIG['hidden_size']] * 
                               self.config.HIGH_LEVEL_CONFIG['num_layers']
            },
            'v2i': {
                'state_dim': self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['state_dim'],
                'num_rsu': self.system_config.NUM_RSU,
                'hidden_sizes': [self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['hidden_size']] * 
                               self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['num_layers']
            },
            'v2v': {
                'state_dim': self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['state_dim'],
                'max_neighbors': 5,  # 假设最大邻车数
                'hidden_sizes': [self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['hidden_size']] * 
                               self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['num_layers']
            },
            'local': {
                'state_dim': self.config.LOW_LEVEL_CONFIG['local_computing']['state_dim'],
                'max_freq': self.system_config.VEHICLE_MAX_FREQ,
                'hidden_sizes': [self.config.LOW_LEVEL_CONFIG['local_computing']['hidden_size']] * 
                               self.config.LOW_LEVEL_CONFIG['local_computing']['num_layers']
            }
        }
    
    def _create_optimizers(self) -> Dict[str, optim.Optimizer]:
        """创建优化器"""
        optimizers = {}

        # 高层网络优化器
        high_level_params = list(self.network.high_level_actor.parameters()) + \
                           list(self.network.high_level_critic.parameters())
        optimizers['high_level'] = optim.Adam(
            high_level_params,
            lr=self.config.HIGH_LEVEL_CONFIG['learning_rate'],
            weight_decay=1e-5  # 添加权重衰减，防止过拟合
        )

        # 低层网络优化器
        v2i_params = list(self.network.v2i_actor.parameters()) + \
                    list(self.network.v2i_critic.parameters())
        optimizers['v2i'] = optim.Adam(
            v2i_params,
            lr=self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['learning_rate'],
            weight_decay=1e-5
        )

        v2v_params = list(self.network.v2v_actor.parameters()) + \
                    list(self.network.v2v_critic.parameters())
        optimizers['v2v'] = optim.Adam(
            v2v_params,
            lr=self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['learning_rate'],
            weight_decay=1e-5
        )

        local_params = list(self.network.local_actor.parameters()) + \
                      list(self.network.local_critic.parameters())
        optimizers['local'] = optim.Adam(
            local_params,
            lr=self.config.LOW_LEVEL_CONFIG['local_computing']['learning_rate'],
            weight_decay=1e-5
        )

        return optimizers
    
    def select_action(self, state: torch.Tensor, vehicle_id: int, 
                     task: Any) -> Dict[str, Any]:
        """选择动作
        
        高层决策：
        - alpha: 任务划分比 [0,1]
        - mode: 卸载方式 (0=V2I, 1=V2V)
        
        触发逻辑：
        - 如果 alpha ≈ 0: 完全本地处理，不触发卸载网络
        - 如果 alpha > 0: 根据mode选择V2I或V2V，触发对应低层网络
        - 如果 alpha < 1: 触发本地计算网络
        
        完整版本：保存所有低层网络的log_probs用于训练
        """
        self.network.eval()
        
        with torch.no_grad():
            # 高层决策
            high_level_action = self.network.get_high_level_action(state)
            alpha = high_level_action['alpha'].item()
            
            # 根据alpha和mode决定实际处理方式
            mode = high_level_action['mode_action'].item()  # 0=V2I, 1=V2V
            
            # 初始化低层log_probs和动作值 (用于训练)
            v2i_rsu_log_prob = 0.0
            v2i_power_log_prob = 0.0
            v2v_neighbor_log_prob = 0.0
            v2v_power_log_prob = 0.0
            local_freq_value = 0.5
            local_freq_log_prob = 0.0
            local_mean = 0.5
            local_log_std = 0.0
            
            # 判断是否需要卸载
            if alpha < 0.01:  # alpha≈0，完全本地处理
                mode_name = 'local'
                # 只触发本地计算网络，但也要计算其他网络的log_probs用于训练
                low_level_action = self.network.get_low_level_action(state, 'local')
                local_freq_value = low_level_action.get('freq', 0.5)
                local_freq_log_prob = low_level_action.get('freq_log_prob', 0.0)
                local_mean = low_level_action.get('mean', 0.5)
                local_log_std = low_level_action.get('log_std', 0.0)
                
                # 计算V2I和V2V的log_probs（虽然不执行，但训练时需要）
                v2i_rsu_logits, v2i_power = self.network.v2i_actor(state)
                v2i_rsu_dist = torch.distributions.Categorical(logits=v2i_rsu_logits)
                v2i_rsu_action_tensor = v2i_rsu_dist.sample()
                v2i_rsu_log_prob = v2i_rsu_dist.log_prob(v2i_rsu_action_tensor).item()
                
                v2v_neighbor_logits, v2v_power = self.network.v2v_actor(state)
                v2v_neighbor_dist = torch.distributions.Categorical(logits=v2v_neighbor_logits)
                v2v_neighbor_action_tensor = v2v_neighbor_dist.sample()
                v2v_neighbor_log_prob = v2v_neighbor_dist.log_prob(v2v_neighbor_action_tensor).item()
                
            else:  # alpha > 0，需要卸载
                mode_name = 'V2I' if mode == 0 else 'V2V'
                # 触发对应的卸载网络
                low_level_action = self.network.get_low_level_action(state, mode_name)
                
                # 保存对应网络的log_probs
                if mode_name == 'V2I':
                    v2i_rsu_logits, v2i_power = self.network.v2i_actor(state)
                    v2i_rsu_dist = torch.distributions.Categorical(logits=v2i_rsu_logits)
                    v2i_rsu_action = low_level_action.get('rsu_action', 0)
                    if isinstance(v2i_rsu_action, torch.Tensor):
                        v2i_rsu_action = v2i_rsu_action.item()
                    v2i_rsu_action_tensor = torch.tensor(v2i_rsu_action, device=self.device)
                    v2i_rsu_log_prob = v2i_rsu_dist.log_prob(v2i_rsu_action_tensor).item()
                    
                    # V2V不执行，但要计算log_prob用于训练
                    v2v_neighbor_logits, v2v_power = self.network.v2v_actor(state)
                    v2v_neighbor_dist = torch.distributions.Categorical(logits=v2v_neighbor_logits)
                    v2v_neighbor_action_tensor = v2v_neighbor_dist.sample()
                    v2v_neighbor_log_prob = v2v_neighbor_dist.log_prob(v2v_neighbor_action_tensor).item()
                else:  # V2V
                    v2v_neighbor_logits, v2v_power = self.network.v2v_actor(state)
                    v2v_neighbor_dist = torch.distributions.Categorical(logits=v2v_neighbor_logits)
                    v2v_neighbor_action = low_level_action.get('neighbor_action', 0)
                    if isinstance(v2v_neighbor_action, torch.Tensor):
                        v2v_neighbor_action = v2v_neighbor_action.item()
                    v2v_neighbor_action_tensor = torch.tensor(v2v_neighbor_action, device=self.device)
                    v2v_neighbor_log_prob = v2v_neighbor_dist.log_prob(v2v_neighbor_action_tensor).item()
                    
                    # V2I不执行，但要计算log_prob用于训练
                    v2i_rsu_logits, v2i_power = self.network.v2i_actor(state)
                    v2i_rsu_dist = torch.distributions.Categorical(logits=v2i_rsu_logits)
                    v2i_rsu_action_tensor = v2i_rsu_dist.sample()
                    v2i_rsu_log_prob = v2i_rsu_dist.log_prob(v2i_rsu_action_tensor).item()
                
                # 如果alpha < 1，还需要本地计算网络（部分卸载）
                if alpha < 0.99:
                    local_action = self.network.get_low_level_action(state, 'local')
                    local_freq_value = local_action.get('freq', 0.5)
                    local_freq_log_prob = local_action.get('freq_log_prob', 0.0)
                    local_mean = local_action.get('mean', 0.5)
                    local_log_std = local_action.get('log_std', 0.0)
                    low_level_action['freq'] = local_freq_value
            
            # 组合动作（执行阶段不需要value）
            action = {
                'alpha': alpha,
                'mode': mode_name,
                'mode_log_prob': high_level_action['mode_log_prob'].item(),
                'v2i_rsu_log_prob': v2i_rsu_log_prob,
                'v2v_neighbor_log_prob': v2v_neighbor_log_prob,
                'local_freq_value': local_freq_value,
                'local_freq_log_prob': local_freq_log_prob,
                'local_mean': local_mean,
                'local_log_std': local_log_std,
                **low_level_action
            }
            
            # 转换为numpy
            for key, value in action.items():
                if isinstance(value, torch.Tensor):
                    action[key] = value.item()
        
        return action
    
    def store_experience(self, experience: Dict[str, Any]):
        """存储经验"""
        self.experience_buffer.add(experience)
    
    def update(self, episode: int):
        """更新网络参数"""
        if len(self.experience_buffer) < self.config.BUFFER_CONFIG['min_buffer_size']:
            return

        # 计算当前学习率（带衰减）- 仅用于高层网络
        current_high_level_lr = self._get_current_learning_rate(episode)
        
        # 计算低层网络的学习率（同样衰减，但基准更低）
        decay_ratio = current_high_level_lr / self.config.HIGH_LEVEL_CONFIG['learning_rate']
        current_v2i_lr = self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['learning_rate'] * decay_ratio
        current_v2v_lr = self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['learning_rate'] * decay_ratio
        current_local_lr = self.config.LOW_LEVEL_CONFIG['local_computing']['learning_rate'] * decay_ratio

        # 更新各个优化器的学习率（保持相对比例）
        for param_group in self.optimizers['high_level'].param_groups:
            param_group['lr'] = current_high_level_lr
        for param_group in self.optimizers['v2i'].param_groups:
            param_group['lr'] = current_v2i_lr
        for param_group in self.optimizers['v2v'].param_groups:
            param_group['lr'] = current_v2v_lr
        for param_group in self.optimizers['local'].param_groups:
            param_group['lr'] = current_local_lr

        # 采样经验
        experiences = self.experience_buffer.sample(self.config.TRAINING_CONFIG['batch_size'])

        # 转换为张量
        batch = self._prepare_batch(experiences)

        # 计算优势函数
        advantages = self._compute_advantages(batch)

        # 更新各个网络
        self._update_high_level_network(batch, advantages)
        self._update_low_level_networks(batch, advantages)

        # 清空缓冲区
        self.experience_buffer.clear()

    def _get_current_learning_rate(self, episode: int) -> float:
        """计算当前学习率（带衰减策略）"""
        if not self.config.TRAINING_CONFIG['use_linear_lr_decay']:
            return self.config.HIGH_LEVEL_CONFIG['learning_rate']

        # 线性衰减策略
        start_episode = self.config.TRAINING_CONFIG['lr_decay_start']
        decay_episodes = self.config.TRAINING_CONFIG['lr_decay_episodes']
        initial_lr = self.config.HIGH_LEVEL_CONFIG['learning_rate']

        if episode < start_episode:
            return initial_lr
        elif episode >= start_episode + decay_episodes:
            return initial_lr * 0.1  # 最终学习率为初始的10%
        else:
            # 线性衰减
            decay_progress = (episode - start_episode) / decay_episodes
            current_lr = initial_lr * (1 - 0.9 * decay_progress)  # 衰减到10%
            return current_lr
    
    def _prepare_batch(self, experiences: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """准备批次数据
        
        CTDE修正：
        - local_state用于Actor训练
        - global_state用于Critic训练
        
        PPO修正：
        - 存储old_log_probs用于clip（高层和所有低层网络）
        """
        batch = {
            'local_states': [],  # Actor用局部状态
            'global_states': [],  # Critic用全局状态
            'actions': [],
            'old_log_probs': [],  # 高层mode的PPO需要
            'old_v2i_rsu_log_probs': [],  # V2I RSU选择的log_probs
            'old_v2v_neighbor_log_probs': [],  # V2V邻车选择的log_probs
            'old_local_freq_values': [],  # 本地计算频率值
            'old_local_freq_log_probs': [],  # 本地计算频率的log_probs
            'old_local_means': [],  # 本地计算频率分布的均值
            'old_local_log_stds': [],  # 本地计算频率分布的log_std
            'rewards': [],
            'next_local_states': [],
            'next_global_states': [],
            'dones': []
        }
        
        for exp in experiences:
            action = exp['action']
            batch['local_states'].append(exp.get('local_state', exp.get('state')))
            batch['global_states'].append(exp.get('global_state', exp.get('state')))
            batch['actions'].append(action)
            batch['old_log_probs'].append(exp.get('action_log_prob', 0.0))
            
            # 提取低层网络的log_probs和分布参数
            batch['old_v2i_rsu_log_probs'].append(action.get('v2i_rsu_log_prob', 0.0))
            batch['old_v2v_neighbor_log_probs'].append(action.get('v2v_neighbor_log_prob', 0.0))
            batch['old_local_freq_values'].append(action.get('local_freq_value', 0.5))
            batch['old_local_freq_log_probs'].append(action.get('local_freq_log_prob', 0.0))
            batch['old_local_means'].append(action.get('local_mean', 0.5))
            batch['old_local_log_stds'].append(action.get('local_log_std', 0.0))
            
            batch['rewards'].append(exp['reward'])
            batch['next_local_states'].append(exp.get('next_local_state', exp.get('next_state')))
            batch['next_global_states'].append(exp.get('next_global_state', exp.get('next_state')))
            batch['dones'].append(exp['done'])
        
        # 转换为张量（保持原始tensor形式，不重复转换）
        batch['local_states'] = torch.stack(batch['local_states']).to(self.device)
        batch['global_states'] = torch.stack(batch['global_states']).to(self.device)
        batch['next_local_states'] = torch.stack(batch['next_local_states']).to(self.device)
        batch['next_global_states'] = torch.stack(batch['next_global_states']).to(self.device)
        batch['old_log_probs'] = torch.tensor(batch['old_log_probs'], dtype=torch.float32).to(self.device)
        batch['old_v2i_rsu_log_probs'] = torch.tensor(batch['old_v2i_rsu_log_probs'], dtype=torch.float32).to(self.device)
        batch['old_v2v_neighbor_log_probs'] = torch.tensor(batch['old_v2v_neighbor_log_probs'], dtype=torch.float32).to(self.device)
        batch['old_local_freq_values'] = torch.tensor(batch['old_local_freq_values'], dtype=torch.float32).to(self.device)
        batch['old_local_freq_log_probs'] = torch.tensor(batch['old_local_freq_log_probs'], dtype=torch.float32).to(self.device)
        batch['old_local_means'] = torch.tensor(batch['old_local_means'], dtype=torch.float32).to(self.device)
        batch['old_local_log_stds'] = torch.tensor(batch['old_local_log_stds'], dtype=torch.float32).to(self.device)
        batch['rewards'] = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        batch['dones'] = torch.tensor(batch['dones'], dtype=torch.float32).to(self.device)
        
        return batch
    
    def _compute_advantages(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算高层网络的优势函数（GAE）
        
        CTDE修正：使用全局状态计算values
        """
        rewards = batch['rewards']
        dones = batch['dones']
        gamma = self.config.TRAINING_CONFIG['gamma']
        gae_lambda = self.config.TRAINING_CONFIG['gae_lambda']
        
        # 使用Critic网络和全局状态计算当前状态价值
        with torch.no_grad():
            values = self.network.high_level_critic(batch['global_states']).squeeze(-1)
            next_values = self.network.high_level_critic(batch['next_global_states']).squeeze(-1)
        
        # 计算GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _compute_gae_advantages(self, critic, local_states, next_local_states, rewards, dones):
        """计算GAE优势函数（用于低层网络）
        
        参数：
            critic: Critic网络
            local_states: 当前局部状态
            next_local_states: 下一局部状态
            rewards: 奖励
            dones: 终止标志
        
        返回：
            advantages: GAE优势
        """
        gamma = self.config.TRAINING_CONFIG['gamma']
        gae_lambda = self.config.TRAINING_CONFIG['gae_lambda']
        
        with torch.no_grad():
            values = critic(local_states).squeeze(-1)
            next_values = critic(next_local_states).squeeze(-1)
        
        # 计算GAE
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _update_high_level_network(self, batch: Dict[str, torch.Tensor],
                                  advantages: torch.Tensor):
        """更新高层网络
        
        CTDE修正：
        - Actor使用local_states训练
        - Critic使用global_states训练
        
        PPO修正：
        - 使用clip机制防止策略更新过大
        """
        self.network.train()

        local_states = batch['local_states']  # Actor用局部状态
        global_states = batch['global_states']  # Critic用全局状态
        rewards = batch['rewards']
        old_log_probs = batch['old_log_probs']  # PPO需要的old log probs

        # 前向传播
        # Actor使用局部状态
        alpha, mode_probs, all_log_probs = self.network.high_level_actor(local_states)
        
        # Critic使用全局状态
        values = self.network.high_level_critic(global_states)

        # 计算TD目标（用于value loss）
        with torch.no_grad():
            next_values = self.network.high_level_critic(batch['next_global_states'])
            td_targets = rewards + self.config.TRAINING_CONFIG['gamma'] * next_values.squeeze(-1) * (1 - batch['dones'])

        # 计算策略损失（PPO with clip）
        # 从batch中提取实际采样的mode
        mode_actions = []
        for action in batch['actions']:
            mode_str = action.get('mode', 'V2I')
            mode_idx = 0 if mode_str == 'V2I' else (1 if mode_str == 'V2V' else 0)
            mode_actions.append(mode_idx)
        mode_actions = torch.tensor(mode_actions, dtype=torch.long).to(self.device)
        
        # 计算当前策略下采样动作的log_prob
        current_log_probs = all_log_probs.gather(1, mode_actions.unsqueeze(1)).squeeze(1)
        
        # 计算ratio = exp(current_log_prob - old_log_prob)
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # PPO clip目标函数
        clip_epsilon = self.config.TRAINING_CONFIG.get('ppo_clip_epsilon', 0.2)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失：使用Huber损失
        value_loss = self._compute_huber_loss(values.squeeze(-1), td_targets)

        # 熵损失：鼓励探索
        entropy_loss = -self.config.HIGH_LEVEL_CONFIG['entropy_coef'] * torch.mean(
            torch.sum(-mode_probs * torch.log(mode_probs + 1e-8), dim=-1)
        )

        # 总损失
        total_loss = (policy_loss +
                     self.config.HIGH_LEVEL_CONFIG['value_loss_coef'] * value_loss +
                     entropy_loss)

        # 反向传播
        self.optimizers['high_level'].zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.network.high_level_actor.parameters(),
            self.config.TRAINING_CONFIG['max_grad_norm']
        )
        torch.nn.utils.clip_grad_norm_(
            self.network.high_level_critic.parameters(),
            self.config.TRAINING_CONFIG['max_grad_norm']
        )

        self.optimizers['high_level'].step()

        # 记录统计信息
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['value_losses'].append(value_loss.item())
        self.training_stats['entropy_losses'].append(entropy_loss.item())

    def _compute_huber_loss(self, values: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """计算Huber损失，更稳定的价值损失函数"""
        error = values - targets
        abs_error = torch.abs(error)

        # Huber损失：结合MSE和MAE的优点
        quadratic_loss = 0.5 * torch.pow(error, 2)
        linear_loss = delta * (abs_error - 0.5 * delta)

        loss = torch.where(abs_error <= delta, quadratic_loss, linear_loss)
        return torch.mean(loss)
    
    def _update_low_level_networks(self, batch: Dict[str, torch.Tensor], 
                                     high_level_advantages: torch.Tensor):
        """更新低层网络（完整独立PPO）
        
        关键改进：每个低层网络独立计算GAE优势！
        - V2I网络：使用V2I Critic计算GAE
        - V2V网络：使用V2V Critic计算GAE  
        - 本地网络：使用Local Critic计算GAE
        
        这样每个网络都有自己完整独立的PPO更新循环
        """
        self.network.train()
        
        local_states = batch['local_states']
        next_local_states = batch['next_local_states']
        rewards = batch['rewards']
        dones = batch['dones']
        
        # 从batch中提取动作信息和old log_probs
        actions_data = batch['actions']
        old_v2i_log_probs = batch['old_v2i_rsu_log_probs']
        old_v2v_log_probs = batch['old_v2v_neighbor_log_probs']
        old_local_freq_values = batch['old_local_freq_values']
        old_local_freq_log_probs = batch['old_local_freq_log_probs']
        old_local_means = batch['old_local_means']
        old_local_log_stds = batch['old_local_log_stds']
        
        # 为每个低层网络独立计算GAE优势和TD目标
        # V2I网络：使用自己的Critic
        v2i_advantages = self._compute_gae_advantages(
            self.network.v2i_critic, local_states, next_local_states, rewards, dones
        )
        with torch.no_grad():
            v2i_next_values = self.network.v2i_critic(next_local_states).squeeze(-1)
            v2i_td_targets = rewards + self.config.TRAINING_CONFIG['gamma'] * v2i_next_values * (1 - dones)
        
        # V2V网络：使用自己的Critic
        v2v_advantages = self._compute_gae_advantages(
            self.network.v2v_critic, local_states, next_local_states, rewards, dones
        )
        with torch.no_grad():
            v2v_next_values = self.network.v2v_critic(next_local_states).squeeze(-1)
            v2v_td_targets = rewards + self.config.TRAINING_CONFIG['gamma'] * v2v_next_values * (1 - dones)
        
        # 本地网络：使用自己的Critic
        local_advantages = self._compute_gae_advantages(
            self.network.local_critic, local_states, next_local_states, rewards, dones
        )
        with torch.no_grad():
            local_next_values = self.network.local_critic(next_local_states).squeeze(-1)
            local_td_targets = rewards + self.config.TRAINING_CONFIG['gamma'] * local_next_values * (1 - dones)
        
        # 更新V2I网络（使用自己的优势和TD目标）
        self._update_v2i_network(local_states, batch['global_states'], actions_data, 
                                v2i_advantages, v2i_td_targets, old_v2i_log_probs)
        
        # 更新V2V网络（使用自己的优势和TD目标）
        self._update_v2v_network(local_states, batch['global_states'], actions_data, 
                                v2v_advantages, v2v_td_targets, old_v2v_log_probs)
        
        # 更新本地计算网络（使用自己的优势和TD目标）
        self._update_local_network(local_states, batch['global_states'], actions_data, 
                                   local_advantages, local_td_targets, old_local_freq_values,
                                   old_local_freq_log_probs, old_local_means, old_local_log_stds)
    
    def _update_v2i_network(self, local_states, global_states, actions_data, advantages, td_targets, old_log_probs):
        """更新V2I调度网络"""
        batch_size = local_states.shape[0]
        
        # V2I Actor：重新前向传播获取当前策略的log_probs
        rsu_logits, power = self.network.v2i_actor(local_states)
        rsu_dist = torch.distributions.Categorical(logits=rsu_logits)
        
        # 从actions_data中提取RSU选择动作（确保长度匹配batch_size）
        rsu_actions = []
        for i in range(batch_size):
            if i < len(actions_data):
                action = actions_data[i]
                rsu_action = action.get('rsu_action', 0)
                if isinstance(rsu_action, torch.Tensor):
                    rsu_actions.append(rsu_action.item())
                else:
                    rsu_actions.append(rsu_action)
            else:
                rsu_actions.append(0)
        rsu_actions = torch.tensor(rsu_actions, dtype=torch.long).to(self.device)
        
        # 计算新的log_probs
        new_log_probs = rsu_dist.log_prob(rsu_actions)
        
        # 计算ratio = π_new / π_old
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clip
        clip_epsilon = self.config.TRAINING_CONFIG['ppo_clip_epsilon']
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        v2i_policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵正则化
        entropy = rsu_dist.entropy().mean()
        entropy_coef = self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['entropy_coef']
        
        # V2I Critic：使用自己的Critic计算价值
        v2i_values = self.network.v2i_critic(local_states)
        v2i_value_loss = self._compute_huber_loss(v2i_values.squeeze(-1), td_targets)
        
        # 总损失
        value_loss_coef = self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['value_loss_coef']
        v2i_total_loss = v2i_policy_loss + value_loss_coef * v2i_value_loss - entropy_coef * entropy
        
        # 更新
        self.optimizers['v2i'].zero_grad()
        v2i_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.v2i_actor.parameters(), 
                                       self.config.TRAINING_CONFIG['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.network.v2i_critic.parameters(),
                                       self.config.TRAINING_CONFIG['max_grad_norm'])
        self.optimizers['v2i'].step()
    
    def _update_v2v_network(self, local_states, global_states, actions_data, advantages, td_targets, old_log_probs):
        """更新V2V调度网络"""
        batch_size = local_states.shape[0]
        
        # V2V Actor：重新前向传播获取当前策略的log_probs
        neighbor_logits, power = self.network.v2v_actor(local_states)
        neighbor_dist = torch.distributions.Categorical(logits=neighbor_logits)
        
        # 从actions_data中提取邻车选择动作（确保长度匹配batch_size）
        neighbor_actions = []
        for i in range(batch_size):
            if i < len(actions_data):
                action = actions_data[i]
                neighbor_action = action.get('neighbor_action', 0)
                if isinstance(neighbor_action, torch.Tensor):
                    neighbor_actions.append(neighbor_action.item())
                else:
                    neighbor_actions.append(neighbor_action)
            else:
                neighbor_actions.append(0)
        neighbor_actions = torch.tensor(neighbor_actions, dtype=torch.long).to(self.device)
        
        # 计算新的log_probs
        new_log_probs = neighbor_dist.log_prob(neighbor_actions)
        
        # 计算ratio = π_new / π_old
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clip
        clip_epsilon = self.config.TRAINING_CONFIG['ppo_clip_epsilon']
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        v2v_policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵正则化
        entropy = neighbor_dist.entropy().mean()
        entropy_coef = self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['entropy_coef']
        
        # V2V Critic：使用自己的Critic计算价值
        v2v_values = self.network.v2v_critic(local_states)
        v2v_value_loss = self._compute_huber_loss(v2v_values.squeeze(-1), td_targets)
        
        # 总损失
        value_loss_coef = self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['value_loss_coef']
        v2v_total_loss = v2v_policy_loss + value_loss_coef * v2v_value_loss - entropy_coef * entropy
        
        # 更新
        self.optimizers['v2v'].zero_grad()
        v2v_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.v2v_actor.parameters(),
                                       self.config.TRAINING_CONFIG['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.network.v2v_critic.parameters(),
                                       self.config.TRAINING_CONFIG['max_grad_norm'])
        self.optimizers['v2v'].step()
    
    def _update_local_network(self, local_states, global_states, actions_data, advantages, td_targets, 
                             old_freq_values, old_freq_log_probs, old_means, old_log_stds):
        """更新本地计算网络（完整高斯PPO）"""
        batch_size = local_states.shape[0]
        
        # 本地Actor：重新前向传播获取当前分布参数
        mean, log_std = self.network.local_actor(local_states)
        mean = mean.squeeze(-1)  # [batch_size]
        log_std = log_std.squeeze(-1)  # [batch_size]
        std = log_std.exp()
        
        # 构建高斯分布
        normal_dist = torch.distributions.Normal(mean, std)
        
        # 计算old actions的新log_probs
        new_log_probs = normal_dist.log_prob(old_freq_values)
        
        # 计算ratio = π_new / π_old
        ratio = torch.exp(new_log_probs - old_freq_log_probs)
        
        # PPO clip
        clip_epsilon = self.config.TRAINING_CONFIG['ppo_clip_epsilon']
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        local_policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵正则化（鼓励探索）
        entropy = normal_dist.entropy().mean()
        entropy_coef = self.config.LOW_LEVEL_CONFIG['local_computing']['entropy_coef']
        
        # 本地Critic：使用自己的Critic计算价值
        local_values = self.network.local_critic(local_states)
        local_value_loss = self._compute_huber_loss(local_values.squeeze(-1), td_targets)
        
        # 总损失
        value_loss_coef = self.config.LOW_LEVEL_CONFIG['local_computing']['value_loss_coef']
        local_total_loss = local_policy_loss + value_loss_coef * local_value_loss - entropy_coef * entropy
        
        # 更新
        self.optimizers['local'].zero_grad()
        local_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.local_actor.parameters(),
                                       self.config.TRAINING_CONFIG['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(self.network.local_critic.parameters(),
                                       self.config.TRAINING_CONFIG['max_grad_norm'])
        self.optimizers['local'].step()
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dicts': {k: v.state_dict() for k, v in self.optimizers.items()},
            'training_stats': self.training_stats,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        for name, optimizer in self.optimizers.items():
            if name in checkpoint['optimizer_state_dicts']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][name])
        
        self.training_stats = checkpoint['training_stats']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'episode_rewards': self.training_stats['episode_rewards'],
            'episode_lengths': self.training_stats['episode_lengths'],
            'avg_policy_loss': np.mean(self.training_stats['policy_losses']) if self.training_stats['policy_losses'] else 0,
            'avg_value_loss': np.mean(self.training_stats['value_losses']) if self.training_stats['value_losses'] else 0,
            'avg_entropy_loss': np.mean(self.training_stats['entropy_losses']) if self.training_stats['entropy_losses'] else 0
        }
    
    def reset_training_stats(self):
        """重置训练统计"""
        for key in self.training_stats:
            self.training_stats[key].clear()
