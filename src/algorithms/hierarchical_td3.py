"""
分层TD3算法实现
结合情境感知自适应噪声控制器

核心创新点：
1. 分层决策架构：高层决定宏观策略，低层执行具体参数
2. 情境感知噪声：根据系统状态（负载、优先级、信道等）动态调整探索噪声
3. 双缓冲区经验回放：分离探索性经验和确定性经验
4. TD3核心机制：双Q网络 + 延迟更新 + 目标平滑
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import copy

from configs.system_config import SystemConfig
from configs.td3_config import TD3Config
from src.algorithms.td3_networks import HierarchicalTD3Networks
from src.algorithms.noise_controller import TaskOffloadNoiseController, DualReplayBuffer


class HierarchicalTD3:
    """
    分层TD3算法
    
    架构：
    - 高层TD3：决定任务划分比、卸载模式、基础功率
    - 低层TD3-V2I：选择RSU、细化功率
    - 低层TD3-V2V：选择邻车、细化功率
    - 低层TD3-Local：决定本地计算频率
    
    创新：
    - 情境感知噪声控制器根据任务属性和系统状态动态调整探索强度
    - 双缓冲区机制平衡探索与利用
    """
    
    def __init__(self, config: TD3Config, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.device = torch.device(config.DEVICE)
        
        # 构建网络配置
        self.network_config = self._build_network_config()
        
        # 创建网络
        self.networks = HierarchicalTD3Networks(self.network_config)
        self.networks.to(self.device)
        
        # 创建优化器
        self.optimizers = self._create_optimizers()
        
        # 情境感知噪声控制器
        self.noise_controller = TaskOffloadNoiseController(config.NOISE_CONFIG)
        
        # 双缓冲区经验回放
        self.replay_buffer = DualReplayBuffer(
            capacity=config.BUFFER_CONFIG['buffer_size'],
            state_dim=config.HIGH_LEVEL_CONFIG['state_dim'],
            action_dim=4 + 6 + 6 + 1,  # high + v2i + v2v + local
        )
        
        # 训练计数器
        self.total_steps = 0
        self.update_count = 0
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'success_rates': []
        }
    
    def _build_network_config(self) -> Dict[str, Any]:
        """构建网络配置"""
        return {
            'high_level': {
                'state_dim': self.config.HIGH_LEVEL_CONFIG['state_dim'],
                'hidden_sizes': self.config.HIGH_LEVEL_CONFIG['hidden_sizes'],
                'use_layer_norm': self.config.HIGH_LEVEL_CONFIG.get('use_layer_norm', True),
                'num_vehicles': self.system_config.NUM_VEHICLES,
            },
            'v2i': {
                'state_dim': self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['state_dim'],
                'num_rsu': self.system_config.NUM_RSU,
                'hidden_sizes': self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['hidden_sizes'],
            },
            'v2v': {
                'state_dim': self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['state_dim'],
                'max_neighbors': 5,
                'hidden_sizes': self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['hidden_sizes'],
            },
            'local': {
                'state_dim': self.config.LOW_LEVEL_CONFIG['local_computing']['state_dim'],
                'hidden_sizes': self.config.LOW_LEVEL_CONFIG['local_computing']['hidden_sizes'],
            }
        }
    
    def _create_optimizers(self) -> Dict[str, optim.Optimizer]:
        """创建优化器"""
        optimizers = {}
        
        # 高层优化器
        optimizers['high_actor'] = optim.Adam(
            self.networks.high_actor.parameters(),
            lr=self.config.HIGH_LEVEL_CONFIG['learning_rate']
        )
        optimizers['high_critic'] = optim.Adam(
            self.networks.high_critic.parameters(),
            lr=self.config.HIGH_LEVEL_CONFIG['learning_rate']
        )
        
        # V2I优化器
        optimizers['v2i_actor'] = optim.Adam(
            self.networks.v2i_actor.parameters(),
            lr=self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['learning_rate']
        )
        optimizers['v2i_critic'] = optim.Adam(
            self.networks.v2i_critic.parameters(),
            lr=self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['learning_rate']
        )
        
        # V2V优化器
        optimizers['v2v_actor'] = optim.Adam(
            self.networks.v2v_actor.parameters(),
            lr=self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['learning_rate']
        )
        optimizers['v2v_critic'] = optim.Adam(
            self.networks.v2v_critic.parameters(),
            lr=self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['learning_rate']
        )
        
        # Local优化器
        optimizers['local_actor'] = optim.Adam(
            self.networks.local_actor.parameters(),
            lr=self.config.LOW_LEVEL_CONFIG['local_computing']['learning_rate']
        )
        optimizers['local_critic'] = optim.Adam(
            self.networks.local_critic.parameters(),
            lr=self.config.LOW_LEVEL_CONFIG['local_computing']['learning_rate']
        )
        
        return optimizers
    
    def select_action(self, local_state: torch.Tensor, vehicle_id: int,
                     context_info: Dict[str, Any] = None,
                     deterministic: bool = False) -> Dict[str, Any]:
        """
        选择动作（带情境感知噪声）
        
        Args:
            local_state: 车辆局部状态
            vehicle_id: 车辆ID
            context_info: 情境信息（用于噪声控制器）
            deterministic: 是否确定性动作（无噪声）
        
        Returns:
            动作字典
        """
        # 训练时需要开启训练模式以启用Gumbel-Softmax的离散探索
        if deterministic:
            self.networks.eval()
        else:
            self.networks.train()
        
        with torch.no_grad():
            # 确保状态是正确的形状
            if local_state.dim() == 1:
                local_state = local_state.unsqueeze(0)
            
            # === 高层决策 ===
            high_action = self.networks.high_actor.get_action(local_state)
            
            alpha = high_action['alpha'].item()
            mode_idx = high_action['mode_idx'].item()  # 0=V2I, 1=V2V
            base_power = high_action['power'].item()
            
            # === 情境感知噪声 ===
            if not deterministic and context_info is not None:
                noise_scale = self.noise_controller.get_noise_scale(context_info)
                
                # 对高层动作添加噪声（增强探索：noise_scale * 1.5）
                alpha_noise = np.random.normal(0, noise_scale * 1.5 * 0.2)
                alpha = np.clip(alpha + alpha_noise, 0, 1)
                
                power_noise = np.random.normal(0, noise_scale * 1.5 * 0.1)
                base_power = np.clip(base_power + power_noise, 0, 1)
                
                # 模式选择：在Phase1时完全随机
                if self.noise_controller.should_use_random_action():
                    mode_idx = np.random.randint(0, 2)
            
            # === 根据模式选择低层动作 ===
            if alpha < 0.01:
                # 完全本地处理
                mode_name = 'local'
                local_action = self.networks.local_actor.get_action(local_state)
                freq = local_action['freq'].item()
                
                if not deterministic and context_info is not None:
                    freq_noise = np.random.normal(0, noise_scale * 0.1)
                    freq = np.clip(freq + freq_noise, 0, 1)
                
                action = {
                    'alpha': alpha,
                    'mode': mode_name,
                    'mode_idx': -1,  # 本地
                    'freq': freq,
                    'power': 0.0,
                    'rsu_action': 0,
                    'neighbor_action': 0,
                    'raw_high_action': high_action['raw_action'].cpu().numpy(),
                    'raw_low_action': local_action['raw_action'].cpu().numpy(),
                }
            
            elif mode_idx == 0:  # V2I
                mode_name = 'V2I'
                v2i_action = self.networks.v2i_actor.get_action(local_state)
                rsu_idx = v2i_action['rsu_idx'].item()
                power = v2i_action['power'].item()
                
                if not deterministic and context_info is not None:
                    power_noise = np.random.normal(0, noise_scale * 0.1)
                    power = np.clip(power + power_noise, 0, 1)
                    
                    if self.noise_controller.should_use_random_action():
                        rsu_idx = np.random.randint(0, self.system_config.NUM_RSU)
                
                # 本地部分（如果alpha < 1）
                local_action = self.networks.local_actor.get_action(local_state)
                freq = local_action['freq'].item() if alpha < 0.99 else 0.0
                
                action = {
                    'alpha': alpha,
                    'mode': mode_name,
                    'mode_idx': 0,
                    'rsu_action': rsu_idx,
                    'power': power,
                    'freq': freq,
                    'neighbor_action': 0,
                    'raw_high_action': high_action['raw_action'].cpu().numpy(),
                    'raw_low_action': v2i_action['raw_action'].cpu().numpy(),
                }
            
            else:  # V2V
                mode_name = 'V2V'
                v2v_action = self.networks.v2v_actor.get_action(local_state)
                neighbor_idx = v2v_action['neighbor_idx'].item()
                power = v2v_action['power'].item()
                
                if not deterministic and context_info is not None:
                    power_noise = np.random.normal(0, noise_scale * 0.1)
                    power = np.clip(power + power_noise, 0, 1)
                    
                    if self.noise_controller.should_use_random_action():
                        neighbor_idx = np.random.randint(0, 5)
                
                # 本地部分
                local_action = self.networks.local_actor.get_action(local_state)
                freq = local_action['freq'].item() if alpha < 0.99 else 0.0
                
                action = {
                    'alpha': alpha,
                    'mode': mode_name,
                    'mode_idx': 1,
                    'neighbor_action': neighbor_idx,
                    'power': power,
                    'freq': freq,
                    'rsu_action': 0,
                    'raw_high_action': high_action['raw_action'].cpu().numpy(),
                    'raw_low_action': v2v_action['raw_action'].cpu().numpy(),
                }
        
        return action
    
    def store_experience(self, experience: Dict[str, Any], is_noisy: bool = True):
        """
        存储经验到双缓冲区
        
        Args:
            experience: 经验字典
            is_noisy: 是否是探索性经验
        """
        # 转换所有tensor为numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x
        
        local_state = to_numpy(experience['local_state'])
        next_local_state = to_numpy(experience['next_local_state'])
        global_state = to_numpy(experience.get('global_state'))
        next_global_state = to_numpy(experience.get('next_global_state'))
        
        self.replay_buffer.push(
            state=local_state,
            action=experience['action'],
            reward=experience['reward'],
            next_state=next_local_state,
            done=experience['done'],
            global_state=global_state,
            next_global_state=next_global_state,
            is_noisy=is_noisy
        )
    
    def update(self, episode: int):
        """
        更新网络参数
        
        Args:
            episode: 当前训练轮数
        """
        if len(self.replay_buffer) < self.config.BUFFER_CONFIG['min_buffer_size']:
            return
        
        self.networks.train()
        self.update_count += 1
        
        # 获取确定性经验采样比例
        det_ratio = self.noise_controller.get_deterministic_ratio()
        
        # 从双缓冲区采样
        batch = self.replay_buffer.sample(
            self.config.BUFFER_CONFIG['batch_size'],
            self.device,
            deterministic_ratio=det_ratio
        )
        
        # 如果采样失败，跳过本次更新
        if batch is None:
            return
        
        # 更新各网络
        critic_loss = self._update_critics(batch)
        
        # 延迟策略更新
        if self.update_count % self.config.TD3_CONFIG['policy_delay'] == 0:
            actor_loss = self._update_actors(batch)
            
            # 软更新目标网络
            self.networks.soft_update(self.config.TD3_CONFIG['tau'])
            
            self.training_stats['actor_losses'].append(actor_loss)
        
        self.training_stats['critic_losses'].append(critic_loss)
        
        # 学习率衰减
        if self.config.TRAINING_CONFIG['use_lr_decay']:
            self._update_learning_rate(episode)
    
    def _update_critics(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新所有Critic网络"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        # 高层Critic使用局部状态（与单车动作一致）
        # 低层Critic本来就是局部状态
        
        gamma = self.config.TD3_CONFIG['gamma']
        policy_noise = self.config.TD3_CONFIG['policy_noise']
        noise_clip = self.config.TD3_CONFIG['noise_clip']
        
        total_critic_loss = 0.0
        
        # === 更新高层Critic ===
        with torch.no_grad():
            # 目标动作（带噪声平滑）
            next_high_action = self.networks.high_actor_target.get_action(next_states)
            next_action = next_high_action['raw_action']
            
            # 添加目标策略噪声
            noise = torch.clamp(
                torch.randn_like(next_action) * policy_noise,
                -noise_clip, noise_clip
            )
            next_action = next_action + noise
            
            # 计算目标Q值（取两个Q网络的最小值）
            target_q1, target_q2 = self.networks.high_critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        # 当前Q值
        high_action = self._extract_high_action(actions)
        current_q1, current_q2 = self.networks.high_critic(states, high_action)
        
        # Critic损失
        high_critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.optimizers['high_critic'].zero_grad()
        high_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.networks.high_critic.parameters(),
            self.config.TRAINING_CONFIG['max_grad_norm']
        )
        self.optimizers['high_critic'].step()
        
        total_critic_loss += high_critic_loss.item()
        
        # === 更新低层Critic（类似流程）===
        total_critic_loss += self._update_low_level_critics(batch, gamma, policy_noise, noise_clip)
        
        return total_critic_loss
    
    def _update_low_level_critics(self, batch: Dict[str, torch.Tensor],
                                   gamma: float, policy_noise: float, noise_clip: float) -> float:
        """更新低层Critic网络"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        total_loss = 0.0
        
        # V2I Critic
        v2i_action = self._extract_v2i_action(actions)
        with torch.no_grad():
            next_v2i_action = self.networks.v2i_actor_target.get_action(next_states)
            next_action = next_v2i_action['raw_action']
            noise = torch.clamp(torch.randn_like(next_action) * policy_noise, -noise_clip, noise_clip)
            next_action = next_action + noise
            
            target_q1, target_q2 = self.networks.v2i_critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        current_q1, current_q2 = self.networks.v2i_critic(states, v2i_action)
        v2i_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.optimizers['v2i_critic'].zero_grad()
        v2i_loss.backward()
        self.optimizers['v2i_critic'].step()
        total_loss += v2i_loss.item()
        
        # V2V Critic（类似V2I）
        v2v_action = self._extract_v2v_action(actions)
        with torch.no_grad():
            next_v2v_action = self.networks.v2v_actor_target.get_action(next_states)
            next_action = next_v2v_action['raw_action']
            noise = torch.clamp(torch.randn_like(next_action) * policy_noise, -noise_clip, noise_clip)
            next_action = next_action + noise
            
            target_q1, target_q2 = self.networks.v2v_critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        current_q1, current_q2 = self.networks.v2v_critic(states, v2v_action)
        v2v_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.optimizers['v2v_critic'].zero_grad()
        v2v_loss.backward()
        self.optimizers['v2v_critic'].step()
        total_loss += v2v_loss.item()
        
        # Local Critic
        local_action = self._extract_local_action(actions)
        with torch.no_grad():
            next_local_action = self.networks.local_actor_target.get_action(next_states)
            next_action = next_local_action['raw_action']
            noise = torch.clamp(torch.randn_like(next_action) * policy_noise, -noise_clip, noise_clip)
            next_action = next_action + noise
            
            target_q1, target_q2 = self.networks.local_critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + gamma * (1 - dones) * target_q
        
        current_q1, current_q2 = self.networks.local_critic(states, local_action)
        local_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.optimizers['local_critic'].zero_grad()
        local_loss.backward()
        self.optimizers['local_critic'].step()
        total_loss += local_loss.item()
        
        return total_loss
    
    def _update_actors(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新所有Actor网络"""
        states = batch['states']
        
        total_actor_loss = 0.0
        
        # === 高层Actor ===
        high_action = self.networks.high_actor.get_action(states)
        high_actor_loss = -self.networks.high_critic.q1(states, high_action['raw_action']).mean()
        
        self.optimizers['high_actor'].zero_grad()
        high_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.networks.high_actor.parameters(),
            self.config.TRAINING_CONFIG['max_grad_norm']
        )
        self.optimizers['high_actor'].step()
        
        total_actor_loss += high_actor_loss.item()
        
        # === 低层Actor ===
        # V2I
        v2i_action = self.networks.v2i_actor.get_action(states)
        v2i_actor_loss = -self.networks.v2i_critic.q1(states, v2i_action['raw_action']).mean()
        
        self.optimizers['v2i_actor'].zero_grad()
        v2i_actor_loss.backward()
        self.optimizers['v2i_actor'].step()
        total_actor_loss += v2i_actor_loss.item()
        
        # V2V
        v2v_action = self.networks.v2v_actor.get_action(states)
        v2v_actor_loss = -self.networks.v2v_critic.q1(states, v2v_action['raw_action']).mean()
        
        self.optimizers['v2v_actor'].zero_grad()
        v2v_actor_loss.backward()
        self.optimizers['v2v_actor'].step()
        total_actor_loss += v2v_actor_loss.item()
        
        # Local
        local_action = self.networks.local_actor.get_action(states)
        local_actor_loss = -self.networks.local_critic.q1(states, local_action['raw_action']).mean()
        
        self.optimizers['local_actor'].zero_grad()
        local_actor_loss.backward()
        self.optimizers['local_actor'].step()
        total_actor_loss += local_actor_loss.item()
        
        return total_actor_loss
    
    def _extract_high_action(self, actions) -> torch.Tensor:
        """从动作列表/字典提取高层动作张量
        
        修复：正确处理batch采样返回的动作列表
        """
        # 批处理：actions是列表
        if isinstance(actions, list):
            batch_actions = []
            for action in actions:
                if isinstance(action, dict) and 'raw_high_action' in action:
                    raw = action['raw_high_action']
                    if isinstance(raw, np.ndarray):
                        batch_actions.append(raw.flatten())
                    else:
                        batch_actions.append(np.array(raw).flatten())
                else:
                    batch_actions.append(np.zeros(4))
            return torch.FloatTensor(np.stack(batch_actions)).to(self.device)
        
        # 单个动作：actions是字典
        elif isinstance(actions, dict) and 'raw_high_action' in actions:
            raw = actions['raw_high_action']
            if isinstance(raw, np.ndarray):
                return torch.FloatTensor(raw.flatten()).unsqueeze(0).to(self.device)
            return torch.FloatTensor(np.array(raw).flatten()).unsqueeze(0).to(self.device)
        
        return torch.zeros(1, 4, device=self.device)
    
    def _extract_v2i_action(self, actions) -> torch.Tensor:
        """提取V2I动作（从raw_low_action）
        
        修复：正确处理batch采样返回的动作列表
        """
        action_dim = self.system_config.NUM_RSU + 1
        
        # 批处理：actions是列表
        if isinstance(actions, list):
            batch_actions = []
            for action in actions:
                if isinstance(action, dict) and 'raw_low_action' in action:
                    raw = action['raw_low_action']
                    if isinstance(raw, np.ndarray):
                        arr = raw.flatten()
                    else:
                        arr = np.array(raw).flatten()
                    # 确保维度正确
                    if len(arr) >= action_dim:
                        batch_actions.append(arr[:action_dim])
                    else:
                        padded = np.zeros(action_dim)
                        padded[:len(arr)] = arr
                        batch_actions.append(padded)
                else:
                    batch_actions.append(np.zeros(action_dim))
            return torch.FloatTensor(np.stack(batch_actions)).to(self.device)
        
        # 单个动作
        elif isinstance(actions, dict) and 'raw_low_action' in actions:
            raw = actions['raw_low_action']
            if isinstance(raw, np.ndarray):
                arr = raw.flatten()
            else:
                arr = np.array(raw).flatten()
            if len(arr) >= action_dim:
                return torch.FloatTensor(arr[:action_dim]).unsqueeze(0).to(self.device)
            padded = np.zeros(action_dim)
            padded[:len(arr)] = arr
            return torch.FloatTensor(padded).unsqueeze(0).to(self.device)
        
        return torch.zeros(1, action_dim, device=self.device)
    
    def _extract_v2v_action(self, actions) -> torch.Tensor:
        """提取V2V动作（从raw_low_action）
        
        修复：正确处理batch采样返回的动作列表
        """
        action_dim = 6  # 5个邻车logit + 1个power
        
        # 批处理：actions是列表
        if isinstance(actions, list):
            batch_actions = []
            for action in actions:
                if isinstance(action, dict) and 'raw_low_action' in action:
                    raw = action['raw_low_action']
                    if isinstance(raw, np.ndarray):
                        arr = raw.flatten()
                    else:
                        arr = np.array(raw).flatten()
                    # 确保维度正确
                    if len(arr) >= action_dim:
                        batch_actions.append(arr[:action_dim])
                    else:
                        padded = np.zeros(action_dim)
                        padded[:len(arr)] = arr
                        batch_actions.append(padded)
                else:
                    batch_actions.append(np.zeros(action_dim))
            return torch.FloatTensor(np.stack(batch_actions)).to(self.device)
        
        # 单个动作
        elif isinstance(actions, dict) and 'raw_low_action' in actions:
            raw = actions['raw_low_action']
            if isinstance(raw, np.ndarray):
                arr = raw.flatten()
            else:
                arr = np.array(raw).flatten()
            if len(arr) >= action_dim:
                return torch.FloatTensor(arr[:action_dim]).unsqueeze(0).to(self.device)
            padded = np.zeros(action_dim)
            padded[:len(arr)] = arr
            return torch.FloatTensor(padded).unsqueeze(0).to(self.device)
        
        return torch.zeros(1, action_dim, device=self.device)
    
    def _extract_local_action(self, actions) -> torch.Tensor:
        """提取Local动作（freq）
        
        修复：正确处理batch采样返回的动作列表
        """
        # 批处理：actions是列表
        if isinstance(actions, list):
            batch_actions = []
            for action in actions:
                if isinstance(action, dict) and 'freq' in action:
                    freq = action['freq']
                    if isinstance(freq, (int, float)):
                        batch_actions.append([float(freq)])
                    elif isinstance(freq, np.ndarray):
                        batch_actions.append([float(freq.flatten()[0])])
                    else:
                        batch_actions.append([float(freq)])
                else:
                    batch_actions.append([0.0])
            return torch.FloatTensor(batch_actions).to(self.device)
        
        # 单个动作
        elif isinstance(actions, dict) and 'freq' in actions:
            freq = actions['freq']
            if isinstance(freq, (int, float)):
                return torch.FloatTensor([[float(freq)]]).to(self.device)
            return torch.FloatTensor([[float(freq)]]).to(self.device)
        
        return torch.zeros(1, 1, device=self.device)
    
    def _update_learning_rate(self, episode: int):
        """学习率衰减"""
        start = self.config.TRAINING_CONFIG['lr_decay_start']
        duration = self.config.TRAINING_CONFIG['lr_decay_episodes']
        min_ratio = self.config.TRAINING_CONFIG['lr_min_ratio']
        
        if episode < start:
            return
        
        progress = min(1.0, (episode - start) / duration)
        ratio = 1.0 - (1.0 - min_ratio) * progress
        
        for name, optimizer in self.optimizers.items():
            base_lr = self.config.HIGH_LEVEL_CONFIG['learning_rate']
            if 'v2i' in name:
                base_lr = self.config.LOW_LEVEL_CONFIG['v2i_scheduler']['learning_rate']
            elif 'v2v' in name:
                base_lr = self.config.LOW_LEVEL_CONFIG['v2v_scheduler']['learning_rate']
            elif 'local' in name:
                base_lr = self.config.LOW_LEVEL_CONFIG['local_computing']['learning_rate']
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * ratio
    
    def update_noise_controller(self, episode: int, is_success: bool):
        """更新噪声控制器状态"""
        self.noise_controller.update_training_stats(episode, is_success)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'networks_state_dict': self.networks.state_dict(),
            'optimizers_state_dict': {k: v.state_dict() for k, v in self.optimizers.items()},
            'training_stats': self.training_stats,
            'noise_controller_stats': self.noise_controller.get_stats(),
            'update_count': self.update_count,
            'total_steps': self.total_steps,
        }, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.networks.load_state_dict(checkpoint['networks_state_dict'])
        
        for name, optimizer in self.optimizers.items():
            if name in checkpoint['optimizers_state_dict']:
                optimizer.load_state_dict(checkpoint['optimizers_state_dict'][name])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.update_count = checkpoint.get('update_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        print(f"模型已加载: {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        noise_stats = self.noise_controller.get_stats()
        
        return {
            'episode_rewards': self.training_stats['episode_rewards'],
            'episode_lengths': self.training_stats['episode_lengths'],
            'avg_actor_loss': np.mean(self.training_stats['actor_losses'][-100:]) if self.training_stats['actor_losses'] else 0,
            'avg_critic_loss': np.mean(self.training_stats['critic_losses'][-100:]) if self.training_stats['critic_losses'] else 0,
            'noise_phase': noise_stats['phase'],
            'noise_scale': noise_stats['global_noise'],
            'success_rate': noise_stats['success_rate'],
            'buffer_stats': self.replay_buffer.get_stats(),
        }

