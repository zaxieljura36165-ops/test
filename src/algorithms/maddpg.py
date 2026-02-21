"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 算法实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from src.algorithms.maddpg_networks import MADDPGActor, MADDPGCritic, soft_update, hard_update
from src.algorithms.replay_buffer import MAReplayBuffer


class MADDPG:
    """MADDPG智能体"""
    
    def __init__(self, config: Dict, num_agents: int, state_dim: int, 
                 global_state_dim: int, device: str = 'cuda'):
        """
        Args:
            config: MADDPG配置
            num_agents: 智能体数量
            state_dim: 局部状态维度
            global_state_dim: 全局状态维度
            device: 设备
        """
        self.config = config
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 动作维度
        self.action_dims = {
            'alpha': 1,
            'mode': 2,
            'target': 6,
            'power': 1,
            'frequency': 1
        }
        
        # 为每个智能体创建Actor网络（本地）
        self.actors = []
        self.target_actors = []
        self.actor_optimizers = []
        
        for _ in range(num_agents):
            actor = MADDPGActor(
                state_dim, 
                hidden_sizes=config.get('actor_hidden_sizes', [256, 128])
            ).to(self.device)
            
            target_actor = MADDPGActor(
                state_dim,
                hidden_sizes=config.get('actor_hidden_sizes', [256, 128])
            ).to(self.device)
            
            hard_update(actor, target_actor)
            
            optimizer = optim.Adam(actor.parameters(), 
                                  lr=config.get('actor_lr', 1e-4))
            
            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.actor_optimizers.append(optimizer)
        
        # 为每个智能体创建Critic网络（集中式）
        self.critics = []
        self.target_critics = []
        self.critic_optimizers = []
        
        for _ in range(num_agents):
            critic = MADDPGCritic(
                global_state_dim,
                num_agents,
                self.action_dims,
                hidden_sizes=config.get('critic_hidden_sizes', [512, 256, 128])
            ).to(self.device)
            
            target_critic = MADDPGCritic(
                global_state_dim,
                num_agents,
                self.action_dims,
                hidden_sizes=config.get('critic_hidden_sizes', [512, 256, 128])
            ).to(self.device)
            
            hard_update(critic, target_critic)
            
            optimizer = optim.Adam(critic.parameters(),
                                  lr=config.get('critic_lr', 1e-3))
            
            self.critics.append(critic)
            self.target_critics.append(target_critic)
            self.critic_optimizers.append(optimizer)
        
        # 经验回放缓冲区
        self.replay_buffer = MAReplayBuffer(
            buffer_size=config.get('buffer_size', 100000),
            num_agents=num_agents
        )
        
        # 训练参数
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.batch_size = config.get('batch_size', 128)
        self.noise_scale = config.get('noise_scale', 0.1)
        self.noise_decay = config.get('noise_decay', 0.9999)
        self.min_noise_scale = config.get('min_noise_scale', 0.01)
        self.warmup_steps = config.get('warmup_steps', 1000)
        
        self.step_count = 0
        self.update_count = 0
    
    def select_action(self, state: torch.Tensor, agent_id: int, 
                     add_noise: bool = True) -> Dict[str, Any]:
        """选择动作
        
        Args:
            state: 局部状态 [state_dim]
            agent_id: 智能体ID
            add_noise: 是否添加探索噪声
        
        Returns:
            action_dict: 动作字典
        """
        self.actors[agent_id].eval()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)  # [1, state_dim]
            
            # 根据训练阶段调整温度
            temperature = 1.0 if self.step_count < self.warmup_steps else 0.5
            
            actions = self.actors[agent_id](state, temperature=temperature, hard=False)
            
            # 添加探索噪声
            if add_noise:
                noise_scale = max(self.noise_scale * (self.noise_decay ** self.step_count),
                                self.min_noise_scale)
                
                actions['alpha'] += torch.randn_like(actions['alpha']) * noise_scale
                actions['power'] += torch.randn_like(actions['power']) * noise_scale
                actions['frequency'] += torch.randn_like(actions['frequency']) * noise_scale
                
                # 裁剪到[0, 1]
                actions['alpha'] = torch.clamp(actions['alpha'], 0.0, 1.0)
                actions['power'] = torch.clamp(actions['power'], 0.0, 1.0)
                actions['frequency'] = torch.clamp(actions['frequency'], 0.0, 1.0)
        
        self.actors[agent_id].train()
        self.step_count += 1
        
        # 转换为环境可用的格式
        return self._convert_action_to_env_format(actions)
    
    def _convert_action_to_env_format(self, actions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """将网络输出转换为环境格式"""
        mode_probs = actions['mode'].cpu().numpy()[0]
        target_probs = actions['target'].cpu().numpy()[0]
        
        mode_idx = np.argmax(mode_probs)
        target_idx = np.argmax(target_probs)
        
        mode_map = ['V2I', 'V2V']
        action = {
            'alpha': actions['alpha'].item(),
            'mode': mode_map[mode_idx],
            'mode_idx': mode_idx,
            'target': target_idx,  # 保留用于训练
            'power': actions['power'].item(),
            'frequency': actions['frequency'].item(),
            'freq': actions['frequency'].item(),  # 环境使用freq字段
            'rsu_action': 0,
            'neighbor_action': 0
        }
        
        # 根据模式映射到环境所需字段
        if mode_idx == 0:  # V2I
            action['rsu_action'] = target_idx
        else:  # V2V
            action['neighbor_action'] = target_idx
        
        return action
    
    def store_experience(self, experience: Dict[str, Any]):
        """存储经验"""
        self.replay_buffer.add(experience)
    
    def update(self, episode: int) -> Dict[str, float]:
        """更新网络
        
        Returns:
            losses: {'actor_loss': float, 'critic_loss': float}
        """
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # 采样批数据
        batch = self.replay_buffer.sample(self.batch_size)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        # 更新每个智能体
        for agent_id in range(self.num_agents):
            actor_loss, critic_loss = self._update_agent(agent_id, batch)
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
        
        # 软更新目标网络
        for agent_id in range(self.num_agents):
            soft_update(self.actors[agent_id], self.target_actors[agent_id], self.tau)
            soft_update(self.critics[agent_id], self.target_critics[agent_id], self.tau)
        
        self.update_count += 1
        
        return {
            'actor_loss': total_actor_loss / self.num_agents,
            'critic_loss': total_critic_loss / self.num_agents
        }
    
    def _update_agent(self, agent_id: int, batch: Dict[str, Any]) -> Tuple[float, float]:
        """更新单个智能体"""
        states = batch['states'][agent_id].to(self.device)
        global_states = batch['global_states'].to(self.device)
        next_states = batch['next_states'][agent_id].to(self.device)
        next_global_states = batch['next_global_states'].to(self.device)
        rewards = batch['rewards'][agent_id].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 当前动作（从batch中提取）
        current_actions = {}
        for aid in range(self.num_agents):
            current_actions[aid] = {
                'alpha': batch['actions'][aid]['alpha'].unsqueeze(-1).to(self.device),
                'mode': F.one_hot(batch['actions'][aid]['mode'].long(), 2).float().to(self.device),
                'target': F.one_hot(batch['actions'][aid]['target'].long(), 6).float().to(self.device),
                'power': batch['actions'][aid]['power'].unsqueeze(-1).to(self.device),
                'frequency': batch['actions'][aid]['frequency'].unsqueeze(-1).to(self.device)
            }
        
        # ==================== 更新Critic ====================
        with torch.no_grad():
            # 获取下一状态的目标动作
            next_target_actions = {}
            for aid in range(self.num_agents):
                next_target_actions[aid] = self.target_actors[aid](
                    next_states if aid == agent_id else batch['next_states'][aid].to(self.device),
                    temperature=0.5,
                    hard=True
                )
            
            # 计算目标Q值
            target_q = self.target_critics[agent_id](next_global_states, next_target_actions)
            target_q = rewards.unsqueeze(-1) + self.gamma * target_q * (1 - dones.unsqueeze(-1))
        
        # 当前Q值
        current_q = self.critics[agent_id](global_states, current_actions)
        
        # Critic损失（MSE）
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # 更新Critic
        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 10.0)
        self.critic_optimizers[agent_id].step()
        
        # ==================== 更新Actor ====================
        # 获取当前策略的动作
        policy_actions = {}
        for aid in range(self.num_agents):
            if aid == agent_id:
                policy_actions[aid] = self.actors[aid](states, temperature=0.5, hard=False)
            else:
                with torch.no_grad():
                    policy_actions[aid] = self.actors[aid](
                        batch['states'][aid].to(self.device),
                        temperature=0.5,
                        hard=False
                    )
        
        # Actor损失（最大化Q值）
        actor_loss = -self.critics[agent_id](global_states, policy_actions).mean()
        
        # 更新Actor
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 10.0)
        self.actor_optimizers[agent_id].step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save_model(self, path: str, episode: int):
        """保存模型"""
        checkpoint = {
            'episode': episode,
            'update_count': self.update_count,
            'step_count': self.step_count,
            'noise_scale': self.noise_scale
        }
        
        for agent_id in range(self.num_agents):
            checkpoint[f'actor_{agent_id}'] = self.actors[agent_id].state_dict()
            checkpoint[f'critic_{agent_id}'] = self.critics[agent_id].state_dict()
            checkpoint[f'actor_optimizer_{agent_id}'] = self.actor_optimizers[agent_id].state_dict()
            checkpoint[f'critic_optimizer_{agent_id}'] = self.critic_optimizers[agent_id].state_dict()
        
        torch.save(checkpoint, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for agent_id in range(self.num_agents):
            self.actors[agent_id].load_state_dict(checkpoint[f'actor_{agent_id}'])
            self.critics[agent_id].load_state_dict(checkpoint[f'critic_{agent_id}'])
            self.actor_optimizers[agent_id].load_state_dict(checkpoint[f'actor_optimizer_{agent_id}'])
            self.critic_optimizers[agent_id].load_state_dict(checkpoint[f'critic_optimizer_{agent_id}'])
            
            hard_update(self.actors[agent_id], self.target_actors[agent_id])
            hard_update(self.critics[agent_id], self.target_critics[agent_id])
        
        self.update_count = checkpoint.get('update_count', 0)
        self.step_count = checkpoint.get('step_count', 0)
        self.noise_scale = checkpoint.get('noise_scale', self.noise_scale)
        
        print(f"模型已加载: {path}")


# 导入F (忘记了)
import torch.nn.functional as F

