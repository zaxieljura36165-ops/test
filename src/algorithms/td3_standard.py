"""
标准TD3算法（非分层，多智能体）
每个车辆一套 Actor/Critic，可使用集中式 Critic（全局状态 + 联合动作）
"""

import copy
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from configs.system_config import SystemConfig
from configs.td3_config import TD3Config
from src.algorithms.td3_networks import TD3Actor, TD3Critic
from src.algorithms.noise_controller import DualReplayBuffer


class StandardTD3:
    """标准TD3（非分层，多智能体）"""

    def __init__(self, config: TD3Config, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.device = torch.device(config.DEVICE)

        # 基础维度
        self.num_agents = self.system_config.NUM_VEHICLES
        self.num_rsu = self.system_config.NUM_RSU
        self.max_neighbors = 5
        self.state_dim = self.config.HIGH_LEVEL_CONFIG['state_dim']
        self.global_state_dim = self.state_dim * self.num_agents

        # 动作维度：alpha(1) + mode(2) + rsu_logits + neighbor_logits + power(1) + freq(1)
        self.action_dim = 1 + 2 + self.num_rsu + self.max_neighbors + 1 + 1
        self.total_action_dim = self.action_dim * self.num_agents

        # Actor/Critic（每个车辆一套）
        hidden_sizes = self.config.HIGH_LEVEL_CONFIG['hidden_sizes']
        critic_hidden = hidden_sizes[:2] if len(hidden_sizes) > 2 else hidden_sizes

        self.actors = []
        self.actor_targets = []
        self.actor_opts = []

        self.critics = []
        self.critic_targets = []
        self.critic_opts = []

        lr = self.config.HIGH_LEVEL_CONFIG['learning_rate']

        for _ in range(self.num_agents):
            actor = TD3Actor(self.state_dim, self.action_dim, hidden_sizes, use_layer_norm=True).to(self.device)
            actor_target = copy.deepcopy(actor)
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.actor_opts.append(optim.Adam(actor.parameters(), lr=lr))

            critic = TD3Critic(self.global_state_dim, self.total_action_dim, critic_hidden, use_layer_norm=True).to(self.device)
            critic_target = copy.deepcopy(critic)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_opts.append(optim.Adam(critic.parameters(), lr=lr))

        # Replay buffers（每个车辆一套）
        self.replay_buffers = [
            DualReplayBuffer(
                capacity=self.config.BUFFER_CONFIG['buffer_size'],
                state_dim=self.state_dim,
                action_dim=self.action_dim
            ) for _ in range(self.num_agents)
        ]

        self.update_counts = [0 for _ in range(self.num_agents)]

    def select_action(self, local_state: torch.Tensor, vehicle_id: int,
                      deterministic: bool = False, noise_std: float = 0.1) -> Dict[str, Any]:
        """选择动作"""
        actor = self.actors[vehicle_id]
        if deterministic:
            actor.eval()
        else:
            actor.train()

        with torch.no_grad():
            if local_state.dim() == 1:
                local_state = local_state.unsqueeze(0)
            local_state = local_state.to(self.device)

            raw_action = actor(local_state)

            # 标准TD3探索噪声：对原始动作向量加高斯噪声
            if not deterministic and noise_std > 0:
                noise = torch.randn_like(raw_action) * noise_std
                raw_action = raw_action + noise

            action = self._process_raw_action(raw_action)
            return action

    def _process_raw_action(self, raw_action: torch.Tensor) -> Dict[str, Any]:
        """将raw_action映射为环境动作"""
        raw = raw_action.squeeze(0).cpu().numpy()

        idx = 0
        alpha = 1.0 / (1.0 + np.exp(-raw[idx]))
        idx += 1

        mode_logits = raw[idx:idx + 2]
        idx += 2

        rsu_logits = raw[idx:idx + self.num_rsu]
        idx += self.num_rsu

        neighbor_logits = raw[idx:idx + self.max_neighbors]
        idx += self.max_neighbors

        power = 1.0 / (1.0 + np.exp(-raw[idx]))
        idx += 1

        freq = 1.0 / (1.0 + np.exp(-raw[idx]))

        mode_idx = int(np.argmax(mode_logits))
        rsu_idx = int(np.argmax(rsu_logits)) if self.num_rsu > 0 else 0
        neighbor_idx = int(np.argmax(neighbor_logits)) if self.max_neighbors > 0 else 0

        # 本地阈值逻辑
        if alpha < 0.01:
            mode = 'local'
            mode_idx = -1
            power = 0.0
            rsu_idx = 0
            neighbor_idx = 0
        else:
            mode = 'V2I' if mode_idx == 0 else 'V2V'

        # 如果几乎全卸载，关闭本地频率
        if alpha >= 0.99:
            freq = 0.0

        action = {
            'alpha': float(alpha),
            'mode': mode,
            'mode_idx': mode_idx,
            'rsu_action': rsu_idx,
            'neighbor_action': neighbor_idx,
            'power': float(power),
            'freq': float(freq),
            'raw_action': raw
        }

        return action

    def store_experience(self, experience: Dict[str, Any], agent_id: int, is_noisy: bool = True):
        """存储经验"""
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x

        self.replay_buffers[agent_id].push(
            state=to_numpy(experience['local_state']),
            action=experience['action'],
            reward=experience['reward'],
            next_state=to_numpy(experience['next_local_state']),
            done=experience['done'],
            global_state=to_numpy(experience.get('global_state')),
            next_global_state=to_numpy(experience.get('next_global_state')),
            is_noisy=is_noisy
        )

    def update(self, episode: int):
        """更新网络参数"""
        gamma = self.config.TD3_CONFIG['gamma']
        policy_noise = self.config.TD3_CONFIG['policy_noise']
        noise_clip = self.config.TD3_CONFIG['noise_clip']
        tau = self.config.TD3_CONFIG['tau']

        for agent_id in range(self.num_agents):
            buffer = self.replay_buffers[agent_id]
            if len(buffer) < self.config.BUFFER_CONFIG['min_buffer_size']:
                continue

            batch = buffer.sample(
                self.config.BUFFER_CONFIG['batch_size'],
                self.device,
                deterministic_ratio=0.0
            )
            if batch is None:
                continue

            actions = batch['actions']
            rewards = batch['rewards']
            dones = batch['dones']

            global_states = batch.get('global_states')
            next_global_states = batch.get('next_global_states')
            if global_states is None or next_global_states is None:
                continue

            # === 构建当前联合动作（来自经验）===
            joint_raw = self._extract_joint_raw_actions(actions)
            joint_action = self._raw_to_action_vec(joint_raw.view(-1, self.action_dim))
            joint_action = joint_action.view(joint_raw.size(0), self.total_action_dim)

            # === 构建下一步联合动作（目标网络）===
            next_states_by_agent = self._extract_all_states(actions, key='all_next_local_states')
            next_action_vecs = []
            with torch.no_grad():
                for aid in range(self.num_agents):
                    raw = self.actor_targets[aid](next_states_by_agent[aid])
                    noise = torch.clamp(torch.randn_like(raw) * policy_noise, -noise_clip, noise_clip)
                    raw = raw + noise
                    vec = self._raw_to_action_vec(raw)
                    next_action_vecs.append(vec)
                next_joint_action = torch.cat(next_action_vecs, dim=-1)

                target_q1, target_q2 = self.critic_targets[agent_id](next_global_states, next_joint_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + gamma * (1 - dones) * target_q

            # === 更新Critic ===
            current_q1, current_q2 = self.critics[agent_id](global_states, joint_action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_opts[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(),
                                           self.config.TRAINING_CONFIG['max_grad_norm'])
            self.critic_opts[agent_id].step()

            # === 更新Actor（延迟）===
            self.update_counts[agent_id] += 1
            if self.update_counts[agent_id] % self.config.TD3_CONFIG['policy_delay'] == 0:
                states_by_agent = self._extract_all_states(actions, key='all_local_states')
                action_vecs = []
                for aid in range(self.num_agents):
                    if aid == agent_id:
                        raw = self.actors[aid](states_by_agent[aid])
                    else:
                        with torch.no_grad():
                            raw = self.actors[aid](states_by_agent[aid])
                    vec = self._raw_to_action_vec(raw)
                    action_vecs.append(vec)
                joint_policy_action = torch.cat(action_vecs, dim=-1)

                actor_loss = -self.critics[agent_id].q1(global_states, joint_policy_action).mean()
                self.actor_opts[agent_id].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(),
                                               self.config.TRAINING_CONFIG['max_grad_norm'])
                self.actor_opts[agent_id].step()

                # Soft update
                self._soft_update(self.actors[agent_id], self.actor_targets[agent_id], tau)
                self._soft_update(self.critics[agent_id], self.critic_targets[agent_id], tau)

    def _extract_joint_raw_actions(self, actions) -> torch.Tensor:
        """从经验动作中提取联合raw_action张量 [B, N, A]"""
        batch_actions = []
        for action in actions:
            raw_list = action.get('joint_raw_actions')
            if raw_list is None:
                raw = action.get('raw_action', np.zeros(self.action_dim))
                raw_list = [raw for _ in range(self.num_agents)]
            batch_actions.append(np.stack([np.array(r).flatten() for r in raw_list]))
        return torch.FloatTensor(np.stack(batch_actions)).to(self.device)

    def _extract_all_states(self, actions, key: str) -> List[torch.Tensor]:
        """提取所有车辆的局部状态/下一状态"""
        all_states = []
        for action in actions:
            states = action.get(key)
            if states is None:
                states = [np.zeros(self.state_dim) for _ in range(self.num_agents)]
            all_states.append(states)

        states_by_agent = []
        for aid in range(self.num_agents):
            states_by_agent.append(
                torch.FloatTensor(np.stack([np.array(s[aid]).flatten() for s in all_states])).to(self.device)
            )
        return states_by_agent

    def _raw_to_action_vec(self, raw_action: torch.Tensor) -> torch.Tensor:
        """将raw_action映射为连续动作向量（用于Critic）"""
        idx = 0
        alpha = torch.sigmoid(raw_action[:, idx:idx + 1])
        idx += 1

        mode_logits = raw_action[:, idx:idx + 2]
        mode_probs = F.softmax(mode_logits, dim=-1)
        idx += 2

        if self.num_rsu > 0:
            rsu_logits = raw_action[:, idx:idx + self.num_rsu]
            rsu_probs = F.softmax(rsu_logits, dim=-1)
        else:
            rsu_probs = torch.zeros(raw_action.size(0), 0, device=raw_action.device)
        idx += self.num_rsu

        if self.max_neighbors > 0:
            neighbor_logits = raw_action[:, idx:idx + self.max_neighbors]
            neighbor_probs = F.softmax(neighbor_logits, dim=-1)
        else:
            neighbor_probs = torch.zeros(raw_action.size(0), 0, device=raw_action.device)
        idx += self.max_neighbors

        power = torch.sigmoid(raw_action[:, idx:idx + 1])
        idx += 1
        freq = torch.sigmoid(raw_action[:, idx:idx + 1])

        return torch.cat([alpha, mode_probs, rsu_probs, neighbor_probs, power, freq], dim=-1)

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, path: str):
        """保存模型"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_targets': [actor.state_dict() for actor in self.actor_targets],
            'critic_targets': [critic.state_dict() for critic in self.critic_targets],
            'actor_opts': [opt.state_dict() for opt in self.actor_opts],
            'critic_opts': [opt.state_dict() for opt in self.critic_opts],
            'update_counts': self.update_counts
        }
        torch.save(checkpoint, path)
        print(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
            self.actor_targets[i].load_state_dict(checkpoint['actor_targets'][i])
            self.critic_targets[i].load_state_dict(checkpoint['critic_targets'][i])
            self.actor_opts[i].load_state_dict(checkpoint['actor_opts'][i])
            self.critic_opts[i].load_state_dict(checkpoint['critic_opts'][i])
        self.update_counts = checkpoint.get('update_counts', self.update_counts)
        print(f"模型已加载: {path}")

    def set_eval(self):
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()   
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()