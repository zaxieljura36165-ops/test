"""
Standard SAC (non-hierarchical, multi-agent with centralized critics).
"""

import copy
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from configs.system_config import SystemConfig
from configs.sac_config import SACConfig
from src.algorithms.sac_networks import SACActor, SACCritic
from src.algorithms.noise_controller import DualReplayBuffer


class StandardSAC:
    """Standard SAC (non-hierarchical, multi-agent)."""

    def __init__(self, config: SACConfig, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.device = torch.device(config.DEVICE)

        # Basic dimensions
        self.num_agents = self.system_config.NUM_VEHICLES
        self.num_rsu = self.system_config.NUM_RSU
        self.max_neighbors = 5
        self.state_dim = self.config.HIGH_LEVEL_CONFIG["state_dim"]
        self.global_state_dim = self.state_dim * self.num_agents

        # Raw action dim: alpha + mode(2) + rsu_logits + neighbor_logits + power + freq
        self.action_dim = 1 + 2 + self.num_rsu + self.max_neighbors + 1 + 1
        self.total_action_dim = self.action_dim * self.num_agents

        hidden_sizes = self.config.HIGH_LEVEL_CONFIG["hidden_sizes"]
        critic_hidden = hidden_sizes[:2] if len(hidden_sizes) > 2 else hidden_sizes
        lr = self.config.HIGH_LEVEL_CONFIG["learning_rate"]
        use_layer_norm = self.config.HIGH_LEVEL_CONFIG.get("use_layer_norm", True)

        # Actor/Critic per agent
        self.actors = []
        self.actor_opts = []
        self.critics = []
        self.critic_targets = []
        self.critic_opts = []

        log_std_min = self.config.SAC_CONFIG.get("log_std_min", -20)
        log_std_max = self.config.SAC_CONFIG.get("log_std_max", 2)

        for _ in range(self.num_agents):
            actor = SACActor(
                self.state_dim,
                self.action_dim,
                hidden_sizes,
                use_layer_norm=use_layer_norm,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
            ).to(self.device)
            self.actors.append(actor)
            self.actor_opts.append(optim.Adam(actor.parameters(), lr=lr))

            critic = SACCritic(
                self.global_state_dim,
                self.total_action_dim,
                critic_hidden,
                use_layer_norm=use_layer_norm,
            ).to(self.device)
            critic_target = copy.deepcopy(critic).to(self.device)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.critic_opts.append(optim.Adam(critic.parameters(), lr=lr))

        # Replay buffers (per agent)
        self.replay_buffers = [
            DualReplayBuffer(
                capacity=self.config.BUFFER_CONFIG["buffer_size"],
                state_dim=self.state_dim,
                action_dim=self.action_dim,
            )
            for _ in range(self.num_agents)
        ]

        # SAC params
        self.gamma = self.config.SAC_CONFIG.get("gamma", 0.99)
        self.tau = self.config.SAC_CONFIG.get("tau", 0.005)
        self.alpha = float(self.config.SAC_CONFIG.get("alpha", 0.2))
        self.auto_alpha = bool(self.config.SAC_CONFIG.get("auto_alpha", False))
        self.target_entropy = self.config.SAC_CONFIG.get("target_entropy", None)
        self.alpha_lr = self.config.SAC_CONFIG.get("alpha_lr", lr)

        if self.target_entropy is None:
            self.target_entropy = -float(self.action_dim)

        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(self.alpha), dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_opt = None

    def _get_alpha_tensor(self) -> torch.Tensor:
        if self.auto_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha, dtype=torch.float32, device=self.device)

    def get_alpha_value(self) -> float:
        if self.auto_alpha:
            return float(self.log_alpha.exp().detach().cpu().item())
        return float(self.alpha)

    def select_action(self, local_state: torch.Tensor, vehicle_id: int, deterministic: bool = False) -> Dict[str, Any]:
        actor = self.actors[vehicle_id]
        if deterministic:
            actor.eval()
        else:
            actor.train()

        with torch.no_grad():
            if local_state.dim() == 1:
                local_state = local_state.unsqueeze(0)
            local_state = local_state.to(self.device)
            raw_action, _, _ = actor.sample(local_state, deterministic=deterministic)
            action = self._process_raw_action(raw_action)
        return action

    def _process_raw_action(self, raw_action: torch.Tensor) -> Dict[str, Any]:
        raw = raw_action.squeeze(0).cpu().numpy()
        idx = 0
        alpha = 1.0 / (1.0 + np.exp(-raw[idx]))
        idx += 1

        mode_logits = raw[idx : idx + 2]
        idx += 2

        rsu_logits = raw[idx : idx + self.num_rsu]
        idx += self.num_rsu

        neighbor_logits = raw[idx : idx + self.max_neighbors]
        idx += self.max_neighbors

        power = 1.0 / (1.0 + np.exp(-raw[idx]))
        idx += 1

        freq = 1.0 / (1.0 + np.exp(-raw[idx]))

        mode_idx = int(np.argmax(mode_logits))
        rsu_idx = int(np.argmax(rsu_logits)) if self.num_rsu > 0 else 0
        neighbor_idx = int(np.argmax(neighbor_logits)) if self.max_neighbors > 0 else 0

        # Local threshold logic
        if alpha < 0.01:
            mode = "local"
            mode_idx = -1
            power = 0.0
            rsu_idx = 0
            neighbor_idx = 0
        else:
            mode = "V2I" if mode_idx == 0 else "V2V"

        # If almost full offload, disable local frequency
        if alpha >= 0.99:
            freq = 0.0

        return {
            "alpha": float(alpha),
            "mode": mode,
            "mode_idx": mode_idx,
            "rsu_action": rsu_idx,
            "neighbor_action": neighbor_idx,
            "power": float(power),
            "freq": float(freq),
            "raw_action": raw,
        }

    def store_experience(self, experience: Dict[str, Any], agent_id: int, is_noisy: bool = True):
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x

        self.replay_buffers[agent_id].push(
            state=to_numpy(experience["local_state"]),
            action=experience["action"],
            reward=experience["reward"],
            next_state=to_numpy(experience["next_local_state"]),
            done=experience["done"],
            global_state=to_numpy(experience.get("global_state")),
            next_global_state=to_numpy(experience.get("next_global_state")),
            is_noisy=is_noisy,
        )

    def update(self, episode: int) -> Dict[str, float]:
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        max_grad_norm = self.config.TRAINING_CONFIG["max_grad_norm"]

        for agent_id in range(self.num_agents):
            buffer = self.replay_buffers[agent_id]
            if len(buffer) < self.config.BUFFER_CONFIG["min_buffer_size"]:
                continue

            batch = buffer.sample(
                self.config.BUFFER_CONFIG["batch_size"], self.device, deterministic_ratio=0.0
            )
            if batch is None:
                continue

            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]

            global_states = batch.get("global_states")
            next_global_states = batch.get("next_global_states")
            if global_states is None or next_global_states is None:
                continue

            # Current joint action (from experience)
            joint_raw = self._extract_joint_raw_actions(actions)
            joint_action = self._raw_to_action_vec(joint_raw.view(-1, self.action_dim))
            joint_action = joint_action.view(joint_raw.size(0), self.total_action_dim)

            # Next joint action (from current policy)
            next_states_by_agent = self._extract_all_states(actions, key="all_next_local_states")
            next_action_vecs = []
            next_log_probs = []
            with torch.no_grad():
                for aid in range(self.num_agents):
                    raw, logp, _ = self.actors[aid].sample(next_states_by_agent[aid], deterministic=False)
                    vec = self._raw_to_action_vec(raw)
                    next_action_vecs.append(vec)
                    if logp is None:
                        logp = torch.zeros((raw.size(0), 1), device=self.device)
                    next_log_probs.append(logp)

                next_joint_action = torch.cat(next_action_vecs, dim=-1)
                target_q1, target_q2 = self.critic_targets[agent_id](
                    next_global_states, next_joint_action
                )
                target_q = torch.min(target_q1, target_q2)
                target_q = target_q - self._get_alpha_tensor() * next_log_probs[agent_id]
                target_q = rewards + self.gamma * (1 - dones) * target_q

            # Critic update
            current_q1, current_q2 = self.critics[agent_id](global_states, joint_action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_opts[agent_id].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), max_grad_norm)
            self.critic_opts[agent_id].step()

            total_critic_loss += float(critic_loss.item())

            # Actor update
            states_by_agent = self._extract_all_states(actions, key="all_local_states")
            action_vecs = []
            log_probs = []
            for aid in range(self.num_agents):
                if aid == agent_id:
                    raw, logp, _ = self.actors[aid].sample(states_by_agent[aid], deterministic=False)
                else:
                    with torch.no_grad():
                        raw, logp, _ = self.actors[aid].sample(
                            states_by_agent[aid], deterministic=False
                        )
                vec = self._raw_to_action_vec(raw)
                action_vecs.append(vec)
                if logp is None:
                    logp = torch.zeros((raw.size(0), 1), device=self.device)
                log_probs.append(logp)

            joint_policy_action = torch.cat(action_vecs, dim=-1)
            actor_loss = (
                self._get_alpha_tensor() * log_probs[agent_id]
                - self.critics[agent_id].q1(global_states, joint_policy_action)
            ).mean()

            self.actor_opts[agent_id].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), max_grad_norm)
            self.actor_opts[agent_id].step()

            total_actor_loss += float(actor_loss.item())

            # Alpha update (once per update call)
            if self.auto_alpha and agent_id == 0:
                stacked_log_probs = torch.stack(log_probs, dim=0).mean(dim=0)
                alpha_loss = -(
                    self.log_alpha * (stacked_log_probs + self.target_entropy).detach()
                ).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

            # Soft update target critics
            self._soft_update(self.critics[agent_id], self.critic_targets[agent_id], self.tau)

        return {
            "actor_loss": total_actor_loss / max(1, self.num_agents),
            "critic_loss": total_critic_loss / max(1, self.num_agents),
        }

    def _extract_joint_raw_actions(self, actions) -> torch.Tensor:
        """Extract joint raw_action tensor [B, N, A] from action dicts."""
        batch_actions = []
        for action in actions:
            raw_list = action.get("joint_raw_actions")
            if raw_list is None:
                raw = action.get("raw_action", np.zeros(self.action_dim))
                raw_list = [raw for _ in range(self.num_agents)]
            batch_actions.append(np.stack([np.array(r).flatten() for r in raw_list]))
        return torch.FloatTensor(np.stack(batch_actions)).to(self.device)

    def _extract_all_states(self, actions, key: str) -> List[torch.Tensor]:
        """Extract all agents' local states from action dicts."""
        all_states = []
        for action in actions:
            states = action.get(key)
            if states is None:
                states = [np.zeros(self.state_dim) for _ in range(self.num_agents)]
            all_states.append(states)

        states_by_agent = []
        for aid in range(self.num_agents):
            states_by_agent.append(
                torch.FloatTensor(
                    np.stack([np.array(s[aid]).flatten() for s in all_states])
                ).to(self.device)
            )
        return states_by_agent

    def _raw_to_action_vec(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map raw_action to continuous action vector for critics."""
        idx = 0
        alpha = torch.sigmoid(raw_action[:, idx : idx + 1])
        idx += 1

        mode_logits = raw_action[:, idx : idx + 2]
        mode_probs = F.softmax(mode_logits, dim=-1)
        idx += 2

        if self.num_rsu > 0:
            rsu_logits = raw_action[:, idx : idx + self.num_rsu]
            rsu_probs = F.softmax(rsu_logits, dim=-1)
        else:
            rsu_probs = torch.zeros(raw_action.size(0), 0, device=raw_action.device)
        idx += self.num_rsu

        if self.max_neighbors > 0:
            neighbor_logits = raw_action[:, idx : idx + self.max_neighbors]
            neighbor_probs = F.softmax(neighbor_logits, dim=-1)
        else:
            neighbor_probs = torch.zeros(raw_action.size(0), 0, device=raw_action.device)
        idx += self.max_neighbors

        power = torch.sigmoid(raw_action[:, idx : idx + 1])
        idx += 1
        freq = torch.sigmoid(raw_action[:, idx : idx + 1])

        return torch.cat([alpha, mode_probs, rsu_probs, neighbor_probs, power, freq], dim=-1)

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, path: str):
        checkpoint = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "critic_targets": [critic.state_dict() for critic in self.critic_targets],
            "actor_opts": [opt.state_dict() for opt in self.actor_opts],
            "critic_opts": [opt.state_dict() for opt in self.critic_opts],
            "alpha": self.get_alpha_value(),
        }
        if self.auto_alpha:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_opt"] = self.alpha_opt.state_dict()
        torch.save(checkpoint, path)
        print(f"Model saved: {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint["actors"][i])
            self.critics[i].load_state_dict(checkpoint["critics"][i])
            self.critic_targets[i].load_state_dict(checkpoint["critic_targets"][i])
            self.actor_opts[i].load_state_dict(checkpoint["actor_opts"][i])
            self.critic_opts[i].load_state_dict(checkpoint["critic_opts"][i])

        if self.auto_alpha and "log_alpha" in checkpoint:
            if self.log_alpha is None:
                self.log_alpha = torch.tensor(
                    np.log(self.alpha),
                    dtype=torch.float32,
                    requires_grad=True,
                    device=self.device,
                )
                self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            if self.alpha_opt is not None and "alpha_opt" in checkpoint:
                self.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
        elif "alpha" in checkpoint:
            self.alpha = float(checkpoint["alpha"])

        print(f"Model loaded: {path}")

    def set_eval(self):
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()

    def set_train(self):
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()
