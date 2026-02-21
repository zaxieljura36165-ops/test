"""
SAC network definitions (actor and critic).
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.algorithms.td3_networks import LayerNormMLP


class SACActor(nn.Module):
    """SAC actor network (Gaussian policy over raw actions)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list,
        use_layer_norm: bool = True,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.feature_net = LayerNormMLP(
            state_dim, hidden_sizes[-1], hidden_sizes[:-1], use_layer_norm=use_layer_norm
        )
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

        self._init_final_layer()

    def _init_final_layer(self):
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample raw action with reparameterization."""
        mean, log_std = self.forward(state)
        if deterministic:
            action = mean
            log_prob = None
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            action = normal.rsample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class SACCritic(nn.Module):
    """SAC critic with twin Q networks."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list, use_layer_norm: bool = True):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1_net = LayerNormMLP(input_dim, 1, hidden_sizes, use_layer_norm=use_layer_norm)
        self.q2_net = LayerNormMLP(input_dim, 1, hidden_sizes, use_layer_norm=use_layer_norm)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)
