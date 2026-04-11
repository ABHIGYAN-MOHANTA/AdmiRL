import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import MAX_QUEUE_JOBS, STATE_DIM


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = MAX_QUEUE_JOBS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, mask: torch.Tensor):
        features = self.shared(state)

        invalid_rows = mask.sum(dim=1) == 0
        if invalid_rows.any():
            mask = mask.clone()
            mask[invalid_rows] = 1.0

        logits = self.actor(features)
        logits = logits.masked_fill(mask == 0, float("-inf"))

        dist = Categorical(logits=logits)
        value = self.critic(features)
        return dist, value

    def critic_value(self, state: torch.Tensor):
        features = self.shared(state)
        value = self.critic(features)
        return value.squeeze(-1)

    def get_action(self, state: torch.Tensor, mask: torch.Tensor):
        dist, value = self.forward(state, mask)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, state: torch.Tensor, mask: torch.Tensor, action: torch.Tensor):
        dist, value = self.forward(state, mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy
