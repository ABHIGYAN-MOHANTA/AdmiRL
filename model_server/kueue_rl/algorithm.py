import numpy as np

from .config import GAE_LAMBDA, GAMMA


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0

    for index in reversed(range(len(rewards))):
        next_value = 0.0 if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_value * (1 - dones[index]) - values[index]
        advantages[index] = last_gae = delta + gamma * lam * (1 - dones[index]) * last_gae

    returns = advantages + values
    return advantages, returns
