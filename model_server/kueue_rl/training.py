from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .algorithm import compute_gae
from .config import (
    BATCH_SIZE,
    CHECKPOINT_JCT_TOLERANCE,
    CLIP_EPS,
    ENTROPY_COEFF,
    GAE_LAMBDA,
    GAMMA,
    LR,
    MAX_QUEUE_JOBS,
    PPO_EPOCHS,
    STATE_DIM,
    VALUE_COEFF,
)
from .kueue_admission import KueueAdmissionEnv, canonical_kueue_preset, is_kueue_preset
from .model import ActorCritic


def collect_episode(env, model: ActorCritic, greedy: bool = False) -> tuple[list[dict], dict]:
    transitions = []
    state = env.observe()

    while not env.done():
        mask = env.action_mask()
        if mask.sum() == 0:
            env._auto_advance_until_runnable()
            state = env.observe()
            mask = env.action_mask()
            if mask.sum() == 0:
                if env.done():
                    break
                continue

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = model(state_tensor, mask_tensor)
            probs = dist.probs.squeeze(0)
            valid_indices = torch.nonzero(mask_tensor.squeeze(0) > 0, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                raise RuntimeError("policy step reached with no valid actions")
            valid_probs = probs[valid_indices]
            if torch.sum(valid_probs) <= 0:
                valid_probs = torch.ones_like(valid_probs) / float(valid_probs.numel())
            else:
                valid_probs = valid_probs / torch.sum(valid_probs)
            if greedy:
                local_index = torch.argmax(valid_probs)
            else:
                local_index = torch.multinomial(valid_probs, num_samples=1).squeeze(0)
            action = valid_indices[local_index]
            log_prob = torch.log(valid_probs[local_index].clamp_min(1e-8))

        action_idx = int(action.item())
        reward, done, info = env.schedule_job(action_idx)
        transitions.append(
            {
                "state": state,
                "mask": mask,
                "action": action_idx,
                "log_prob": float(log_prob.item()),
                "value": float(value.squeeze(-1).item()),
                "reward": float(reward),
                "done": float(done),
                "info": info,
            }
        )
        state = env.observe()

    return transitions, env.summary()


def _mean_summary(summaries: list[dict], key: str) -> float:
    values = [float(summary.get(key, 0.0) or 0.0) for summary in summaries]
    return float(np.mean(values)) if values else 0.0


def _checkpoint_signature_for_preset(workload_preset: str, stats: dict) -> tuple[float, ...]:
    if workload_preset in {
        "kueue-lingjun-gang-starvation",
        "kueue-lingjun-gang-starvation-cohort",
    }:
        return (
            stats["val_flavor_head_blocking_seconds"],
            stats["val_idle_quota_while_blocked"],
            stats["val_avg_small_wait_seconds"],
            stats["val_makespan_seconds"],
            -stats["val_throughput_jobs_per_minute"],
            -stats["val_gang_admission_ratio"],
            stats["val_p95_workload_wait_seconds"],
            stats["val_p95_job_completion_seconds"],
            stats["val_fair_share_violation_count"],
            stats["val_avg_fragmentation"],
        )
    if workload_preset == "kueue-lingjun-gang-topology-provisioning":
        return (
            stats["val_avg_critical_completion_seconds"],
            stats["val_avg_critical_wait_seconds"],
            stats["val_avg_topology_aware_completion_seconds"],
            -stats["val_topology_hit_rate"],
            stats["val_avg_provisioning_delay_seconds"],
            stats["val_makespan_seconds"],
            -stats["val_throughput_jobs_per_minute"],
            stats["val_fair_share_violation_count"],
            stats["val_avg_fragmentation"],
        )
    if workload_preset == "kueue-lingjun-gang-elastic-topology":
        return (
            stats["val_avg_elastic_wait_seconds"],
            stats["val_avg_elastic_completion_seconds"],
            stats["val_p95_elastic_wait_seconds"],
            stats["val_p95_elastic_completion_seconds"],
            stats["val_avg_gang_wait_seconds"],
            stats["val_avg_gang_completion_seconds"],
            stats["val_p95_gang_wait_seconds"],
            stats["val_p95_gang_completion_seconds"],
            stats["val_avg_topology_aware_wait_seconds"],
            stats["val_makespan_seconds"],
            -stats["val_throughput_jobs_per_minute"],
            -stats["val_elastic_completion_ratio"],
            -stats["val_gang_admission_ratio"],
            stats["val_avg_fragmentation"],
        )
    if workload_preset == "kueue-lingjun-gang-elastic-profile-cohort":
        return (
            stats["val_avg_elastic_completion_seconds"],
            stats["val_avg_elastic_wait_seconds"],
            stats["val_avg_gang_completion_seconds"],
            stats["val_avg_gang_wait_seconds"],
            stats["val_makespan_seconds"],
            -stats["val_throughput_jobs_per_minute"],
            -stats["val_elastic_completion_ratio"],
            stats["val_avg_fragmentation"],
        )
    return (
        stats["val_p95_workload_wait_seconds"],
        -stats["val_gang_admission_ratio"],
        stats["val_p95_job_completion_seconds"],
        -stats["val_topology_hit_rate"],
        stats["val_flavor_head_blocking_seconds"],
        stats["val_idle_quota_while_blocked"],
        stats["val_fair_share_violation_count"],
        stats["val_avg_provisioning_delay_seconds"],
        stats["val_avg_fragmentation"],
    )


def _checkpoint_primary_metric_for_preset(workload_preset: str, stats: dict) -> float:
    if workload_preset in {
        "kueue-lingjun-gang-starvation",
        "kueue-lingjun-gang-starvation-cohort",
    }:
        return (
            stats["val_flavor_head_blocking_seconds"]
            + (60.0 * stats["val_idle_quota_while_blocked"])
            + (0.08 * stats["val_avg_small_wait_seconds"])
            + (0.04 * stats["val_makespan_seconds"])
            - (0.5 * stats["val_throughput_jobs_per_minute"])
            + (0.10 * stats["val_p95_workload_wait_seconds"])
        )
    if workload_preset == "kueue-lingjun-gang-topology-provisioning":
        return (
            stats["val_avg_critical_completion_seconds"]
            + (0.55 * stats["val_avg_critical_wait_seconds"])
            + (0.30 * stats["val_avg_topology_aware_completion_seconds"])
            + ((1.0 - stats["val_topology_hit_rate"]) * 120.0)
            + (0.08 * stats["val_avg_provisioning_delay_seconds"])
            + (0.04 * stats["val_makespan_seconds"])
            - (0.30 * stats["val_throughput_jobs_per_minute"])
        )
    if workload_preset == "kueue-lingjun-gang-elastic-topology":
        return (
            (0.80 * stats["val_avg_elastic_wait_seconds"])
            + (0.55 * stats["val_avg_elastic_completion_seconds"])
            + (0.45 * stats["val_avg_gang_wait_seconds"])
            + (0.28 * stats["val_avg_gang_completion_seconds"])
            + (0.20 * stats["val_p95_elastic_wait_seconds"])
            + (0.12 * stats["val_p95_gang_wait_seconds"])
            + (0.10 * stats["val_avg_topology_aware_wait_seconds"])
            + (0.04 * stats["val_makespan_seconds"])
            - (0.45 * stats["val_throughput_jobs_per_minute"])
            - (9.0 * stats["val_elastic_completion_ratio"])
            - (2.0 * stats["val_gang_admission_ratio"])
            + (0.05 * stats["val_avg_small_wait_seconds"])
        )
    if workload_preset == "kueue-lingjun-gang-elastic-profile-cohort":
        return (
            stats["val_avg_elastic_completion_seconds"]
            + (0.45 * stats["val_avg_elastic_wait_seconds"])
            + (0.20 * stats["val_avg_gang_completion_seconds"])
            + (0.12 * stats["val_avg_gang_wait_seconds"])
            + (0.05 * stats["val_makespan_seconds"])
            - (0.35 * stats["val_throughput_jobs_per_minute"])
            - (8.0 * stats["val_elastic_completion_ratio"])
        )
    return stats["val_p95_workload_wait_seconds"]


def _build_env(
    *,
    seed: int,
    num_jobs: int,
    arrival_span: float,
    workload_preset: str,
    cluster_layout: str | None,
    trace_split: str,
    trace_train_fraction: float,
):
    return KueueAdmissionEnv(
        seed=seed,
        num_jobs=num_jobs,
        arrival_span=arrival_span,
        workload_preset=workload_preset,
        cluster_layout=cluster_layout,
        trace_split=trace_split,
        trace_train_fraction=trace_train_fraction,
    )


def stack_trajectories(trajectories: list[list[dict]]) -> dict:
    flat = [transition for episode in trajectories for transition in episode]
    if not flat:
        raise ValueError("no transitions collected")
    return {
        "states": torch.tensor(np.array([step["state"] for step in flat]), dtype=torch.float32),
        "masks": torch.tensor(np.array([step["mask"] for step in flat]), dtype=torch.float32),
        "actions": torch.tensor([step["action"] for step in flat], dtype=torch.long),
        "old_log_probs": torch.tensor([step["log_prob"] for step in flat], dtype=torch.float32),
        "values": np.array([step["value"] for step in flat], dtype=np.float32),
        "rewards": np.array([step["reward"] for step in flat], dtype=np.float32),
        "dones": np.array([step["done"] for step in flat], dtype=np.float32),
    }


def ppo_update_policy(model: ActorCritic, optimizer, batch: dict, ppo_epochs: int = PPO_EPOCHS):
    states = batch["states"]
    masks = batch["masks"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    values = batch["values"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    if len(rewards) > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    advantages, returns = compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0

    num_states = len(states)
    for _ in range(ppo_epochs):
        indices = np.random.permutation(num_states)
        for start in range(0, num_states, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_states)
            idx = indices[start:end]

            mb_states = states[idx]
            mb_masks = masks[idx]
            mb_actions = actions[idx]
            mb_old_log_probs = old_log_probs[idx]
            mb_advantages = advantages[idx]
            mb_returns = returns[idx]

            new_log_probs, new_values, entropy = model.evaluate(mb_states, mb_masks, mb_actions)
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, mb_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + VALUE_COEFF * value_loss + ENTROPY_COEFF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            with torch.no_grad():
                total_kl += (mb_old_log_probs - new_log_probs).mean().item()

    num_updates = ppo_epochs * max(1, math.ceil(num_states / BATCH_SIZE))
    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "kl": total_kl / num_updates,
        "num_transitions": num_states,
        "num_updates": num_updates,
    }


def train_policy(
    iterations: int,
    episodes_per_iteration: int,
    num_jobs: int,
    arrival_span: float,
    workload_preset: str = "kueue-lingjun-gang-starvation-cohort",
    base_seed: int = 7,
    cluster_layout: str | None = None,
    train_trace_split: str = "all",
    eval_trace_split: str = "all",
    trace_train_fraction: float = 0.75,
    validation_episodes: int = 3,
) -> tuple[ActorCritic, list[dict]]:
    workload_preset = canonical_kueue_preset(workload_preset)
    if not is_kueue_preset(workload_preset):
        raise ValueError(f"unsupported Kueue preset: {workload_preset}")

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    model = ActorCritic(state_dim=STATE_DIM, n_actions=MAX_QUEUE_JOBS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_signature = None
    best_primary_metric = None

    for iteration in range(iterations):
        trajectories = []
        episode_summaries = []
        for episode_index in range(episodes_per_iteration):
            env = _build_env(
                seed=base_seed + (iteration * episodes_per_iteration) + episode_index,
                num_jobs=num_jobs,
                arrival_span=arrival_span,
                workload_preset=workload_preset,
                cluster_layout=cluster_layout or None,
                trace_split=train_trace_split,
                trace_train_fraction=trace_train_fraction,
            )
            episode_trajectory, summary = collect_episode(env, model, greedy=False)
            trajectories.append(episode_trajectory)
            episode_summaries.append(summary)

        batch = stack_trajectories(trajectories)
        stats = ppo_update_policy(model, optimizer, batch)
        stats["iteration"] = iteration + 1
        stats["avg_reward"] = float(np.mean([summary["total_reward"] for summary in episode_summaries]))
        stats["avg_workload_wait_seconds"] = _mean_summary(episode_summaries, "avg_workload_wait_seconds")
        stats["avg_job_completion_seconds"] = _mean_summary(episode_summaries, "avg_job_completion_seconds")
        stats["avg_p95_workload_wait_seconds"] = _mean_summary(episode_summaries, "p95_workload_wait_seconds")
        stats["avg_p95_job_completion_seconds"] = _mean_summary(episode_summaries, "p95_job_completion_seconds")
        stats["avg_gang_admission_ratio"] = _mean_summary(episode_summaries, "gang_admission_ratio")
        stats["avg_gang_wait_seconds"] = _mean_summary(episode_summaries, "avg_gang_wait_seconds")
        stats["avg_p95_gang_wait_seconds"] = _mean_summary(episode_summaries, "p95_gang_wait_seconds")
        stats["avg_small_wait_seconds"] = _mean_summary(episode_summaries, "avg_small_wait_seconds")
        stats["avg_gang_completion_seconds"] = _mean_summary(episode_summaries, "avg_gang_completion_seconds")
        stats["avg_p95_gang_completion_seconds"] = _mean_summary(episode_summaries, "p95_gang_completion_seconds")
        stats["avg_small_completion_seconds"] = _mean_summary(episode_summaries, "avg_small_completion_seconds")
        stats["avg_topology_aware_wait_seconds"] = _mean_summary(episode_summaries, "avg_topology_aware_wait_seconds")
        stats["avg_topology_aware_completion_seconds"] = _mean_summary(episode_summaries, "avg_topology_aware_completion_seconds")
        stats["avg_critical_wait_seconds"] = _mean_summary(episode_summaries, "avg_critical_wait_seconds")
        stats["avg_critical_completion_seconds"] = _mean_summary(episode_summaries, "avg_critical_completion_seconds")
        stats["avg_elastic_wait_seconds"] = _mean_summary(episode_summaries, "avg_elastic_wait_seconds")
        stats["avg_p95_elastic_wait_seconds"] = _mean_summary(episode_summaries, "p95_elastic_wait_seconds")
        stats["avg_elastic_completion_seconds"] = _mean_summary(episode_summaries, "avg_elastic_completion_seconds")
        stats["avg_p95_elastic_completion_seconds"] = _mean_summary(episode_summaries, "p95_elastic_completion_seconds")
        stats["avg_elastic_initial_scale_fraction"] = _mean_summary(episode_summaries, "avg_elastic_initial_scale_fraction")
        stats["elastic_completion_ratio"] = _mean_summary(episode_summaries, "elastic_completion_ratio")
        stats["avg_makespan_seconds"] = _mean_summary(episode_summaries, "makespan_seconds")
        stats["avg_throughput_jobs_per_minute"] = _mean_summary(episode_summaries, "throughput_jobs_per_minute")
        stats["avg_throughput_gpu_per_minute"] = _mean_summary(episode_summaries, "throughput_gpu_per_minute")
        stats["avg_topology_hit_rate"] = _mean_summary(episode_summaries, "topology_hit_rate")
        stats["avg_flavor_head_blocking_seconds"] = _mean_summary(episode_summaries, "flavor_head_blocking_seconds")
        stats["avg_idle_quota_while_blocked"] = _mean_summary(episode_summaries, "idle_quota_while_blocked")
        stats["avg_provisioning_delay_seconds"] = _mean_summary(episode_summaries, "avg_provisioning_delay_seconds")
        stats["avg_fragmentation"] = _mean_summary(episode_summaries, "avg_gpu_fragmentation")
        stats["avg_fair_share_violation_count"] = _mean_summary(episode_summaries, "fair_share_violation_count")

        validation_summaries = []
        for offset in range(max(1, validation_episodes)):
            env = _build_env(
                seed=base_seed + 5000 + offset,
                num_jobs=num_jobs,
                arrival_span=arrival_span,
                workload_preset=workload_preset,
                cluster_layout=cluster_layout or None,
                trace_split=eval_trace_split,
                trace_train_fraction=trace_train_fraction,
            )
            _, validation_summary = collect_episode(env, model, greedy=True)
            validation_summaries.append(validation_summary)

        stats["val_avg_workload_wait_seconds"] = _mean_summary(validation_summaries, "avg_workload_wait_seconds")
        stats["val_p95_workload_wait_seconds"] = _mean_summary(validation_summaries, "p95_workload_wait_seconds")
        stats["val_avg_job_completion_seconds"] = _mean_summary(validation_summaries, "avg_job_completion_seconds")
        stats["val_p95_job_completion_seconds"] = _mean_summary(validation_summaries, "p95_job_completion_seconds")
        stats["val_gang_admission_ratio"] = _mean_summary(validation_summaries, "gang_admission_ratio")
        stats["val_avg_gang_wait_seconds"] = _mean_summary(validation_summaries, "avg_gang_wait_seconds")
        stats["val_p95_gang_wait_seconds"] = _mean_summary(validation_summaries, "p95_gang_wait_seconds")
        stats["val_avg_small_wait_seconds"] = _mean_summary(validation_summaries, "avg_small_wait_seconds")
        stats["val_avg_gang_completion_seconds"] = _mean_summary(validation_summaries, "avg_gang_completion_seconds")
        stats["val_p95_gang_completion_seconds"] = _mean_summary(validation_summaries, "p95_gang_completion_seconds")
        stats["val_avg_small_completion_seconds"] = _mean_summary(validation_summaries, "avg_small_completion_seconds")
        stats["val_avg_topology_aware_wait_seconds"] = _mean_summary(validation_summaries, "avg_topology_aware_wait_seconds")
        stats["val_avg_topology_aware_completion_seconds"] = _mean_summary(validation_summaries, "avg_topology_aware_completion_seconds")
        stats["val_avg_critical_wait_seconds"] = _mean_summary(validation_summaries, "avg_critical_wait_seconds")
        stats["val_avg_critical_completion_seconds"] = _mean_summary(validation_summaries, "avg_critical_completion_seconds")
        stats["val_avg_elastic_wait_seconds"] = _mean_summary(validation_summaries, "avg_elastic_wait_seconds")
        stats["val_p95_elastic_wait_seconds"] = _mean_summary(validation_summaries, "p95_elastic_wait_seconds")
        stats["val_avg_elastic_completion_seconds"] = _mean_summary(validation_summaries, "avg_elastic_completion_seconds")
        stats["val_p95_elastic_completion_seconds"] = _mean_summary(validation_summaries, "p95_elastic_completion_seconds")
        stats["val_avg_elastic_initial_scale_fraction"] = _mean_summary(validation_summaries, "avg_elastic_initial_scale_fraction")
        stats["val_elastic_completion_ratio"] = _mean_summary(validation_summaries, "elastic_completion_ratio")
        stats["val_makespan_seconds"] = _mean_summary(validation_summaries, "makespan_seconds")
        stats["val_throughput_jobs_per_minute"] = _mean_summary(validation_summaries, "throughput_jobs_per_minute")
        stats["val_throughput_gpu_per_minute"] = _mean_summary(validation_summaries, "throughput_gpu_per_minute")
        stats["val_topology_hit_rate"] = _mean_summary(validation_summaries, "topology_hit_rate")
        stats["val_flavor_head_blocking_seconds"] = _mean_summary(validation_summaries, "flavor_head_blocking_seconds")
        stats["val_idle_quota_while_blocked"] = _mean_summary(validation_summaries, "idle_quota_while_blocked")
        stats["val_avg_provisioning_delay_seconds"] = _mean_summary(validation_summaries, "avg_provisioning_delay_seconds")
        stats["val_avg_fragmentation"] = _mean_summary(validation_summaries, "avg_gpu_fragmentation")
        stats["val_fair_share_violation_count"] = _mean_summary(validation_summaries, "fair_share_violation_count")

        candidate_signature = _checkpoint_signature_for_preset(workload_preset, stats)
        candidate_primary = _checkpoint_primary_metric_for_preset(workload_preset, stats)
        if best_signature is None:
            best_signature = candidate_signature
            best_primary_metric = candidate_primary
            best_state_dict = copy.deepcopy(model.state_dict())
            stats["selected_checkpoint"] = True
        else:
            current_primary = best_primary_metric if best_primary_metric is not None else candidate_primary
            primary_cutoff = current_primary * (1.0 + CHECKPOINT_JCT_TOLERANCE)
            materially_better_primary = candidate_primary < (current_primary * 0.995)
            comparable_primary = candidate_primary <= primary_cutoff

            if materially_better_primary or (comparable_primary and candidate_signature < best_signature):
                best_signature = candidate_signature
                best_primary_metric = candidate_primary
                best_state_dict = copy.deepcopy(model.state_dict())
                stats["selected_checkpoint"] = True
            else:
                stats["selected_checkpoint"] = False
        history.append(stats)

    model.load_state_dict(best_state_dict)
    return model, history


def evaluate_model(
    model: ActorCritic,
    seeds: list[int],
    num_jobs: int,
    arrival_span: float,
    workload_preset: str = "kueue-lingjun-gang-starvation-cohort",
    greedy: bool = True,
    cluster_layout: str | None = None,
    trace_split: str = "all",
    trace_train_fraction: float = 0.75,
) -> list[dict]:
    workload_preset = canonical_kueue_preset(workload_preset)
    summaries = []
    for seed in seeds:
        env = _build_env(
            seed=seed,
            num_jobs=num_jobs,
            arrival_span=arrival_span,
            workload_preset=workload_preset,
            cluster_layout=cluster_layout or None,
            trace_split=trace_split,
            trace_train_fraction=trace_train_fraction,
        )
        _, summary = collect_episode(env, model, greedy=greedy)
        summary["seed"] = seed
        summaries.append(summary)
    return summaries
