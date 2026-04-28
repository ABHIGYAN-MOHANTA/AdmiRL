from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import (
    DEFAULT_ARRIVAL_SPAN,
    DEFAULT_EPISODE_JOBS,
    DEFAULT_EPISODES_PER_ITERATION,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_TRAINING_ITERATIONS,
    DEFAULT_WORKLOAD_PRESET,
    MAX_CLUSTER_NODES,
    MAX_QUEUE_JOBS,
    STATE_DIM,
)
from .kueue_admission import (
    canonical_kueue_preset,
    default_cluster_layout_for_kueue_preset,
    is_kueue_preset,
)
from .lingjun import load_lingjun_workloads
from .training import evaluate_model, train_policy


def normalize_trace_split(raw: str) -> str:
    value = str(raw or "all").strip().lower()
    if value not in {"all", "train", "test"}:
        raise ValueError(f"unsupported trace split: {raw}")
    return value


def normalize_trace_train_fraction(raw: float) -> float:
    value = float(raw)
    if value <= 0.0 or value >= 1.0:
        raise ValueError("trace_train_fraction must be in the open interval (0, 1)")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description="PPO trainer for Kueue gang-admission policies")
    parser.add_argument("--iterations", type=int, default=DEFAULT_TRAINING_ITERATIONS, help="PPO outer iterations")
    parser.add_argument("--episodes-per-iteration", type=int, default=DEFAULT_EPISODES_PER_ITERATION, help="Episodes collected before each PPO update")
    parser.add_argument("--num-jobs", type=int, default=DEFAULT_EPISODE_JOBS, help="Workloads per episode")
    parser.add_argument("--arrival-span", type=float, default=DEFAULT_ARRIVAL_SPAN, help="Arrival spread override in simulated seconds")
    parser.add_argument(
        "--workload-preset",
        type=str,
        default=DEFAULT_WORKLOAD_PRESET,
        help="Kueue preset (for example: kueue-lingjun-gang-starvation, kueue-lingjun-gang-starvation-cohort, kueue-lingjun-gang-topology-provisioning)",
    )
    parser.add_argument("--cluster-layout", type=str, default="", help="Optional cluster layout override")
    parser.add_argument("--base-seed", type=int, default=7, help="Base random seed")
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="How many evaluation seeds to run after training")
    parser.add_argument("--train-trace-split", type=str, default="all", help="Trace split for training episodes: all, train, or test")
    parser.add_argument("--eval-trace-split", type=str, default="all", help="Trace split for evaluation episodes: all, train, or test")
    parser.add_argument("--trace-train-fraction", type=float, default=0.75, help="Chronological fraction reserved for the training split")
    parser.add_argument("--save", type=str, default="", help="Optional path to save the learned checkpoint")
    parser.add_argument("--output", type=str, default="", help="Optional path to save JSON training/eval summary")
    return parser


def train(args):
    print("=" * 72)
    print("AdmiRL Gang Trainer")
    print("=" * 72)
    print(f"State dim: {STATE_DIM} | Action slots: {MAX_QUEUE_JOBS} | Max nodes: {MAX_CLUSTER_NODES}")
    print(
        f"Iterations={args.iterations}, episodes/iter={args.episodes_per_iteration}, "
        f"jobs/episode={args.num_jobs}, arrival_span={args.arrival_span}, preset={args.workload_preset}"
    )

    train_trace_split = normalize_trace_split(args.train_trace_split)
    eval_trace_split = normalize_trace_split(args.eval_trace_split)
    trace_train_fraction = normalize_trace_train_fraction(args.trace_train_fraction)

    workload_preset = canonical_kueue_preset(args.workload_preset)
    if not is_kueue_preset(workload_preset):
        raise ValueError(f"unsupported Kueue preset: {args.workload_preset}")

    total_rows = len(load_lingjun_workloads(trace_split="all", train_fraction=trace_train_fraction))
    train_rows = len(load_lingjun_workloads(trace_split="train", train_fraction=trace_train_fraction))
    test_rows = len(load_lingjun_workloads(trace_split="test", train_fraction=trace_train_fraction))
    print(
        f"Lingjun split: train={train_trace_split}, eval={eval_trace_split}, "
        f"train_fraction={trace_train_fraction:.2f} "
        f"(rows: total={total_rows}, train={train_rows}, test={test_rows})"
    )

    model, history = train_policy(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        num_jobs=args.num_jobs,
        arrival_span=args.arrival_span,
        workload_preset=workload_preset,
        base_seed=args.base_seed,
        cluster_layout=args.cluster_layout or None,
        train_trace_split=train_trace_split,
        eval_trace_split=eval_trace_split,
        trace_train_fraction=trace_train_fraction,
        validation_episodes=max(1, args.eval_episodes),
    )

    print("\nTraining progress:")
    for row in history:
        print(
            f"  iter={row['iteration']:02d} "
            f"avg_reward={row['avg_reward']:+.3f} "
            f"avg_wait={row['avg_workload_wait_seconds']:.2f} "
            f"avg_completion={row['avg_job_completion_seconds']:.2f} "
            f"gang={row['avg_gang_admission_ratio']:.3f} "
            f"topo={row['avg_topology_hit_rate']:.3f}"
        )

    eval_seeds = [args.base_seed + 1000 + index for index in range(args.eval_episodes)]
    evaluations = evaluate_model(
        model,
        seeds=eval_seeds,
        num_jobs=args.num_jobs,
        arrival_span=args.arrival_span,
        workload_preset=workload_preset,
        greedy=True,
        cluster_layout=args.cluster_layout or None,
        trace_split=eval_trace_split,
        trace_train_fraction=trace_train_fraction,
    )
    aggregate = {
        "avg_workload_wait_seconds": sum(item["avg_workload_wait_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "p95_workload_wait_seconds": sum(item["p95_workload_wait_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "p99_workload_wait_seconds": sum(item["p99_workload_wait_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "avg_job_completion_seconds": sum(item["avg_job_completion_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "avg_topology_aware_wait_seconds": sum(item["avg_topology_aware_wait_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "avg_topology_aware_completion_seconds": sum(item["avg_topology_aware_completion_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "avg_critical_wait_seconds": sum(item["avg_critical_wait_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "avg_critical_completion_seconds": sum(item["avg_critical_completion_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "p95_job_completion_seconds": sum(item["p95_job_completion_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "p99_job_completion_seconds": sum(item["p99_job_completion_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "gang_admission_ratio": sum(item["gang_admission_ratio"] for item in evaluations) / max(len(evaluations), 1),
        "topology_hit_rate": sum(item["topology_hit_rate"] for item in evaluations) / max(len(evaluations), 1),
        "flavor_head_blocking_seconds": sum(item["flavor_head_blocking_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "idle_quota_while_blocked": sum(item["idle_quota_while_blocked"] for item in evaluations) / max(len(evaluations), 1),
        "avg_provisioning_delay_seconds": sum(item["avg_provisioning_delay_seconds"] for item in evaluations) / max(len(evaluations), 1),
        "avg_gpu_fragmentation": sum(item["avg_gpu_fragmentation"] for item in evaluations) / max(len(evaluations), 1),
        "fair_share_violation_count": sum(item["fair_share_violation_count"] for item in evaluations) / max(len(evaluations), 1),
    }

    result = {
        "config": {
            "iterations": args.iterations,
            "episodes_per_iteration": args.episodes_per_iteration,
            "num_jobs": args.num_jobs,
            "arrival_span": args.arrival_span,
            "workload_preset": workload_preset,
            "cluster_layout": args.cluster_layout or default_cluster_layout_for_kueue_preset(workload_preset),
            "base_seed": args.base_seed,
            "eval_episodes": args.eval_episodes,
            "train_trace_split": train_trace_split,
            "eval_trace_split": eval_trace_split,
            "trace_train_fraction": trace_train_fraction,
        },
        "training_history": history,
        "evaluation": evaluations,
        "evaluation_summary": aggregate,
    }

    print("\nEvaluation summary:")
    print(json.dumps(aggregate, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "state_dim": STATE_DIM,
                "n_actions": MAX_QUEUE_JOBS,
                "max_cluster_nodes": MAX_CLUSTER_NODES,
                "workload_preset": workload_preset,
                "cluster_layout": args.cluster_layout or default_cluster_layout_for_kueue_preset(workload_preset),
                "num_jobs": args.num_jobs,
                "arrival_span": args.arrival_span,
                "base_seed": args.base_seed,
                "train_trace_split": train_trace_split,
                "eval_trace_split": eval_trace_split,
                "trace_train_fraction": trace_train_fraction,
            },
            save_path,
        )
        print(f"\nSaved checkpoint to {save_path}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Saved summary to {output_path}")

    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
