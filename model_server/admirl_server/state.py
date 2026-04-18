from __future__ import annotations

import math
import threading

DASHBOARD_HISTORY_LIMIT = 120
BENCHMARK_HISTORY_LIMIT = 240
VALID_RUNTIME_POLICIES = {
    "blocked_guard",
    "learned_multi_objective",
}


def runtime_policy_uses_checkpoint(policy_name: str) -> bool:
    return str(policy_name or "").strip().lower() == "learned_multi_objective"


def effective_runtime_policy_name(runtime_policy: str, has_checkpoint: bool) -> str:
    policy_name = str(runtime_policy or "").strip().lower() or "blocked_guard"
    if runtime_policy_uses_checkpoint(policy_name) and not has_checkpoint:
        return "blocked_guard"
    return policy_name


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(float(max(0.0, value)) for value in values)

    def percentile(rank: float) -> float:
        index = max(0, min(len(ordered) - 1, int(math.ceil(rank * len(ordered))) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / float(len(ordered)),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": ordered[-1],
    }


class ModelServerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_decision = {
            "timestamp": None,
            "request": None,
            "response": None,
        }
        self.runtime_policy = "blocked_guard"
        self.learned_checkpoint_path = ""
        self.learned_checkpoint = None
        self.request_latency_ms: dict[str, list[float]] = {}
        self.decision_history: list[dict[str, object]] = []
        self.decision_pair_counts: dict[tuple[str, str], int] = {}
        self.benchmark_history: list[dict[str, object]] = []
        self.benchmark_status: dict[str, object] = {}

    def reset_runtime_metrics_locked(self) -> None:
        self.request_latency_ms = {}
        self.decision_history = []
        self.decision_pair_counts = {}
        self.reset_benchmark_metrics_locked()
        self.last_decision = {
            "timestamp": None,
            "request": None,
            "response": None,
        }

    def reset_benchmark_metrics_locked(self) -> None:
        self.benchmark_history = []
        self.benchmark_status = {}

    def record_request_latency_locked(self, route_name: str, elapsed_ms: float) -> None:
        key = str(route_name or "unknown").strip() or "unknown"
        self.request_latency_ms.setdefault(key, []).append(float(max(0.0, elapsed_ms)))

    def record_decision_event_locked(
        self,
        *,
        route_name: str,
        timestamp: str | None,
        request_state: dict,
        response: dict,
        elapsed_ms: float,
    ) -> None:
        source = (
            str(response.get("advisor_source") or response.get("source") or "unknown").strip()
            or "unknown"
        )
        effective_policy = effective_runtime_policy_name(
            self.runtime_policy,
            self.learned_checkpoint is not None,
        )
        raw_candidates = request_state.get("candidates")
        candidates = raw_candidates if isinstance(raw_candidates, list) else []
        pair_scores = response.get("pair_scores") if isinstance(response.get("pair_scores"), dict) else {}
        ordered_scores = sorted(
            (float(score) for score in pair_scores.values()),
            reverse=True,
        )
        top_score = ordered_scores[0] if ordered_scores else 0.0
        next_score = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
        sample = {
            "timestamp": timestamp,
            "route_name": str(route_name or "unknown").strip() or "unknown",
            "source": source,
            "effective_policy": effective_policy,
            "runtime_policy": str(self.runtime_policy or "").strip() or "blocked_guard",
            "checkpoint_loaded": bool(self.learned_checkpoint is not None),
            "candidate_count": len(candidates),
            "workload_count": len({
                str(candidate.get("workload_id", "") or "").strip()
                for candidate in candidates
                if str(candidate.get("workload_id", "") or "").strip()
            }),
            "immediate_fit_count": sum(1 for candidate in candidates if bool(candidate.get("immediate_fit", False))),
            "provisionable_only_count": sum(
                1
                for candidate in candidates
                if bool(candidate.get("provisionable", False)) and not bool(candidate.get("immediate_fit", False))
            ),
            "elastic_candidate_count": sum(1 for candidate in candidates if bool(candidate.get("elastic_enabled", False))),
            "topology_candidate_count": sum(1 for candidate in candidates if bool(candidate.get("topology_aware", False))),
            "ranked_workload_count": len(response.get("ranked_workloads") or []),
            "protected_active": bool(response.get("protected_workload")),
            "protected_total_gpu": int(response.get("protected_total_gpu", 0) or 0),
            "fallback_active": "fallback" in source,
            "latency_ms": float(max(0.0, elapsed_ms)),
            "top_score": float(top_score),
            "score_gap": float(max(0.0, top_score - next_score)),
        }
        self.decision_history.append(sample)
        if len(self.decision_history) > DASHBOARD_HISTORY_LIMIT:
            self.decision_history = self.decision_history[-DASHBOARD_HISTORY_LIMIT:]
        self.decision_pair_counts[(source, effective_policy)] = int(
            self.decision_pair_counts.get((source, effective_policy), 0)
        ) + 1

    def record_benchmark_snapshot_locked(self, snapshot: dict) -> None:
        sample = {
            "timestamp": str(snapshot.get("timestamp") or ""),
            "preset": str(snapshot.get("preset") or ""),
            "arm": str(snapshot.get("arm") or ""),
            "namespace": str(snapshot.get("namespace") or ""),
            "seed": int(snapshot.get("seed", 0) or 0),
            "phase": str(snapshot.get("phase") or "running"),
            "active": bool(snapshot.get("active", True)),
            "elapsed_seconds": float(snapshot.get("elapsed_seconds", 0.0) or 0.0),
            "timeout_seconds": float(snapshot.get("timeout_seconds", 0.0) or 0.0),
            "progress_ratio": float(snapshot.get("progress_ratio", 0.0) or 0.0),
            "jobs_total": int(snapshot.get("jobs_total", 0) or 0),
            "jobs_completed": int(snapshot.get("jobs_completed", 0) or 0),
            "jobs_running": int(snapshot.get("jobs_running", 0) or 0),
            "jobs_pending": int(snapshot.get("jobs_pending", 0) or 0),
            "avg_gang_wait_seconds": float(snapshot.get("avg_gang_wait_seconds", 0.0) or 0.0),
            "avg_small_wait_seconds": float(snapshot.get("avg_small_wait_seconds", 0.0) or 0.0),
            "avg_elastic_wait_seconds": float(snapshot.get("avg_elastic_wait_seconds", 0.0) or 0.0),
            "head_gang_blocked_seconds": float(snapshot.get("head_gang_blocked_seconds", 0.0) or 0.0),
            "head_gang_current_blocked_seconds": float(
                snapshot.get(
                    "head_gang_current_blocked_seconds",
                    snapshot.get("head_gang_blocked_seconds", 0.0),
                )
                or 0.0
            ),
            "throughput_jobs_per_minute": float(snapshot.get("throughput_jobs_per_minute", 0.0) or 0.0),
            "throughput_gpu_per_minute": float(snapshot.get("throughput_gpu_per_minute", 0.0) or 0.0),
            "job_completion_ratio": float(snapshot.get("job_completion_ratio", 0.0) or 0.0),
            "gang_completion_ratio": float(snapshot.get("gang_completion_ratio", 0.0) or 0.0),
            "elastic_completion_ratio": float(snapshot.get("elastic_completion_ratio", 0.0) or 0.0),
            "topology_hit_rate": float(snapshot.get("topology_hit_rate", 0.0) or 0.0),
            "small_job_bypass_count": float(snapshot.get("small_job_bypass_count", 0.0) or 0.0),
            "head_gang_workload": str(snapshot.get("head_gang_workload") or ""),
        }
        self.benchmark_status = sample
        self.benchmark_history.append(sample)
        if len(self.benchmark_history) > BENCHMARK_HISTORY_LIMIT:
            self.benchmark_history = self.benchmark_history[-BENCHMARK_HISTORY_LIMIT:]

    def runtime_metrics_snapshot_locked(self) -> dict[str, object]:
        return {
            "request_latency_ms": {
                route_name: _latency_summary(values)
                for route_name, values in sorted(self.request_latency_ms.items())
            },
            "benchmark": self.benchmark_status_snapshot_locked(),
        }

    def benchmark_status_snapshot_locked(self, *, window: int = 60) -> dict[str, object]:
        limit = max(1, int(window))
        samples = list(self.benchmark_history[-limit:])
        latest = dict(self.benchmark_status) if self.benchmark_status else {}
        return {
            "active": bool(latest.get("active", False)),
            "history_points": len(self.benchmark_history),
            "latest": latest,
            "samples": samples,
        }

    def prometheus_metrics_locked(self) -> str:
        lines = [
            "# HELP admirl_requests_total Total HTTP requests handled by AdmiRL runtime routes.",
            "# TYPE admirl_requests_total gauge",
        ]
        for route_name, values in sorted(self.request_latency_ms.items()):
            lines.append(f'admirl_requests_total{{route="{_prometheus_escape(route_name)}"}} {len(values)}')

        lines.extend([
            "# HELP admirl_request_latency_mean_ms Mean request latency in milliseconds by route.",
            "# TYPE admirl_request_latency_mean_ms gauge",
        ])
        for route_name, values in sorted(self.request_latency_ms.items()):
            summary = _latency_summary(values)
            lines.append(
                f'admirl_request_latency_mean_ms{{route="{_prometheus_escape(route_name)}"}} '
                f'{summary["mean_ms"]:.6f}'
            )

        lines.extend([
            "# HELP admirl_request_latency_p95_ms P95 request latency in milliseconds by route.",
            "# TYPE admirl_request_latency_p95_ms gauge",
        ])
        for route_name, values in sorted(self.request_latency_ms.items()):
            summary = _latency_summary(values)
            lines.append(
                f'admirl_request_latency_p95_ms{{route="{_prometheus_escape(route_name)}"}} '
                f'{summary["p95_ms"]:.6f}'
            )

        lines.extend([
            "# HELP admirl_decisions_total Total admission decisions by source and effective policy.",
            "# TYPE admirl_decisions_total gauge",
        ])
        for (source, policy), count in sorted(self.decision_pair_counts.items()):
            lines.append(
                'admirl_decisions_total{source="%s",effective_policy="%s"} %d'
                % (_prometheus_escape(source), _prometheus_escape(policy), count)
            )

        latest = self.decision_history[-1] if self.decision_history else {}
        latest_fields = {
            "admirl_last_latency_ms": float(latest.get("latency_ms", 0.0) or 0.0),
            "admirl_last_candidate_count": float(latest.get("candidate_count", 0) or 0.0),
            "admirl_last_immediate_fit_candidates": float(latest.get("immediate_fit_count", 0) or 0.0),
            "admirl_last_provisionable_only_candidates": float(latest.get("provisionable_only_count", 0) or 0.0),
            "admirl_last_elastic_candidates": float(latest.get("elastic_candidate_count", 0) or 0.0),
            "admirl_last_topology_candidates": float(latest.get("topology_candidate_count", 0) or 0.0),
            "admirl_last_ranked_workloads": float(latest.get("ranked_workload_count", 0) or 0.0),
            "admirl_last_top_score": float(latest.get("top_score", 0.0) or 0.0),
            "admirl_last_score_gap": float(latest.get("score_gap", 0.0) or 0.0),
            "admirl_last_protected_total_gpu": float(latest.get("protected_total_gpu", 0) or 0.0),
        }
        for metric_name, value in latest_fields.items():
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value:.6f}")

        lines.extend([
            "# HELP admirl_runtime_policy_info Current runtime and effective policy labels.",
            "# TYPE admirl_runtime_policy_info gauge",
        ])
        lines.append(
            'admirl_runtime_policy_info{runtime_policy="%s",effective_policy="%s",checkpoint_loaded="%s"} 1'
            % (
                _prometheus_escape(str(self.runtime_policy or "").strip() or "blocked_guard"),
                _prometheus_escape(
                    effective_runtime_policy_name(
                        self.runtime_policy,
                        self.learned_checkpoint is not None,
                    )
                ),
                "true" if self.learned_checkpoint is not None else "false",
            )
        )

        benchmark = self.benchmark_status or {}
        lines.extend([
            "# HELP admirl_benchmark_run_info Current active benchmark metadata.",
            "# TYPE admirl_benchmark_run_info gauge",
        ])
        lines.append(
            'admirl_benchmark_run_info{preset="%s",arm="%s",namespace="%s",seed="%s",phase="%s",active="%s"} %d'
            % (
                _prometheus_escape(str(benchmark.get("preset", "") or "")),
                _prometheus_escape(str(benchmark.get("arm", "") or "")),
                _prometheus_escape(str(benchmark.get("namespace", "") or "")),
                _prometheus_escape(str(benchmark.get("seed", "") or "")),
                _prometheus_escape(str(benchmark.get("phase", "") or "")),
                "true" if bool(benchmark.get("active", False)) else "false",
                1 if benchmark else 0,
            )
        )

        benchmark_fields = {
            "admirl_benchmark_progress_ratio": float(benchmark.get("progress_ratio", 0.0) or 0.0),
            "admirl_benchmark_elapsed_seconds": float(benchmark.get("elapsed_seconds", 0.0) or 0.0),
            "admirl_benchmark_timeout_seconds": float(benchmark.get("timeout_seconds", 0.0) or 0.0),
            "admirl_benchmark_jobs_total": float(benchmark.get("jobs_total", 0) or 0.0),
            "admirl_benchmark_jobs_completed": float(benchmark.get("jobs_completed", 0) or 0.0),
            "admirl_benchmark_jobs_running": float(benchmark.get("jobs_running", 0) or 0.0),
            "admirl_benchmark_jobs_pending": float(benchmark.get("jobs_pending", 0) or 0.0),
            "admirl_benchmark_avg_gang_wait_seconds": float(benchmark.get("avg_gang_wait_seconds", 0.0) or 0.0),
            "admirl_benchmark_avg_small_wait_seconds": float(benchmark.get("avg_small_wait_seconds", 0.0) or 0.0),
            "admirl_benchmark_avg_elastic_wait_seconds": float(benchmark.get("avg_elastic_wait_seconds", 0.0) or 0.0),
            "admirl_benchmark_head_gang_blocked_seconds": float(
                benchmark.get("head_gang_blocked_seconds", 0.0) or 0.0
            ),
            "admirl_benchmark_head_gang_current_blocked_seconds": float(
                benchmark.get("head_gang_current_blocked_seconds", 0.0) or 0.0
            ),
            "admirl_benchmark_throughput_jobs_per_minute": float(
                benchmark.get("throughput_jobs_per_minute", 0.0) or 0.0
            ),
            "admirl_benchmark_throughput_gpu_per_minute": float(
                benchmark.get("throughput_gpu_per_minute", 0.0) or 0.0
            ),
            "admirl_benchmark_job_completion_ratio": float(benchmark.get("job_completion_ratio", 0.0) or 0.0),
            "admirl_benchmark_gang_completion_ratio": float(benchmark.get("gang_completion_ratio", 0.0) or 0.0),
            "admirl_benchmark_elastic_completion_ratio": float(
                benchmark.get("elastic_completion_ratio", 0.0) or 0.0
            ),
            "admirl_benchmark_topology_hit_rate": float(benchmark.get("topology_hit_rate", 0.0) or 0.0),
            "admirl_benchmark_small_job_bypass_count": float(benchmark.get("small_job_bypass_count", 0.0) or 0.0),
        }
        for metric_name, value in benchmark_fields.items():
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value:.6f}")
        return "\n".join(lines) + "\n"


def _prometheus_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


state = ModelServerState()
