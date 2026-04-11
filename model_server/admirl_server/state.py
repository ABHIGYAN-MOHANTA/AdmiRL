from __future__ import annotations

import math
import threading

DEFAULT_SCORE = 50
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

    def reset_runtime_metrics_locked(self) -> None:
        self.request_latency_ms = {}

    def record_request_latency_locked(self, route_name: str, elapsed_ms: float) -> None:
        key = str(route_name or "unknown").strip() or "unknown"
        self.request_latency_ms.setdefault(key, []).append(float(max(0.0, elapsed_ms)))

    def runtime_metrics_snapshot_locked(self) -> dict[str, object]:
        return {
            "request_latency_ms": {
                route_name: _latency_summary(values)
                for route_name, values in sorted(self.request_latency_ms.items())
            }
        }


state = ModelServerState()
