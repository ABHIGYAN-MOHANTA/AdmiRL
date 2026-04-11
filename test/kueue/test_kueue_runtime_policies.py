from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVER_ROOT = REPO_ROOT / "model_server"
for path in (REPO_ROOT, MODEL_SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from model_server.admirl_server.kueue_runtime import build_kueue_admission_response


def _candidate(
    *,
    action_id: str,
    workload_id: str,
    flavor_name: str,
    wait_seconds: float,
    worker_count: int,
    total_gpu: int,
    per_worker_gpu: int,
    immediate_fit: bool,
    provisionable: bool,
    queue_class: str,
    topology_aware: bool = False,
    topology_preference: str = "",
    competing_older_pressure: float = 0.0,
    elastic_enabled: bool = False,
    min_worker_count: int | None = None,
    preferred_worker_count: int | None = None,
    max_worker_count: int | None = None,
    scale_tag: str = "fixed",
    scale_fraction: float = 1.0,
) -> dict:
    return {
        "action_id": action_id,
        "workload_id": workload_id,
        "flavor_name": flavor_name,
        "queue_name": "lq-a",
        "cluster_queue": "cq-a",
        "fairshare_group": "fg-a",
        "priority": 5,
        "wait_seconds": wait_seconds,
        "runtime_seconds": 600.0,
        "worker_count": worker_count,
        "total_gpu": total_gpu,
        "per_worker_gpu": per_worker_gpu,
        "topology_aware": topology_aware,
        "topology_preference": topology_preference,
        "flavor_domain": flavor_name.rsplit("-", 1)[-1].upper(),
        "immediate_fit": immediate_fit,
        "provisionable": provisionable,
        "available_gpu": total_gpu if immediate_fit else max(0, total_gpu - per_worker_gpu),
        "total_gpu_capacity": total_gpu,
        "fairshare_debt": 0.0,
        "requeue_count": 0,
        "queue_class": queue_class,
        "flavor_gpu_size": per_worker_gpu,
        "oversize_gpu": 0,
        "competing_older_pressure": competing_older_pressure,
        "elastic_enabled": elastic_enabled,
        "min_worker_count": int(min_worker_count if min_worker_count is not None else worker_count),
        "preferred_worker_count": int(preferred_worker_count if preferred_worker_count is not None else worker_count),
        "max_worker_count": int(max_worker_count if max_worker_count is not None else worker_count),
        "scale_tag": scale_tag,
        "scale_fraction": scale_fraction,
    }


class KueueRuntimePoliciesTests(unittest.TestCase):
    class _DummyPolicy:
        def __init__(self, probs: list[float]):
            self.workload_preset = "kueue-lingjun-gang-starvation-cohort"
            self._probs = torch.tensor([probs], dtype=torch.float32)
            self.model = self

        def __call__(self, state_tensor, mask_tensor):
            return SimpleNamespace(probs=self._probs), torch.zeros((1, 1), dtype=torch.float32)

    def test_blocked_guard_penalizes_small_overlap_and_protects_blocked_gang(self) -> None:
        request_state = {
            "request_mode": "kueue-admission",
            "time": 50.0,
            "candidates": [
                _candidate(
                    action_id="gang@a",
                    workload_id="ns/gang",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=40.0,
                    worker_count=2,
                    total_gpu=8,
                    per_worker_gpu=4,
                    immediate_fit=False,
                    provisionable=True,
                    queue_class="gang",
                ),
                _candidate(
                    action_id="small-overlap@a",
                    workload_id="ns/small-overlap",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=45.0,
                    worker_count=1,
                    total_gpu=4,
                    per_worker_gpu=4,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
                _candidate(
                    action_id="small-safe@b",
                    workload_id="ns/small-safe",
                    flavor_name="rf-2gpu-b",
                    wait_seconds=10.0,
                    worker_count=1,
                    total_gpu=2,
                    per_worker_gpu=2,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
            ],
        }

        response = build_kueue_admission_response(request_state, policy=None, policy_mode="blocked_guard")

        self.assertEqual(response["source"], "blocked-guard-kueue")
        self.assertEqual(response["ranked_workloads"][0], "ns/gang")
        self.assertLess(
            response["pair_scores"]["small-overlap@a"],
            response["pair_scores"]["small-safe@b"],
        )

    def test_learned_multi_objective_keeps_blocked_guardrail_even_if_policy_likes_overlap(self) -> None:
        request_state = {
            "request_mode": "kueue-admission",
            "time": 50.0,
            "candidates": [
                _candidate(
                    action_id="gang@a",
                    workload_id="ns/gang",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=40.0,
                    worker_count=2,
                    total_gpu=8,
                    per_worker_gpu=4,
                    immediate_fit=False,
                    provisionable=True,
                    queue_class="gang",
                ),
                _candidate(
                    action_id="small-overlap@a",
                    workload_id="ns/small-overlap",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=45.0,
                    worker_count=1,
                    total_gpu=4,
                    per_worker_gpu=4,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
                _candidate(
                    action_id="small-safe@b",
                    workload_id="ns/small-safe",
                    flavor_name="rf-2gpu-b",
                    wait_seconds=10.0,
                    worker_count=1,
                    total_gpu=2,
                    per_worker_gpu=2,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
            ],
        }

        response = build_kueue_admission_response(
            request_state,
            policy=self._DummyPolicy([0.05, 0.90, 0.05]),
            policy_mode="learned_multi_objective",
        )

        self.assertEqual(response["source"], "learned-multi-objective-starvation-guard-kueue")
        self.assertEqual(response["ranked_workloads"][0], "ns/gang")
        self.assertLess(
            response["pair_scores"]["small-overlap@a"],
            response["pair_scores"]["small-safe@b"],
        )

    def test_learned_multi_objective_can_use_policy_residual_on_safe_choices(self) -> None:
        request_state = {
            "request_mode": "kueue-admission",
            "time": 25.0,
            "candidates": [
                _candidate(
                    action_id="small-a@b",
                    workload_id="ns/small-a",
                    flavor_name="rf-2gpu-b",
                    wait_seconds=12.0,
                    worker_count=1,
                    total_gpu=2,
                    per_worker_gpu=2,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
                _candidate(
                    action_id="small-b@c",
                    workload_id="ns/small-b",
                    flavor_name="rf-2gpu-c",
                    wait_seconds=12.0,
                    worker_count=1,
                    total_gpu=2,
                    per_worker_gpu=2,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
            ],
        }

        response = build_kueue_admission_response(
            request_state,
            policy=self._DummyPolicy([0.10, 0.90]),
            policy_mode="learned_multi_objective",
        )

        self.assertIn(response["source"], {"learned-multi-objective-kueue", "learned-multi-objective-safe-kueue"})
        self.assertEqual(response["ranked_workloads"][0], "ns/small-b")

    def test_blocked_guard_ignores_loaded_checkpoint(self) -> None:
        request_state = {
            "request_mode": "kueue-admission",
            "time": 25.0,
            "candidates": [
                _candidate(
                    action_id="gang@a",
                    workload_id="ns/gang",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=20.0,
                    worker_count=2,
                    total_gpu=8,
                    per_worker_gpu=4,
                    immediate_fit=False,
                    provisionable=True,
                    queue_class="gang",
                ),
                _candidate(
                    action_id="small@a",
                    workload_id="ns/small",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=20.0,
                    worker_count=1,
                    total_gpu=4,
                    per_worker_gpu=4,
                    immediate_fit=True,
                    provisionable=True,
                    queue_class="small",
                ),
            ],
        }

        response = build_kueue_admission_response(
            request_state,
            policy=self._DummyPolicy([0.05, 0.95]),
            policy_mode="blocked_guard",
        )

        self.assertIn(response["source"], {"blocked-guard-kueue", "blocked-guard-fallback-kueue"})
        self.assertEqual(response["ranked_workloads"][0], "ns/gang")

    def test_learned_multi_objective_prefers_safe_elastic_scale_under_blocked_pressure(self) -> None:
        policy = self._DummyPolicy([0.02, 0.75, 0.20, 0.03])
        policy.workload_preset = "kueue-lingjun-gang-elastic-topology"

        request_state = {
            "request_mode": "kueue-admission",
            "time": 25.0,
            "candidates": [
                _candidate(
                    action_id="critical@c",
                    workload_id="ns/critical",
                    flavor_name="rf-4gpu-c",
                    wait_seconds=18.0,
                    worker_count=2,
                    total_gpu=8,
                    per_worker_gpu=4,
                    immediate_fit=False,
                    provisionable=False,
                    queue_class="gang",
                    topology_aware=True,
                    topology_preference="C",
                ),
                _candidate(
                    action_id="elastic@a@w1",
                    workload_id="ns/elastic",
                    flavor_name="rf-4gpu-a",
                    wait_seconds=12.0,
                    worker_count=1,
                    total_gpu=4,
                    per_worker_gpu=4,
                    immediate_fit=True,
                    provisionable=False,
                    queue_class="small",
                    topology_aware=True,
                    topology_preference="C",
                    competing_older_pressure=0.0,
                    elastic_enabled=True,
                    min_worker_count=1,
                    preferred_worker_count=2,
                    max_worker_count=2,
                    scale_tag="min",
                    scale_fraction=0.5,
                ),
                _candidate(
                    action_id="elastic@c@w1",
                    workload_id="ns/elastic",
                    flavor_name="rf-4gpu-c",
                    wait_seconds=12.0,
                    worker_count=1,
                    total_gpu=4,
                    per_worker_gpu=4,
                    immediate_fit=True,
                    provisionable=False,
                    queue_class="small",
                    topology_aware=True,
                    topology_preference="C",
                    competing_older_pressure=1.5,
                    elastic_enabled=True,
                    min_worker_count=1,
                    preferred_worker_count=2,
                    max_worker_count=2,
                    scale_tag="min",
                    scale_fraction=0.5,
                ),
                _candidate(
                    action_id="elastic@c@w2",
                    workload_id="ns/elastic",
                    flavor_name="rf-4gpu-c",
                    wait_seconds=12.0,
                    worker_count=2,
                    total_gpu=8,
                    per_worker_gpu=4,
                    immediate_fit=False,
                    provisionable=True,
                    queue_class="gang",
                    topology_aware=True,
                    topology_preference="C",
                    competing_older_pressure=1.5,
                    elastic_enabled=True,
                    min_worker_count=1,
                    preferred_worker_count=2,
                    max_worker_count=2,
                    scale_tag="max",
                    scale_fraction=1.0,
                ),
            ],
        }

        response = build_kueue_admission_response(
            request_state,
            policy=policy,
            policy_mode="learned_multi_objective",
        )

        self.assertEqual(response["source"], "learned-multi-objective-elastic-safe-kueue")
        self.assertGreater(
            response["pair_scores"]["elastic@a@w1"],
            response["pair_scores"]["elastic@c@w1"],
        )
        self.assertGreater(
            response["pair_scores"]["elastic@a@w1"],
            response["pair_scores"]["elastic@c@w2"],
        )

if __name__ == "__main__":
    unittest.main()
