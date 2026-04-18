from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVER_ROOT = REPO_ROOT / "model_server"
for path in (REPO_ROOT, MODEL_SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_server.admirl_server.state import ModelServerState


class ModelServerRuntimeMetricsTests(unittest.TestCase):
    def test_runtime_metrics_snapshot_summarizes_route_latencies(self) -> None:
        state = ModelServerState()
        with state.lock:
            state.record_request_latency_locked("kueue_admission_advice", 10.0)
            state.record_request_latency_locked("kueue_admission_advice", 30.0)
            snapshot = state.runtime_metrics_snapshot_locked()

        self.assertEqual(snapshot["request_latency_ms"]["kueue_admission_advice"]["count"], 2)
        self.assertAlmostEqual(snapshot["request_latency_ms"]["kueue_admission_advice"]["mean_ms"], 20.0)

    def test_reset_runtime_metrics_clears_samples(self) -> None:
        state = ModelServerState()
        with state.lock:
            state.record_request_latency_locked("kueue_admission_advice", 12.0)
            state.last_decision["timestamp"] = "2026-04-12T04:35:43"
            state.last_decision["request"] = {"request_mode": "kueue-admission"}
            state.last_decision["response"] = {"source": "blocked-guard-kueue"}
            state.reset_runtime_metrics_locked()
            snapshot = state.runtime_metrics_snapshot_locked()

        self.assertEqual(snapshot["request_latency_ms"], {})
        self.assertIsNone(state.last_decision["timestamp"])
        self.assertIsNone(state.last_decision["request"])
        self.assertIsNone(state.last_decision["response"])

    def test_record_decision_event_tracks_latest_sample_fields(self) -> None:
        state = ModelServerState()
        request_state = {
            "candidates": [
                {
                    "workload_id": "ns/gang",
                    "immediate_fit": False,
                    "provisionable": True,
                    "elastic_enabled": True,
                    "topology_aware": True,
                },
                {
                    "workload_id": "ns/small",
                    "immediate_fit": True,
                    "provisionable": True,
                    "elastic_enabled": False,
                    "topology_aware": False,
                },
            ]
        }
        response = {
            "source": "learned-multi-objective-kueue",
            "ranked_workloads": ["ns/gang", "ns/small"],
            "pair_scores": {
                "gang@a": 12.0,
                "small@b": 8.5,
            },
            "protected_workload": "ns/gang",
            "protected_total_gpu": 8,
        }

        with state.lock:
            state.learned_checkpoint = object()
            state.runtime_policy = "learned_multi_objective"
            state.record_decision_event_locked(
                route_name="kueue_admission_advice",
                timestamp="2026-04-12T01:02:03",
                request_state=request_state,
                response=response,
                elapsed_ms=17.5,
            )
            sample = state.decision_history[-1]

        self.assertEqual(sample["candidate_count"], 2)
        self.assertEqual(sample["immediate_fit_count"], 1)
        self.assertTrue(sample["protected_active"])
        self.assertAlmostEqual(sample["score_gap"], 3.5)

    def test_prometheus_metrics_export_includes_runtime_and_decision_metrics(self) -> None:
        state = ModelServerState()
        with state.lock:
            state.record_request_latency_locked("kueue_admission_advice", 10.0)
            state.record_request_latency_locked("kueue_admission_advice", 30.0)
            state.record_decision_event_locked(
                route_name="kueue_admission_advice",
                timestamp="2026-04-12T01:02:03",
                request_state={"candidates": [{"workload_id": "ns/a", "immediate_fit": True, "provisionable": True}]},
                response={"source": "blocked-guard-kueue", "pair_scores": {"a@x": 5.0}},
                elapsed_ms=22.0,
            )
            metrics_text = state.prometheus_metrics_locked()

        self.assertIn('admirl_requests_total{route="kueue_admission_advice"} 2', metrics_text)
        self.assertIn('admirl_decisions_total{source="blocked-guard-kueue",effective_policy="blocked_guard"} 1', metrics_text)
        self.assertIn("admirl_last_latency_ms 22.000000", metrics_text)
        self.assertIn('admirl_runtime_policy_info{runtime_policy="blocked_guard",effective_policy="blocked_guard",checkpoint_loaded="false"} 1', metrics_text)

    def test_benchmark_snapshot_is_exported_to_runtime_and_prometheus(self) -> None:
        state = ModelServerState()
        with state.lock:
            state.record_benchmark_snapshot_locked(
                {
                    "preset": "kueue-lingjun-gang-elastic-topology",
                    "arm": "learned-elastic-default",
                    "namespace": "demo-ns",
                    "seed": 11,
                    "phase": "running",
                    "active": True,
                    "progress_ratio": 0.25,
                    "elapsed_seconds": 18.0,
                    "jobs_total": 8,
                    "jobs_completed": 2,
                    "jobs_running": 4,
                    "jobs_pending": 2,
                    "avg_gang_wait_seconds": 55.0,
                    "avg_elastic_wait_seconds": 32.0,
                    "head_gang_current_blocked_seconds": 61.0,
                    "throughput_jobs_per_minute": 6.5,
                    "gang_completion_ratio": 0.4,
                }
            )
            snapshot = state.runtime_metrics_snapshot_locked()
            metrics_text = state.prometheus_metrics_locked()

        self.assertTrue(snapshot["benchmark"]["active"])
        self.assertEqual(snapshot["benchmark"]["latest"]["arm"], "learned-elastic-default")
        self.assertIn('admirl_benchmark_run_info{preset="kueue-lingjun-gang-elastic-topology",arm="learned-elastic-default"', metrics_text)
        self.assertIn("admirl_benchmark_jobs_completed 2.000000", metrics_text)
        self.assertIn("admirl_benchmark_head_gang_current_blocked_seconds 61.000000", metrics_text)


if __name__ == "__main__":
    unittest.main()
