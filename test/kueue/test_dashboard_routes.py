from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVER_ROOT = REPO_ROOT / "model_server"
for path in (REPO_ROOT, MODEL_SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_server.admirl_server import create_app
from model_server.admirl_server.state import state


class DashboardRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()
        with state.lock:
            state.last_decision = {
                "timestamp": None,
                "request": None,
                "response": None,
            }
            state.runtime_policy = "blocked_guard"
            state.learned_checkpoint = None
            state.learned_checkpoint_path = ""
            state.reset_runtime_metrics_locked()

    def test_metrics_endpoint(self) -> None:
        with state.lock:
            state.record_request_latency_locked("kueue_admission_advice", 12.0)
            state.record_decision_event_locked(
                route_name="kueue_admission_advice",
                timestamp="2026-04-12T01:02:03",
                request_state={"candidates": [{"workload_id": "ns/a", "immediate_fit": True, "provisionable": True}]},
                response={"source": "blocked-guard-kueue", "pair_scores": {"a@x": 4.0}},
                elapsed_ms=12.0,
            )

        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        text = metrics.get_data(as_text=True)
        self.assertIn("admirl_requests_total", text)
        self.assertIn("admirl_decisions_total", text)

    def test_root_redirects_to_grafana_dashboard(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn(
            "http://127.0.0.1:3000/d/admirl-runtime/admirl-runtime-and-benchmark-operations",
            response.headers["Location"],
        )

    def test_removed_runtime_status_endpoint_returns_404(self) -> None:
        response = self.client.get("/api/runtime-status")
        self.assertEqual(response.status_code, 404)

    def test_benchmark_progress_and_status_endpoints(self) -> None:
        progress = self.client.post(
            "/api/benchmark/progress",
            json={
                "preset": "kueue-lingjun-gang-starvation-cohort",
                "arm": "cohort-learned",
                "namespace": "demo-ns",
                "seed": 7,
                "phase": "running",
                "active": True,
                "progress_ratio": 0.5,
                "jobs_total": 8,
                "jobs_completed": 4,
                "jobs_running": 3,
                "jobs_pending": 1,
                "avg_gang_wait_seconds": 42.0,
                "head_gang_current_blocked_seconds": 55.0,
            },
        )
        self.assertEqual(progress.status_code, 200)

        status = self.client.get("/api/benchmark/status?window=10")
        self.assertEqual(status.status_code, 200)
        payload = status.get_json()
        self.assertTrue(payload["active"])
        self.assertEqual(payload["latest"]["arm"], "cohort-learned")
        self.assertEqual(payload["latest"]["jobs_completed"], 4)


if __name__ == "__main__":
    unittest.main()
