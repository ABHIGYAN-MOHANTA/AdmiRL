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


class DecisionRoutesTests(unittest.TestCase):
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

    def test_admission_advice_route_records_last_decision(self) -> None:
        response = self.client.post(
            "/api/kueue/admission-advice",
            json={
                "request_mode": "kueue-admission",
                "candidates": [
                    {
                        "action_id": "demo/job-a@rf-4gpu-a",
                        "workload_id": "demo/job-a",
                        "flavor_name": "rf-4gpu-a",
                        "queue_name": "lq-small",
                        "cluster_queue": "cq-small",
                        "queue_class": "small",
                        "runtime_seconds": 600,
                        "wait_seconds": 10,
                        "worker_count": 1,
                        "total_gpu": 4,
                        "per_worker_gpu": 4,
                        "immediate_fit": True,
                        "provisionable": True,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("ranked_workloads", payload)
        with state.lock:
            self.assertIsNotNone(state.last_decision["timestamp"])
            self.assertEqual(
                state.last_decision["request"]["candidates"][0]["workload_id"],
                "demo/job-a",
            )

    def test_removed_admission_order_alias_returns_404(self) -> None:
        response = self.client.post("/api/kueue/admission-order", json={"candidates": []})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
