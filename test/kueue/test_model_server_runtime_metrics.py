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
            state.record_request_latency_locked("kueue_admission_order", 20.0)
            snapshot = state.runtime_metrics_snapshot_locked()

        self.assertEqual(snapshot["request_latency_ms"]["kueue_admission_advice"]["count"], 2)
        self.assertAlmostEqual(snapshot["request_latency_ms"]["kueue_admission_advice"]["mean_ms"], 20.0)
        self.assertEqual(snapshot["request_latency_ms"]["kueue_admission_order"]["count"], 1)

    def test_reset_runtime_metrics_clears_samples(self) -> None:
        state = ModelServerState()
        with state.lock:
            state.record_request_latency_locked("kueue_admission_advice", 12.0)
            state.reset_runtime_metrics_locked()
            snapshot = state.runtime_metrics_snapshot_locked()

        self.assertEqual(snapshot["request_latency_ms"], {})


if __name__ == "__main__":
    unittest.main()
