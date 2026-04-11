from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVER_ROOT = REPO_ROOT / "model_server"
for path in (REPO_ROOT, MODEL_SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_server.kueue_rl.training import (
    _checkpoint_primary_metric_for_preset,
    _checkpoint_signature_for_preset,
)


def _stats(**overrides: float) -> dict:
    base = {
        "val_p95_workload_wait_seconds": 100.0,
        "val_gang_admission_ratio": 1.0,
        "val_p95_job_completion_seconds": 200.0,
        "val_topology_hit_rate": 1.0,
        "val_flavor_head_blocking_seconds": 20.0,
        "val_idle_quota_while_blocked": 1.0,
        "val_avg_small_wait_seconds": 25.0,
        "val_makespan_seconds": 60.0,
        "val_throughput_jobs_per_minute": 8.0,
        "val_fair_share_violation_count": 0.0,
        "val_avg_provisioning_delay_seconds": 30.0,
        "val_avg_fragmentation": 0.2,
        "val_avg_job_completion_seconds": 180.0,
        "val_avg_gang_wait_seconds": 90.0,
        "val_avg_gang_completion_seconds": 140.0,
        "val_p95_gang_completion_seconds": 220.0,
        "val_avg_small_completion_seconds": 120.0,
        "val_avg_topology_aware_wait_seconds": 80.0,
        "val_avg_topology_aware_completion_seconds": 135.0,
        "val_avg_critical_wait_seconds": 85.0,
        "val_avg_critical_completion_seconds": 130.0,
        "val_avg_elastic_wait_seconds": 70.0,
        "val_avg_elastic_completion_seconds": 120.0,
        "val_elastic_completion_ratio": 0.8,
    }
    base.update(overrides)
    return base


class KueueTrainingSelectionTests(unittest.TestCase):
    def test_starvation_primary_metric_penalizes_idle_quota_while_blocked(self) -> None:
        tighter = _checkpoint_primary_metric_for_preset("kueue-lingjun-gang-starvation", _stats(val_idle_quota_while_blocked=0.0))
        leakier = _checkpoint_primary_metric_for_preset("kueue-lingjun-gang-starvation", _stats(val_idle_quota_while_blocked=2.0))
        self.assertLess(tighter, leakier)

    def test_starvation_signature_prefers_lower_small_job_wait_when_blocking_ties(self) -> None:
        better = _checkpoint_signature_for_preset(
            "kueue-lingjun-gang-starvation-cohort",
            _stats(val_avg_small_wait_seconds=18.0),
        )
        worse = _checkpoint_signature_for_preset(
            "kueue-lingjun-gang-starvation-cohort",
            _stats(val_avg_small_wait_seconds=26.0),
        )
        self.assertLess(better, worse)

    def test_starvation_primary_metric_rewards_better_throughput_when_blocking_ties(self) -> None:
        faster = _checkpoint_primary_metric_for_preset(
            "kueue-lingjun-gang-starvation-cohort",
            _stats(val_throughput_jobs_per_minute=9.0),
        )
        slower = _checkpoint_primary_metric_for_preset(
            "kueue-lingjun-gang-starvation-cohort",
            _stats(val_throughput_jobs_per_minute=7.0),
        )
        self.assertLess(faster, slower)

    def test_topology_provisioning_primary_metric_prefers_faster_critical_jobs(self) -> None:
        better = _checkpoint_primary_metric_for_preset(
            "kueue-lingjun-gang-topology-provisioning",
            _stats(
                val_avg_critical_completion_seconds=70.0,
                val_avg_critical_wait_seconds=35.0,
                val_avg_topology_aware_completion_seconds=90.0,
                val_topology_hit_rate=1.0,
                val_avg_provisioning_delay_seconds=15.0,
                val_throughput_jobs_per_minute=8.5,
            ),
        )
        worse = _checkpoint_primary_metric_for_preset(
            "kueue-lingjun-gang-topology-provisioning",
            _stats(
                val_avg_critical_completion_seconds=180.0,
                val_avg_critical_wait_seconds=120.0,
                val_avg_topology_aware_completion_seconds=220.0,
                val_topology_hit_rate=0.5,
                val_avg_provisioning_delay_seconds=45.0,
                val_throughput_jobs_per_minute=6.0,
            ),
        )
        self.assertLess(better, worse)

    def test_elastic_profile_cohort_primary_metric_prefers_faster_elastic_completion(self) -> None:
        better = _checkpoint_primary_metric_for_preset(
            "kueue-lingjun-gang-elastic-profile-cohort",
            _stats(
                val_avg_elastic_completion_seconds=70.0,
                val_avg_elastic_wait_seconds=25.0,
                val_avg_gang_completion_seconds=95.0,
                val_avg_gang_wait_seconds=30.0,
                val_makespan_seconds=40.0,
                val_throughput_jobs_per_minute=12.0,
                val_elastic_completion_ratio=1.0,
            ),
        )
        worse = _checkpoint_primary_metric_for_preset(
            "kueue-lingjun-gang-elastic-profile-cohort",
            _stats(
                val_avg_elastic_completion_seconds=150.0,
                val_avg_elastic_wait_seconds=90.0,
                val_avg_gang_completion_seconds=130.0,
                val_avg_gang_wait_seconds=60.0,
                val_makespan_seconds=60.0,
                val_throughput_jobs_per_minute=7.0,
                val_elastic_completion_ratio=0.5,
            ),
        )
        self.assertLess(better, worse)


if __name__ == "__main__":
    unittest.main()
