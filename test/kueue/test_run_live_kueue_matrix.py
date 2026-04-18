from __future__ import annotations

import os
import subprocess
import unittest
from unittest import mock
from tempfile import TemporaryDirectory
from pathlib import Path
import json

from test.kueue import run_live_kueue_matrix as live_matrix
from test.kueue.run_live_kueue_matrix import (
    _augment_arm_metrics,
    _augment_result_pack,
    _build_benchmark_progress_payload,
    _count_small_job_bypass_while_gang_pending,
    _format_build_date,
    _kueue_image_name,
    _load_existing_result_pack,
    _runtime_flavor_for_node,
)


class RunLiveKueueMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        live_matrix._KUEUE_SOURCE_DIR_CACHE = None
        live_matrix._KUEUE_IMAGE_CACHE = None
        live_matrix._KUEUE_BUILD_DATE_CACHE = None

    def test_kueue_image_name_uses_source_hash_prefix(self) -> None:
        self.assertEqual(_kueue_image_name("0123456789abcdef"), "kueue-admirl:0123456789ab")

    def test_format_build_date_is_utc_iso(self) -> None:
        self.assertEqual(_format_build_date(0.0), "1970-01-01T00:00:00Z")

    def test_kueue_source_dir_prefers_explicit_override(self) -> None:
        with TemporaryDirectory() as tmpdir:
            override_dir = Path(tmpdir) / "custom-kueue"
            override_dir.mkdir()
            with mock.patch.dict(os.environ, {"ADMIRL_KUEUE_SOURCE_DIR": str(override_dir)}, clear=False):
                source_dir = live_matrix.kueue_source_dir()
        self.assertEqual(source_dir, override_dir.resolve())

    def test_prepare_kueue_fork_source_clones_configured_branch(self) -> None:
        commands: list[tuple[list[str], Path]] = []

        def fake_run(cmd: list[str], *, cwd: Path, **_: object) -> subprocess.CompletedProcess[str]:
            commands.append((cmd, cwd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "fork-cache"
            env = {
                "ADMIRL_KUEUE_GIT_URL": "https://example.com/custom-kueue.git",
                "ADMIRL_KUEUE_GIT_REF": "paper-branch",
                "ADMIRL_KUEUE_CACHE_DIR": str(cache_dir),
            }
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(live_matrix, "_run", side_effect=fake_run):
                source_dir = live_matrix._prepare_kueue_fork_source()

        self.assertEqual(source_dir, cache_dir.resolve())
        self.assertEqual(
            commands[0][0],
            ["git", "clone", "--branch", "paper-branch", "--single-branch", "https://example.com/custom-kueue.git", str(cache_dir.resolve())],
        )
        self.assertEqual(commands[0][1], cache_dir.resolve().parent)

    def test_kueue_source_dir_auto_falls_back_to_local_tree(self) -> None:
        with TemporaryDirectory() as tmpdir:
            local_kueue = Path(tmpdir) / "desktop-kueue"
            local_kueue.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"ADMIRL_KUEUE_SOURCE_MODE": "auto"}, clear=False), \
                mock.patch.object(live_matrix, "LOCAL_KUEUE_SOURCE_DIR", local_kueue), \
                mock.patch.object(live_matrix, "_prepare_kueue_fork_source", side_effect=RuntimeError("offline")):
                source_dir = live_matrix.kueue_source_dir()

        self.assertEqual(source_dir, local_kueue.resolve())

    def test_kueue_source_metadata_reads_git_fields_when_present(self) -> None:
        def fake_run(cmd: list[str], *, cwd: Path, **_: object) -> subprocess.CompletedProcess[str]:
            joined = " ".join(cmd)
            if "remote get-url origin" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="https://example.com/kueue.git\n", stderr="")
            if "rev-parse HEAD" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")
            if "rev-parse --abbrev-ref HEAD" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="paper-branch\n", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        with TemporaryDirectory() as tmpdir, mock.patch.object(live_matrix, "_run", side_effect=fake_run):
            metadata = live_matrix._kueue_source_metadata(Path(tmpdir))

        self.assertEqual(metadata["git_url"], "https://example.com/kueue.git")
        self.assertEqual(metadata["git_commit"], "deadbeef")
        self.assertEqual(metadata["git_ref"], "paper-branch")

    def test_bypass_counter_uses_relative_admission_times(self) -> None:
        group_rows = [
            {
                "queue_class": "gang",
                "arrival_time": 1.9,
                "wait_seconds": 8.0,
            },
            {
                "queue_class": "small",
                "arrival_time": 3.2,
                "wait_seconds": 17.0,
            },
            {
                "queue_class": "small",
                "arrival_time": 4.4,
                "wait_seconds": 30.0,
            },
        ]

        bypass_count = _count_small_job_bypass_while_gang_pending(
            group_rows,
            head_gang_arrival=1.9,
            head_gang_admit=9.9,
        )

        self.assertEqual(bypass_count, 0)

    def test_runtime_flavor_for_node_matches_annotation_flavor_suffix(self) -> None:
        annotations = {
            "admirl.ai/scaled-runtime-rf-4gpu-a": "30",
            "admirl.ai/scaled-runtime-rf-4gpu-c": "8",
        }

        self.assertEqual(
            _runtime_flavor_for_node("kwok-gpu-rf-4gpu-c-004", annotations),
            "rf-4gpu-c",
        )
        self.assertEqual(
            _runtime_flavor_for_node("kwok-gpu-rf-4gpu-a-000", annotations),
            "rf-4gpu-a",
        )

    def test_topology_provisioner_delay_is_relative_to_first_scaled_arrival(self) -> None:
        launched = []

        class DummyProc:
            def poll(self):
                return 0

        def fake_popen(cmd, **kwargs):
            launched.append((cmd, kwargs))
            return DummyProc()

        workloads = [
            {"arrival_time": 100.0},
            {"arrival_time": 120.0},
        ]
        with TemporaryDirectory() as tmpdir, mock.patch.object(subprocess, "Popen", side_effect=fake_popen):
            procs = live_matrix.start_provisioners(
                Path(tmpdir),
                "kueue-lingjun-gang-topology-provisioning",
                "training-gang-topology-provisioning",
                workloads=workloads,
                time_scale=10.0,
            )
            for _, log_file in procs:
                log_file.close()

        self.assertEqual(len(launched), 1)
        cmd = launched[0][0]
        delay_index = cmd.index("--delay-seconds")
        self.assertEqual(cmd[delay_index + 1], "18.0")

    def test_kwok_layout_map_includes_broader_elastic_profile_preset(self) -> None:
        self.assertEqual(
            live_matrix.PRESET_TO_KWOK_LAYOUT["kueue-lingjun-gang-elastic-profile-cohort"],
            "training-gang-elastic-profile-cohort",
        )

    def test_cleanup_kueue_test_state_deletes_spare_nodes_first(self) -> None:
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(live_matrix, "_run", side_effect=fake_run), mock.patch.object(
            live_matrix,
            "_kubectl_json_optional",
            return_value={"items": []},
        ), mock.patch.object(live_matrix.time, "sleep", return_value=None):
            live_matrix.cleanup_kueue_test_state()

        self.assertTrue(commands)
        self.assertEqual(
            commands[0],
            ["kubectl", "delete", "node", "-l", "admirl.ai/spare=true", "--ignore-not-found=true", "--wait=false"],
        )

    def test_bypass_counter_counts_only_earlier_small_admissions(self) -> None:
        group_rows = [
            {
                "queue_class": "gang",
                "arrival_time": 2.0,
                "wait_seconds": 10.0,
            },
            {
                "queue_class": "small",
                "arrival_time": 3.0,
                "wait_seconds": 2.0,
            },
            {
                "queue_class": "small",
                "arrival_time": 4.0,
                "wait_seconds": 8.0,
            },
            {
                "queue_class": "small",
                "arrival_time": 5.0,
                "wait_seconds": 7.0,
            },
        ]

        bypass_count = _count_small_job_bypass_while_gang_pending(
            group_rows,
            head_gang_arrival=2.0,
            head_gang_admit=12.0,
        )

        self.assertEqual(bypass_count, 1)

    def test_build_benchmark_progress_payload_extracts_live_story_metrics(self) -> None:
        payload = _build_benchmark_progress_payload(
            {
                "job_completion_ratio": 0.5,
                "jobs_total": 8,
                "jobs_completed": 4,
                "jobs_running": 3,
                "jobs_pending": 1,
                "avg_gang_wait_seconds": 42.0,
                "avg_small_wait_seconds": 9.0,
                "avg_elastic_wait_seconds": 18.0,
                "head_gang_blocked_seconds": 55.0,
                "head_gang_current_blocked_seconds": 61.0,
                "throughput_jobs_per_minute": 7.5,
                "gang_completion_ratio": 0.5,
                "elastic_completion_ratio": 0.75,
                "topology_hit_rate": 1.0,
                "small_job_bypass_count_while_gang_pending": 2,
                "head_gang_workload": "gang-a",
            },
            workload_preset="kueue-lingjun-gang-starvation-cohort",
            arm_name="cohort-learned",
            seed=7,
            namespace="demo-ns",
            phase="running",
            elapsed_seconds=30.0,
            timeout_seconds=300.0,
            active=True,
        )

        self.assertEqual(payload["preset"], "kueue-lingjun-gang-starvation-cohort")
        self.assertEqual(payload["arm"], "cohort-learned")
        self.assertEqual(payload["jobs_running"], 3)
        self.assertEqual(payload["head_gang_current_blocked_seconds"], 61.0)
        self.assertEqual(payload["small_job_bypass_count"], 2.0)

    def test_bypass_counter_ignores_disjoint_flavor_noise(self) -> None:
        group_rows = [
            {
                "queue_class": "gang",
                "scaled_arrival_time": 2.0,
                "wait_seconds": 10.0,
                "candidate_flavors": ["rf-4gpu-a"],
            },
            {
                "queue_class": "small",
                "scaled_arrival_time": 3.0,
                "wait_seconds": 2.0,
                "candidate_flavors": ["rf-2gpu-b"],
            },
            {
                "queue_class": "small",
                "scaled_arrival_time": 4.0,
                "wait_seconds": 2.0,
                "candidate_flavors": ["rf-4gpu-a"],
            },
        ]

        bypass_count = _count_small_job_bypass_while_gang_pending(
            group_rows,
            head_gang_arrival=2.0,
            head_gang_admit=12.0,
            head_gang_candidate_flavors=["rf-4gpu-a"],
        )

        self.assertEqual(bypass_count, 1)

    def test_augment_arm_metrics_tracks_actual_head_flavor_admissions(self) -> None:
        metrics = _augment_arm_metrics(
            {
                "jobs_total": 3,
                "jobs_completed": 3,
                "topology_hit_rate": 0.0,
                "raw_groups": [
                    {
                        "workload_name": "gang-b",
                        "queue_class": "gang",
                        "candidate_flavors": ["rf-4gpu-b"],
                        "admitted_flavors": [],
                        "total_gpu": 8,
                        "runtime_seconds": 100.0,
                        "scaled_arrival_time": 1.0,
                        "wait_seconds": 10.0,
                        "completion_seconds": 20.0,
                        "completed": True,
                    },
                    {
                        "workload_name": "small-on-b",
                        "queue_class": "small",
                        "candidate_flavors": ["rf-4gpu-b", "rf-4gpu-a"],
                        "admitted_flavors": ["rf-4gpu-b"],
                        "total_gpu": 4,
                        "runtime_seconds": 50.0,
                        "scaled_arrival_time": 2.0,
                        "wait_seconds": 2.0,
                        "completion_seconds": 8.0,
                        "completed": True,
                    },
                    {
                        "workload_name": "small-on-a",
                        "queue_class": "small",
                        "candidate_flavors": ["rf-4gpu-b", "rf-4gpu-a"],
                        "admitted_flavors": ["rf-4gpu-a"],
                        "total_gpu": 4,
                        "runtime_seconds": 50.0,
                        "scaled_arrival_time": 3.0,
                        "wait_seconds": 2.0,
                        "completion_seconds": 8.0,
                        "completed": True,
                    },
                ],
            }
        )

        self.assertEqual(metrics["small_job_bypass_count_while_gang_pending"], 2)
        self.assertEqual(metrics["small_job_head_flavor_admissions_while_gang_pending"], 1)
        self.assertEqual(metrics["small_job_head_flavor_gpu_while_gang_pending"], 4.0)

    def test_augment_arm_metrics_adds_paper_ready_latency_and_fairness_metrics(self) -> None:
        metrics = _augment_arm_metrics(
            {
                "jobs_total": 3,
                "jobs_completed": 3,
                "topology_hit_rate": 1.0,
                "raw_groups": [
                    {
                        "workload_name": "gang-a",
                        "queue_class": "gang",
                        "total_gpu": 8,
                        "runtime_seconds": 100.0,
                        "scaled_arrival_time": 1.0,
                        "wait_seconds": 30.0,
                        "completion_seconds": 45.0,
                        "completed": True,
                    },
                    {
                        "workload_name": "small-a",
                        "queue_class": "small",
                        "total_gpu": 4,
                        "runtime_seconds": 60.0,
                        "scaled_arrival_time": 2.0,
                        "wait_seconds": 10.0,
                        "completion_seconds": 22.0,
                        "completed": True,
                    },
                    {
                        "workload_name": "small-b",
                        "queue_class": "small",
                        "total_gpu": 4,
                        "runtime_seconds": 60.0,
                        "scaled_arrival_time": 3.0,
                        "wait_seconds": 12.0,
                        "completion_seconds": 24.0,
                        "completed": True,
                    },
                ],
            }
        )

        self.assertEqual(metrics["head_gang_workload"], "gang-a")
        self.assertEqual(metrics["p50_gang_wait_seconds"], 30.0)
        self.assertEqual(metrics["avg_small_wait_seconds"], 11.0)
        self.assertAlmostEqual(metrics["avg_gang_minus_small_wait_seconds"], 19.0)
        self.assertAlmostEqual(metrics["avg_gang_to_small_wait_ratio"], 30.0 / 11.0)
        self.assertEqual(metrics["small_job_bypass_count_while_gang_pending"], 2)
        self.assertEqual(metrics["small_job_bypass_fraction_while_gang_pending"], 1.0)
        self.assertEqual(metrics["small_job_bypass_gpu_while_gang_pending"], 8.0)
        self.assertEqual(metrics["makespan_seconds"], 45.0)
        self.assertAlmostEqual(metrics["throughput_jobs_per_minute"], 3.0 * 60.0 / 45.0)
        self.assertAlmostEqual(metrics["throughput_gpu_per_minute"], 16.0 * 60.0 / 45.0)
        self.assertAlmostEqual(metrics["head_gang_wait_to_runtime_ratio"], 0.3)
        self.assertAlmostEqual(metrics["gpu_weighted_gang_slowdown_vs_isolated"], 45.0 / 100.0)

    def test_augment_arm_metrics_ignores_disjoint_bypass_noise(self) -> None:
        metrics = _augment_arm_metrics(
            {
                "jobs_total": 3,
                "jobs_completed": 3,
                "topology_hit_rate": 0.0,
                "raw_groups": [
                    {
                        "workload_name": "gang-a",
                        "queue_class": "gang",
                        "candidate_flavors": ["rf-4gpu-a"],
                        "total_gpu": 8,
                        "runtime_seconds": 100.0,
                        "scaled_arrival_time": 1.0,
                        "wait_seconds": 10.0,
                        "completion_seconds": 20.0,
                        "completed": True,
                    },
                    {
                        "workload_name": "noise-b",
                        "queue_class": "small",
                        "candidate_flavors": ["rf-2gpu-b"],
                        "total_gpu": 2,
                        "runtime_seconds": 60.0,
                        "scaled_arrival_time": 2.0,
                        "wait_seconds": 3.0,
                        "completion_seconds": 9.0,
                        "completed": True,
                    },
                    {
                        "workload_name": "small-a",
                        "queue_class": "small",
                        "candidate_flavors": ["rf-4gpu-a"],
                        "total_gpu": 4,
                        "runtime_seconds": 60.0,
                        "scaled_arrival_time": 3.0,
                        "wait_seconds": 12.0,
                        "completion_seconds": 18.0,
                        "completed": True,
                    },
                ],
            }
        )

        self.assertEqual(metrics["small_jobs_arriving_while_head_gang_pending"], 2)
        self.assertEqual(metrics["small_job_bypass_count_while_gang_pending"], 0)
        self.assertEqual(metrics["small_job_bypass_fraction_while_gang_pending"], 0.0)

    def test_augment_result_pack_adds_stock_vs_candidate_comparison(self) -> None:
        result_pack = _augment_result_pack(
            {
                "workload_preset": "kueue-lingjun-gang-starvation-cohort",
                "arms": {
                    "stock-best-effort-default": {
                        "jobs_total": 2,
                        "jobs_completed": 2,
                        "topology_hit_rate": 1.0,
                        "raw_groups": [
                            {
                                "workload_name": "gang-a",
                                "queue_class": "gang",
                                "total_gpu": 8,
                                "runtime_seconds": 100.0,
                                "scaled_arrival_time": 1.0,
                                "wait_seconds": 80.0,
                                "completion_seconds": 95.0,
                                "completed": True,
                            },
                            {
                                "workload_name": "small-a",
                                "queue_class": "small",
                                "total_gpu": 4,
                                "runtime_seconds": 60.0,
                                "scaled_arrival_time": 2.0,
                                "wait_seconds": 5.0,
                                "completion_seconds": 18.0,
                                "completed": True,
                            },
                        ],
                    },
                    "learned-best-effort-default": {
                        "jobs_total": 2,
                        "jobs_completed": 2,
                        "topology_hit_rate": 1.0,
                        "raw_groups": [
                            {
                                "workload_name": "gang-a",
                                "queue_class": "gang",
                                "total_gpu": 8,
                                "runtime_seconds": 100.0,
                                "scaled_arrival_time": 1.0,
                                "wait_seconds": 20.0,
                                "completion_seconds": 35.0,
                                "completed": True,
                            },
                            {
                                "workload_name": "small-a",
                                "queue_class": "small",
                                "total_gpu": 4,
                                "runtime_seconds": 60.0,
                                "scaled_arrival_time": 2.0,
                                "wait_seconds": 8.0,
                                "completion_seconds": 21.0,
                                "completed": True,
                            },
                        ],
                    },
                },
            }
        )

        comparison = result_pack["comparisons"]["stock-best-effort-default__vs__learned-best-effort-default"]
        headline = comparison["metrics"]["head_gang_blocked_seconds"]
        throughput = comparison["metrics"]["throughput_jobs_per_minute"]

        self.assertAlmostEqual(headline["baseline"], 80.0)
        self.assertAlmostEqual(headline["candidate"], 20.0)
        self.assertAlmostEqual(headline["improvement_fraction"], 0.75)
        self.assertEqual(headline["winner"], "candidate")
        self.assertEqual(throughput["direction"], "higher")
        self.assertEqual(throughput["winner"], "candidate")
        self.assertIn("stock-best-effort-default__vs__learned-best-effort-default", result_pack["paper_summary"])

    def test_load_existing_result_pack_restores_candidate_flavors_from_meta(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            arm_dir = root / "stock-best-effort-default"
            arm_dir.mkdir(parents=True)
            (arm_dir / "live-results.json").write_text(
                json.dumps(
                    {
                        "raw_groups": [
                            {
                                "workload_name": "starve-head-gang",
                                "queue_class": "gang",
                                "scaled_arrival_time": 0.04,
                                "wait_seconds": 14.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (arm_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "workloads": [
                            {
                                "workload_id": "starve-head-gang",
                                "candidate_flavors": ["rf-4gpu-a"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = _load_existing_result_pack(root)
            row = loaded["arms"]["stock-best-effort-default"]["raw_groups"][0]
            self.assertEqual(row["candidate_flavors"], ["rf-4gpu-a"])


if __name__ == "__main__":
    unittest.main()
