from __future__ import annotations

import unittest

from test.kueue.run_live_kueue_source_compare import (
    SourceProfile,
    _default_candidate_arm_for_preset,
    _requires_checkpoint,
    _matrix_command,
    _profile_env,
    aggregate_comparison_runs,
)


class RunLiveKueueSourceCompareTests(unittest.TestCase):
    def test_profile_env_sets_source_controls(self) -> None:
        profile = SourceProfile(
            label="upstream-default",
            arm="stock-best-effort-default",
            source_mode="git",
            source_dir=None,
            git_url="https://github.com/kubernetes-sigs/kueue.git",
            git_ref="v0.15.0",
            checkpoint_path=None,
        )
        env = _profile_env(profile)
        self.assertEqual(env["ADMIRL_KUEUE_SOURCE_MODE"], "git")
        self.assertEqual(env["ADMIRL_KUEUE_GIT_URL"], "https://github.com/kubernetes-sigs/kueue.git")
        self.assertEqual(env["ADMIRL_KUEUE_GIT_REF"], "v0.15.0")
        self.assertNotIn("ADMIRL_KUEUE_SOURCE_DIR", env)

    def test_matrix_command_includes_profile_arm_and_checkpoint(self) -> None:
        profile = SourceProfile(
            label="ours",
            arm="learned-best-effort-default",
            source_mode="local",
            source_dir="/Users/abhigyan/Desktop/kueue",
            git_url=None,
            git_ref=None,
            checkpoint_path="/tmp/model.pt",
            runtime_policy="blocked_guard",
        )
        cmd = _matrix_command(
            profile=profile,
            workload_preset="kueue-lingjun-gang-starvation-cohort",
            seed=7,
            num_jobs=8,
            arrival_span=120.0,
            trace_split="test",
            trace_train_fraction=0.75,
            runtime_scale=120.0,
            time_scale=10.0,
            output_root="/tmp/out",
        )
        self.assertIn("learned-best-effort-default", cmd)
        self.assertIn("/tmp/model.pt", cmd)
        self.assertIn("--runtime-policy-override", cmd)
        self.assertIn("blocked_guard", cmd)
        self.assertIn("test/kueue/run_live_kueue_matrix.py", cmd)

    def test_default_candidate_arm_prefers_elastic_arm_for_elastic_presets(self) -> None:
        self.assertEqual(
            _default_candidate_arm_for_preset("kueue-lingjun-gang-elastic-topology"),
            "learned-elastic-default",
        )
        self.assertEqual(
            _default_candidate_arm_for_preset("kueue-lingjun-gang-starvation-cohort"),
            "learned-best-effort-default",
        )

    def test_requires_checkpoint_for_learned_arms(self) -> None:
        self.assertTrue(_requires_checkpoint("learned-elastic-default"))
        self.assertTrue(_requires_checkpoint("learned-best-effort-default"))
        self.assertFalse(_requires_checkpoint("stock-best-effort-default"))
        self.assertFalse(_requires_checkpoint("heuristic-elastic-default"))

    def test_aggregate_comparison_runs_tracks_mean_and_wins(self) -> None:
        aggregate = aggregate_comparison_runs(
            [
                {
                    "metrics": {
                        "head_gang_blocked_seconds": {
                            "baseline": 60.0,
                            "candidate": 12.0,
                            "improvement_fraction": 0.8,
                            "winner": "candidate",
                        },
                        "throughput_jobs_per_minute": {
                            "baseline": 5.0,
                            "candidate": 6.0,
                            "improvement_fraction": 0.2,
                            "winner": "candidate",
                        },
                    }
                },
                {
                    "metrics": {
                        "head_gang_blocked_seconds": {
                            "baseline": 40.0,
                            "candidate": 20.0,
                            "improvement_fraction": 0.5,
                            "winner": "candidate",
                        },
                        "throughput_jobs_per_minute": {
                            "baseline": 4.0,
                            "candidate": 3.0,
                            "improvement_fraction": -0.25,
                            "winner": "baseline",
                        },
                    }
                },
            ]
        )
        self.assertAlmostEqual(aggregate["metrics"]["head_gang_blocked_seconds"]["baseline_mean"], 50.0)
        self.assertAlmostEqual(aggregate["metrics"]["head_gang_blocked_seconds"]["candidate_mean"], 16.0)
        self.assertAlmostEqual(aggregate["metrics"]["head_gang_blocked_seconds"]["improvement_fraction_mean"], 0.65)
        self.assertEqual(aggregate["metrics"]["head_gang_blocked_seconds"]["candidate_wins"], 2)
        self.assertEqual(aggregate["metrics"]["throughput_jobs_per_minute"]["candidate_wins"], 1)


if __name__ == "__main__":
    unittest.main()
