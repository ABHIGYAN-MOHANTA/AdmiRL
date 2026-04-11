from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SERVER_ROOT = REPO_ROOT / "model_server"
for path in (REPO_ROOT, MODEL_SERVER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_server.kueue_rl.cluster import NodeState
from model_server.kueue_rl.kueue_admission import (
    CandidateAction,
    blocked_head_context,
    build_kueue_workloads,
    heuristic_candidate_priority,
    kueue_state_vector,
    workload_profile_prefer_c_scalar,
    workload_runtime_multiplier,
)
from model_server.admirl_server.kueue_runtime import (
    _apply_safe_immediate_override,
    _has_blocked_bypass_risk,
    _runtime_time_from_request,
    _sorted_runtime_candidates,
)
from test.kueue.manifests import pod_group_docs


def _candidate(
    *,
    action_id: str,
    workload_id: str,
    flavor_name: str,
    queue_class: str,
    immediate_fit: bool,
    total_gpu: int,
    worker_count: int,
) -> CandidateAction:
    return CandidateAction(
        action_id=action_id,
        workload_id=workload_id,
        flavor_name=flavor_name,
        queue_name="lq-test",
        cluster_queue="training-cluster-queue",
        fairshare_group="test",
        priority=8 if queue_class == "gang" else 4,
        wait_seconds=30.0 if workload_id == "head" else 10.0,
        runtime_seconds=600.0,
        worker_count=worker_count,
        total_gpu=total_gpu,
        per_worker_gpu=max(1, total_gpu // max(worker_count, 1)),
        topology_aware=False,
        topology_preference="",
        flavor_domain=flavor_name.rsplit("-", 1)[-1].upper(),
        immediate_fit=immediate_fit,
        provisionable=not immediate_fit,
        available_gpu=4 if not immediate_fit else total_gpu,
        total_gpu_capacity=max(8, total_gpu),
        fairshare_debt=0.0,
        requeue_count=0,
        queue_class=queue_class,
        flavor_gpu_size=4,
        oversize_gpu=0,
        competing_older_pressure=0.0,
        elastic_enabled=False,
        min_worker_count=worker_count,
        preferred_worker_count=worker_count,
        max_worker_count=worker_count,
        scale_tag="fixed",
        scale_fraction=1.0,
    )


class KueuePresetHardeningTest(unittest.TestCase):
    def test_gang_starvation_preset_has_same_cq_bypass_window(self):
        workloads = build_kueue_workloads(
            seed=7,
            num_jobs=8,
            arrival_span=120.0,
            workload_preset="kueue-lingjun-gang-starvation",
        )

        ids = {item.workload_id for item in workloads}
        self.assertIn("starve-blocker-0", ids)
        self.assertIn("starve-head-gang", ids)
        self.assertNotIn("starve-blocker-1", ids)

        head = next(item for item in workloads if item.workload_id == "starve-head-gang")
        bypass = sorted(
            (item for item in workloads if item.workload_id.startswith("starve-bypass-a-")),
            key=lambda item: item.workload_id,
        )
        self.assertGreaterEqual(len(bypass), 4)
        self.assertEqual(head.cluster_queue, "training-cluster-queue")
        self.assertEqual(head.candidate_flavors, ("rf-4gpu-a",))
        self.assertEqual(head.worker_count, 2)
        self.assertEqual(head.per_worker_gpu, 4)
        self.assertTrue(all(item.cluster_queue == "training-cluster-queue" for item in bypass))
        self.assertTrue(all(item.candidate_flavors == ("rf-4gpu-a",) for item in bypass))
        self.assertTrue(all(item.arrival_time > head.arrival_time for item in bypass))

    def test_gang_starvation_cohort_preset_uses_sibling_cluster_queues(self):
        workloads = build_kueue_workloads(
            seed=7,
            num_jobs=8,
            arrival_span=120.0,
            workload_preset="kueue-lingjun-gang-starvation-cohort",
        )

        by_id = {item.workload_id: item for item in workloads}
        self.assertEqual(by_id["cohort-head-gang"].cluster_queue, "training-cluster-queue-gang")
        bypass = [item for item in workloads if item.workload_id.startswith("cohort-small-a-")]
        self.assertTrue(all(item.cluster_queue == "training-cluster-queue-small" for item in bypass))

    def test_topology_provisioning_preset_creates_borrow_or_fallback_choice(self):
        workloads = build_kueue_workloads(
            seed=7,
            num_jobs=6,
            arrival_span=120.0,
            workload_preset="kueue-lingjun-gang-topology-provisioning",
        )

        by_id = {item.workload_id: item for item in workloads}
        for workload_id in ("etp-blocker-c", "0-etp-critical-c-head", "1-etp-flex-c-0", "2-etp-b-gang", "3-etp-noise-b-0", "4-etp-noise-b-1"):
            self.assertIn(workload_id, by_id)

        blocker = by_id["etp-blocker-c"]
        critical = by_id["0-etp-critical-c-head"]
        flex = by_id["1-etp-flex-c-0"]
        self.assertEqual(blocker.cluster_queue, "training-cluster-queue-gang")
        self.assertEqual(blocker.queue_class, "small")
        self.assertEqual(blocker.candidate_flavors, ("rf-4gpu-c",))
        self.assertEqual(critical.cluster_queue, "training-cluster-queue-gang")
        self.assertEqual(critical.candidate_flavors, ("rf-4gpu-c",))
        self.assertEqual(critical.worker_count, 2)
        self.assertTrue(critical.topology_aware)
        self.assertEqual(critical.topology_preference, "C")
        self.assertEqual(flex.cluster_queue, "training-cluster-queue-small")
        self.assertEqual(flex.queue_class, "small")
        self.assertEqual(flex.worker_count, 1)
        self.assertEqual(flex.candidate_flavors, ("rf-4gpu-c", "rf-4gpu-a"))
        self.assertTrue(flex.topology_aware)
        self.assertEqual(flex.topology_preference, "C")
        self.assertGreater(workload_runtime_multiplier(flex.workload_id, "rf-4gpu-a"), 1.0)

    def test_elastic_topology_preset_exposes_scale_bounds(self):
        workloads = build_kueue_workloads(
            seed=7,
            num_jobs=6,
            arrival_span=120.0,
            workload_preset="kueue-lingjun-gang-elastic-topology",
        )

        by_id = {item.workload_id: item for item in workloads}
        flex = by_id["1-eet-flex-elastic"]
        critical = by_id["0-eet-critical-c-head"]
        self.assertTrue(flex.elastic_enabled)
        self.assertEqual(flex.min_worker_count, 1)
        self.assertEqual(flex.preferred_worker_count, 2)
        self.assertEqual(flex.max_worker_count, 2)
        self.assertEqual(flex.candidate_flavors, ("rf-4gpu-c", "rf-4gpu-a"))
        self.assertTrue(critical.topology_aware)
        self.assertEqual(critical.candidate_flavors, ("rf-4gpu-c",))
        # Keep the critical gang arrival ahead of the synthetic spare-C provision
        # event at 8s so the elastic benchmark continues to exercise the blocked
        # topology choice instead of becoming an immediate-fit case.
        self.assertLess(critical.arrival_time, 8.0)
        self.assertGreater(flex.arrival_time, critical.arrival_time)
        self.assertLess(flex.arrival_time, 8.0)
        self.assertEqual(workload_profile_prefer_c_scalar(flex.workload_id), 0.0)
        self.assertGreater(workload_runtime_multiplier(flex.workload_id, "rf-4gpu-a"), 1.0)

    def test_elastic_profile_cohort_preset_mix_contains_multiple_elastic_profiles(self):
        workloads = build_kueue_workloads(
            seed=7,
            num_jobs=11,
            arrival_span=120.0,
            workload_preset="kueue-lingjun-gang-elastic-profile-cohort",
        )

        by_id = {item.workload_id: item for item in workloads}
        critical = by_id["0-eep-critical-c-head"]
        flex_c = by_id["1-eep-flex-elastic-c-0"]
        flex_aux = by_id["2-eep-flex-elastic-c-aux"]
        second_flex_c = by_id["7-eep-flex-elastic-c-1"]
        goodput_c = by_id["8-goodput-c-elastic-0"]
        goodput_a = by_id["9-goodput-a-elastic-0"]

        self.assertTrue(critical.topology_aware)
        self.assertEqual(critical.candidate_flavors, ("rf-4gpu-c",))
        self.assertTrue(flex_c.elastic_enabled)
        self.assertEqual(flex_c.cluster_queue, "training-cluster-queue-flex")
        self.assertEqual(flex_c.candidate_flavors, ("rf-4gpu-c", "rf-4gpu-a"))
        self.assertEqual(flex_c.min_worker_count, 1)
        self.assertEqual(flex_c.max_worker_count, 2)
        self.assertTrue(flex_aux.elastic_enabled)
        self.assertEqual(flex_aux.candidate_flavors, ("rf-4gpu-c", "rf-4gpu-a"))
        self.assertTrue(flex_aux.topology_aware)
        self.assertGreater(workload_runtime_multiplier(flex_aux.workload_id, "rf-4gpu-a"), 1.0)
        self.assertGreater(workload_runtime_multiplier(second_flex_c.workload_id, "rf-4gpu-a"), 1.0)
        self.assertTrue(goodput_c.elastic_enabled)
        self.assertTrue(goodput_a.elastic_enabled)
        self.assertEqual(goodput_c.candidate_flavors, ("rf-4gpu-a", "rf-4gpu-c"))
        self.assertEqual(goodput_a.candidate_flavors, ("rf-4gpu-a", "rf-4gpu-c"))
        self.assertGreater(workload_runtime_multiplier(goodput_c.workload_id, "rf-4gpu-a"), 1.0)
        self.assertGreater(workload_runtime_multiplier(goodput_a.workload_id, "rf-4gpu-c"), 1.0)

    def test_topology_provisioning_state_vector_distinguishes_flavor_mismatch(self):
        waiting = [
            next(
                item
                for item in build_kueue_workloads(
                    seed=7,
                    num_jobs=6,
                    arrival_span=120.0,
                    workload_preset="kueue-lingjun-gang-topology-provisioning",
                )
                if item.workload_id == "1-etp-flex-c-0"
            )
        ]
        match = _candidate(
            action_id="1-etp-flex-c-0@rf-4gpu-c",
            workload_id="1-etp-flex-c-0",
            flavor_name="rf-4gpu-c",
            queue_class="small",
            immediate_fit=True,
            total_gpu=4,
            worker_count=1,
        )
        mismatch = _candidate(
            action_id="1-etp-flex-c-0@rf-4gpu-a",
            workload_id="1-etp-flex-c-0",
            flavor_name="rf-4gpu-a",
            queue_class="small",
            immediate_fit=True,
            total_gpu=4,
            worker_count=1,
        )
        match = CandidateAction(**{**match.__dict__, "topology_aware": True, "topology_preference": "C"})
        mismatch = CandidateAction(**{**mismatch.__dict__, "topology_aware": True, "topology_preference": "C"})
        nodes = [
            NodeState(name="node-a", domain="A", cpu_total=32000, mem_total=128 * 1024**3, gpu_total=4),
            NodeState(name="node-c", domain="C", cpu_total=32000, mem_total=128 * 1024**3, gpu_total=4),
        ]
        state = kueue_state_vector(
            candidates=[match, mismatch],
            waiting=waiting,
            running=[],
            future=[],
            nodes=nodes,
            time_now=0.0,
            blocked_seconds=0.0,
            idle_quota_while_blocked=0.0,
            fair_share_violations=0,
        )
        match_slice = state[:22]
        mismatch_slice = state[22:44]
        self.assertNotEqual(float(match_slice[19]), float(mismatch_slice[19]))

    def test_topology_aware_pods_export_scheduler_visible_hint_labels(self):
        workload = next(
            item
            for item in build_kueue_workloads(
                seed=7,
                num_jobs=6,
                arrival_span=120.0,
                workload_preset="kueue-lingjun-gang-topology-provisioning",
            )
            if item.workload_id == "0-etp-critical-c-head"
        )

        docs = pod_group_docs(workload, namespace="default")
        self.assertTrue(docs)
        labels = docs[0]["metadata"]["labels"]
        self.assertEqual(labels["admirl.ai/topology-preference"], "C")
        self.assertEqual(labels["gpu-topology-hint"], "C")

    def test_blocked_bypass_risk_only_triggers_for_overlapping_smalls(self):
        blocked = [
            _candidate(
                action_id="head@rf-4gpu-b",
                workload_id="head",
                flavor_name="rf-4gpu-b",
                queue_class="gang",
                immediate_fit=False,
                total_gpu=8,
                worker_count=2,
            ),
            _candidate(
                action_id="small@rf-4gpu-b",
                workload_id="small",
                flavor_name="rf-4gpu-b",
                queue_class="small",
                immediate_fit=True,
                total_gpu=4,
                worker_count=1,
            ),
        ]
        self.assertTrue(_has_blocked_bypass_risk(blocked, blocked_head_context(blocked)))

        disjoint = [
            _candidate(
                action_id="head@rf-4gpu-b",
                workload_id="head",
                flavor_name="rf-4gpu-b",
                queue_class="gang",
                immediate_fit=False,
                total_gpu=8,
                worker_count=2,
            ),
            _candidate(
                action_id="small@rf-2gpu-a",
                workload_id="small",
                flavor_name="rf-2gpu-a",
                queue_class="small",
                immediate_fit=True,
                total_gpu=2,
                worker_count=1,
            ),
        ]
        self.assertFalse(_has_blocked_bypass_risk(disjoint, blocked_head_context(disjoint)))

    def test_matching_topology_flavor_outranks_mismatched_exact_fit(self):
        match = CandidateAction(
            **{
                **_candidate(
                    action_id="topo@rf-4gpu-c",
                    workload_id="topo",
                    flavor_name="rf-4gpu-c",
                    queue_class="gang",
                    immediate_fit=True,
                    total_gpu=8,
                    worker_count=2,
                ).__dict__,
                "topology_aware": True,
                "topology_preference": "C",
                "flavor_gpu_size": 4,
            }
        )
        mismatch = CandidateAction(
            **{
                **_candidate(
                    action_id="topo@rf-4gpu-a",
                    workload_id="topo",
                    flavor_name="rf-4gpu-a",
                    queue_class="gang",
                    immediate_fit=True,
                    total_gpu=8,
                    worker_count=2,
                ).__dict__,
                "topology_aware": True,
                "topology_preference": "C",
                "flavor_gpu_size": 4,
            }
        )

        self.assertGreater(heuristic_candidate_priority(match), heuristic_candidate_priority(mismatch))

    def test_sorted_runtime_candidates_restores_training_order(self):
        small = _candidate(
            action_id="small@rf-2gpu-a",
            workload_id="small",
            flavor_name="rf-2gpu-a",
            queue_class="small",
            immediate_fit=True,
            total_gpu=2,
            worker_count=1,
        )
        gang = _candidate(
            action_id="gang@rf-4gpu-b",
            workload_id="gang",
            flavor_name="rf-4gpu-b",
            queue_class="gang",
            immediate_fit=False,
            total_gpu=8,
            worker_count=2,
        )

        ordered = _sorted_runtime_candidates([small, gang])
        self.assertEqual([item.action_id for item in ordered], ["gang@rf-4gpu-b", "small@rf-2gpu-a"])

    def test_safe_immediate_override_promotes_disjoint_immediate_fit(self):
        provisionable_same_pool = _candidate(
            action_id="bypass@rf-4gpu-a",
            workload_id="bypass",
            flavor_name="rf-4gpu-a",
            queue_class="small",
            immediate_fit=False,
            total_gpu=4,
            worker_count=1,
        )
        immediate_disjoint = _candidate(
            action_id="noise@rf-2gpu-b",
            workload_id="noise",
            flavor_name="rf-2gpu-b",
            queue_class="small",
            immediate_fit=True,
            total_gpu=2,
            worker_count=1,
        )

        adjusted, override = _apply_safe_immediate_override(
            candidates=[provisionable_same_pool, immediate_disjoint],
            probs=[0.49, 0.51],
            pair_scores={
                provisionable_same_pool.action_id: heuristic_candidate_priority(provisionable_same_pool),
                immediate_disjoint.action_id: heuristic_candidate_priority(immediate_disjoint),
            },
        )

        self.assertIsNotNone(override)
        assert override is not None
        self.assertEqual(override["promote_action_id"], immediate_disjoint.action_id)
        self.assertGreater(adjusted[immediate_disjoint.action_id], adjusted[provisionable_same_pool.action_id])

    def test_safe_immediate_override_does_not_flip_immediate_gang_ranking(self):
        heuristic_gang = CandidateAction(
            **{
                **_candidate(
                    action_id="topo-d@rf-4gpu-c",
                    workload_id="topo-d",
                    flavor_name="rf-4gpu-c",
                    queue_class="gang",
                    immediate_fit=True,
                    total_gpu=8,
                    worker_count=2,
                ).__dict__,
                "topology_aware": True,
                "topology_preference": "C",
                "flavor_gpu_size": 4,
            }
        )
        learned_gang = CandidateAction(
            **{
                **_candidate(
                    action_id="topo-c@rf-4gpu-a",
                    workload_id="topo-c",
                    flavor_name="rf-4gpu-a",
                    queue_class="gang",
                    immediate_fit=True,
                    total_gpu=8,
                    worker_count=2,
                ).__dict__,
                "topology_aware": True,
                "topology_preference": "A",
                "flavor_gpu_size": 4,
            }
        )

        adjusted, override = _apply_safe_immediate_override(
            candidates=[heuristic_gang, learned_gang],
            probs=[0.40, 0.60],
            pair_scores={
                heuristic_gang.action_id: heuristic_candidate_priority(heuristic_gang),
                learned_gang.action_id: heuristic_candidate_priority(learned_gang),
            },
        )

        self.assertIsNone(override)
        self.assertEqual(adjusted[heuristic_gang.action_id], heuristic_candidate_priority(heuristic_gang))
        self.assertEqual(adjusted[learned_gang.action_id], heuristic_candidate_priority(learned_gang))

    def test_runtime_time_uses_candidate_wait_when_request_time_is_wall_clock(self):
        candidate = _candidate(
            action_id="small@rf-2gpu-a",
            workload_id="small",
            flavor_name="rf-2gpu-a",
            queue_class="small",
            immediate_fit=True,
            total_gpu=2,
            worker_count=1,
        )
        self.assertEqual(
            _runtime_time_from_request({"time": 9_999_999_999.0}, [candidate]),
            candidate.wait_seconds,
        )


if __name__ == "__main__":
    unittest.main()
