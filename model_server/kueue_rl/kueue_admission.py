from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Iterable

import numpy as np

from .config import (
    JOB_FEAT_DIM,
    MAX_CLUSTER_NODES,
    MAX_CPU_MILLI,
    MAX_JOB_DURATION,
    MAX_JOB_GPU,
    MAX_MEM_BYTES,
    MAX_NODE_GPU,
    MAX_QUEUE_JOBS,
    MAX_TIME_HORIZON,
    MAX_WAIT_TIME,
    NODE_FEAT_DIM,
    STATE_DIM,
)
from .cluster import NodeState, cluster_gpu_fragmentation, cluster_gpu_utilization, largest_free_gpu_block, percentile, topology_scalar
from .lingjun import LingjunWorkloadSpec, class_mix, load_lingjun_workloads


KUEUE_WORKLOAD_PRESET_ALIASES = {
    "kueue-lingjun-gang-starvation": "kueue-lingjun-gang-starvation",
    "kueue-lingjun-gang-starvation-cohort": "kueue-lingjun-gang-starvation-cohort",
    "kueue-lingjun-gang-topology-provisioning": "kueue-lingjun-gang-topology-provisioning",
    "kueue-lingjun-gang-elastic-topology": "kueue-lingjun-gang-elastic-topology",
    "kueue-lingjun-gang-elastic-profile-cohort": "kueue-lingjun-gang-elastic-profile-cohort",
}

KUEUE_CLUSTER_LAYOUTS = {
    "kueue-gang-starvation": [
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "A"},
        {"count": 2, "gpus": 2, "cpu_milli": 16000, "mem_bytes": 64 * (1024**3), "domain": "B"},
        {"count": 1, "gpus": 6, "cpu_milli": 32000, "mem_bytes": 128 * (1024**3), "domain": "C"},
    ],
    "kueue-gang-topology-provisioning": [
        # Keep C first so upstream default flavor ordering prefers the scarce
        # topology-aligned pool unless the advisor overrides it.
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "C"},
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "A"},
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "B"},
    ],
    "kueue-gang-elastic-topology": [
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "C"},
        {"count": 1, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "A"},
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "B"},
    ],
    "kueue-gang-elastic-profile-cohort": [
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "C"},
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "A"},
        {"count": 2, "gpus": 4, "cpu_milli": 24000, "mem_bytes": 96 * (1024**3), "domain": "B"},
    ],
}

KUEUE_PRESET_TO_LAYOUT = {
    "kueue-lingjun-gang-starvation": "kueue-gang-starvation",
    "kueue-lingjun-gang-starvation-cohort": "kueue-gang-starvation",
    "kueue-lingjun-gang-topology-provisioning": "kueue-gang-topology-provisioning",
    "kueue-lingjun-gang-elastic-topology": "kueue-gang-elastic-topology",
    "kueue-lingjun-gang-elastic-profile-cohort": "kueue-gang-elastic-profile-cohort",
}


@dataclass(frozen=True)
class KueueWorkload:
    workload_id: str
    queue_name: str
    cluster_queue: str
    fairshare_group: str
    queue_class: str
    priority: int
    worker_count: int
    per_worker_gpu: int
    per_worker_cpu_milli: int
    per_worker_mem_bytes: int
    total_gpu: int
    runtime_seconds: float
    arrival_time: float
    topology_aware: bool
    topology_preference: str
    restart_count: int
    candidate_flavors: tuple[str, ...]
    elastic_enabled: bool = False
    min_worker_count: int = 1
    preferred_worker_count: int = 1
    max_worker_count: int = 1
    profile_name: str = ""

    @property
    def is_gang(self) -> bool:
        return self.worker_count >= 2

    @property
    def is_elastic(self) -> bool:
        return self.elastic_enabled and self.max_worker_count > self.min_worker_count


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    workload_id: str
    flavor_name: str
    queue_name: str
    cluster_queue: str
    fairshare_group: str
    priority: int
    wait_seconds: float
    runtime_seconds: float
    worker_count: int
    total_gpu: int
    per_worker_gpu: int
    topology_aware: bool
    topology_preference: str
    flavor_domain: str
    immediate_fit: bool
    provisionable: bool
    available_gpu: int
    total_gpu_capacity: int
    fairshare_debt: float
    requeue_count: int
    queue_class: str
    flavor_gpu_size: int
    oversize_gpu: int
    competing_older_pressure: float
    elastic_enabled: bool
    min_worker_count: int
    preferred_worker_count: int
    max_worker_count: int
    scale_tag: str
    scale_fraction: float
    profile_name: str = ""


@dataclass
class RunningWorkload:
    workload: KueueWorkload
    flavor_name: str
    allocations: list[str]
    start_time: float
    end_time: float
    topology_hit: bool
    provisioning_delay: float
    admitted_worker_count: int
    remaining_work: float
    last_progress_at: float
    expanded: bool = False


@dataclass
class ProvisionEvent:
    complete_at: float
    flavor_name: str
    nodes_to_add: int


@dataclass
class KueueFairShareState:
    launched: dict[str, int] = field(default_factory=dict)
    completed: dict[str, int] = field(default_factory=dict)


def canonical_kueue_preset(preset: str) -> str:
    return KUEUE_WORKLOAD_PRESET_ALIASES.get(preset, preset)


def is_kueue_preset(preset: str) -> bool:
    return canonical_kueue_preset(preset) in KUEUE_PRESET_TO_LAYOUT


def default_cluster_layout_for_kueue_preset(preset: str) -> str:
    canonical = canonical_kueue_preset(preset)
    if canonical not in KUEUE_PRESET_TO_LAYOUT:
        raise ValueError(f"unsupported Kueue preset: {preset}")
    return KUEUE_PRESET_TO_LAYOUT[canonical]


def _flavor_name(group: dict) -> str:
    return f"rf-{group['gpus']}gpu-{group['domain'].lower()}"


def kueue_nodes_for_layout(layout: str) -> list[NodeState]:
    groups = KUEUE_CLUSTER_LAYOUTS.get(layout)
    if groups is None:
        raise ValueError(f"unknown Kueue cluster layout: {layout}")
    nodes: list[NodeState] = []
    for group in groups:
        flavor = _flavor_name(group)
        for index in range(group["count"]):
            node = NodeState(
                name=f"{flavor}-{index:02d}",
                domain=group["domain"],
                cpu_total=group["cpu_milli"],
                mem_total=group["mem_bytes"],
                gpu_total=group["gpus"],
            )
            nodes.append(node)
    return nodes


def _rescale_arrivals(workloads: list[KueueWorkload], arrival_span: float) -> list[KueueWorkload]:
    if not workloads:
        return []
    if arrival_span <= 0:
        return list(workloads)
    max_arrival = max(item.arrival_time for item in workloads)
    if max_arrival <= 0:
        return list(workloads)
    scaled: list[KueueWorkload] = []
    for item in workloads:
        scaled.append(
            KueueWorkload(
                **{
                    **item.__dict__,
                    "arrival_time": (item.arrival_time / max_arrival) * arrival_span,
                }
            )
        )
    return scaled


def _queue_name_for_group(group: str) -> str:
    token = "".join(ch for ch in (group or "default").lower() if ch.isalnum())
    token = token[:8] or "default"
    return f"lq-{token}"


def _pick_template(
    pool: list[LingjunWorkloadSpec],
    rng: random.Random,
    predicate,
    fallback: list[LingjunWorkloadSpec],
) -> LingjunWorkloadSpec:
    filtered = [item for item in pool if predicate(item)]
    source = filtered or fallback or pool
    if not source:
        raise ValueError("no Lingjun workloads available for Kueue suite construction")
    return rng.choice(source)


def _workload_from_template(
    template: LingjunWorkloadSpec,
    *,
    workload_id: str,
    arrival_time: float,
    candidate_flavors: tuple[str, ...],
    fairshare_group: str,
    runtime_seconds: float | None = None,
    worker_count: int | None = None,
    per_worker_gpu: int | None = None,
    topology_aware: bool | None = None,
    topology_preference: str | None = None,
    priority: int | None = None,
    elastic_enabled: bool = False,
    min_worker_count: int | None = None,
    preferred_worker_count: int | None = None,
    max_worker_count: int | None = None,
) -> KueueWorkload:
    actual_worker_count = max(1, int(worker_count if worker_count is not None else template.worker_count))
    actual_per_worker_gpu = max(1, int(per_worker_gpu if per_worker_gpu is not None else template.per_worker_gpu))
    actual_runtime = max(60.0, float(runtime_seconds if runtime_seconds is not None else template.runtime_seconds))
    actual_topology_aware = template.topology_aware if topology_aware is None else bool(topology_aware)
    actual_topology_preference = (
        (template.topology_preference or "")
        if topology_preference is None
        else str(topology_preference)
    )
    actual_priority = int(priority if priority is not None else template.priority)
    actual_group = fairshare_group or template.fairshare_group or "default"
    actual_min_workers = max(1, int(min_worker_count if min_worker_count is not None else actual_worker_count))
    actual_preferred_workers = max(actual_min_workers, int(preferred_worker_count if preferred_worker_count is not None else actual_worker_count))
    actual_max_workers = max(actual_preferred_workers, int(max_worker_count if max_worker_count is not None else actual_worker_count))
    return KueueWorkload(
        workload_id=workload_id,
        queue_name=_queue_name_for_group(actual_group),
        cluster_queue="training-cluster-queue",
        fairshare_group=actual_group,
        queue_class="gang" if actual_worker_count >= 2 else "small",
        priority=actual_priority,
        worker_count=actual_worker_count,
        per_worker_gpu=actual_per_worker_gpu,
        per_worker_cpu_milli=max(1000, template.per_worker_cpu_milli),
        per_worker_mem_bytes=max(2 * (1024**3), template.per_worker_mem_bytes),
        total_gpu=actual_worker_count * actual_per_worker_gpu,
        runtime_seconds=actual_runtime,
        arrival_time=max(0.0, float(arrival_time)),
        topology_aware=actual_topology_aware,
        topology_preference=actual_topology_preference,
        restart_count=max(0, int(template.restart_count)),
        candidate_flavors=tuple(dict.fromkeys(candidate_flavors)),
        elastic_enabled=bool(elastic_enabled),
        min_worker_count=actual_min_workers,
        preferred_worker_count=actual_preferred_workers,
        max_worker_count=actual_max_workers,
        profile_name=workload_id,
    )


def _build_gang_starvation_suite(
    *,
    seed: int,
    num_jobs: int,
    trace_split: str,
    train_fraction: float,
) -> list[KueueWorkload]:
    rng = random.Random(seed)
    pool = list(load_lingjun_workloads(trace_split=trace_split, train_fraction=train_fraction))
    if not pool:
        return []
    gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4]
    single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4]
    noise_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu <= 2]

    workloads = [
        _workload_from_template(
            _pick_template(single_pool, rng, lambda item: item.per_worker_gpu >= 4, pool),
            workload_id="starve-blocker-0",
            arrival_time=0.0,
            candidate_flavors=("rf-4gpu-a",),
            fairshare_group="starve-a",
            runtime_seconds=900.0,
            per_worker_gpu=4,
            topology_aware=False,
            topology_preference="",
            priority=5,
        ),
        _workload_from_template(
            _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
            workload_id="starve-head-gang",
            arrival_time=0.08,
            candidate_flavors=("rf-4gpu-a",),
            fairshare_group="starve-gang",
            runtime_seconds=960.0,
            worker_count=2,
            per_worker_gpu=4,
            topology_aware=False,
            topology_preference="",
            priority=8,
        ),
    ]

    extra_jobs = max(0, num_jobs - len(workloads))
    for index in range(extra_jobs):
        # Create a one-block-short bypass window inside a single ClusterQueue:
        # while the older gang waits for the second 4-GPU block, younger same-flavor
        # small jobs can keep serially consuming the only free block.
        if index == extra_jobs - 1:
            template = _pick_template(noise_pool, rng, lambda item: item.per_worker_gpu <= 2, pool)
            workloads.append(
                _workload_from_template(
                    template,
                    workload_id=f"starve-noise-{index}",
                    arrival_time=24.0,
                    candidate_flavors=("rf-2gpu-b",),
                    fairshare_group="starve-b",
                    runtime_seconds=max(240.0, min(480.0, template.runtime_seconds)),
                    per_worker_gpu=2,
                    topology_aware=False,
                    topology_preference="",
                    priority=2,
                )
            )
            continue
        template = _pick_template(single_pool, rng, lambda item: item.per_worker_gpu >= 4, pool)
        workloads.append(
            _workload_from_template(
                template,
                workload_id=f"starve-bypass-a-{index}",
                arrival_time=0.12 + (index * 0.02),
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="starve-a",
                runtime_seconds=max(540.0, min(720.0, template.runtime_seconds)),
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            )
        )
    return workloads[:num_jobs]


def _with_queue_binding(
    workload: KueueWorkload,
    *,
    queue_name: str,
    cluster_queue: str,
    fairshare_group: str,
) -> KueueWorkload:
    return replace(
        workload,
        queue_name=queue_name,
        cluster_queue=cluster_queue,
        fairshare_group=fairshare_group,
    )


def _build_gang_starvation_cohort_suite(
    *,
    seed: int,
    num_jobs: int,
    trace_split: str,
    train_fraction: float,
) -> list[KueueWorkload]:
    rng = random.Random(seed)
    pool = list(load_lingjun_workloads(trace_split=trace_split, train_fraction=train_fraction))
    if not pool:
        return []
    gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4]
    single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4]

    gang_queue = "lq-starvega"
    gang_cluster_queue = "training-cluster-queue-gang"
    small_queue = "lq-starvea"
    small_cluster_queue = "training-cluster-queue-small"

    workloads = [
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.per_worker_gpu >= 4, pool),
                workload_id="cohort-blocker-gang",
                arrival_time=0.0,
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="starve-gang",
                runtime_seconds=720.0,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="starve-gang",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="cohort-head-gang",
                arrival_time=1.0,
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="starve-gang",
                runtime_seconds=960.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=8,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="starve-gang",
        ),
    ]

    extra_jobs = max(0, num_jobs - len(workloads))
    for index in range(extra_jobs):
        template = _pick_template(single_pool, rng, lambda item: item.per_worker_gpu >= 4, pool)
        workloads.append(
            _with_queue_binding(
                _workload_from_template(
                    template,
                    workload_id=f"cohort-small-a-{index}",
                    arrival_time=5.0 + float(index),
                    candidate_flavors=("rf-4gpu-a",),
                    fairshare_group="starve-small",
                    runtime_seconds=1200.0,
                    per_worker_gpu=4,
                    topology_aware=False,
                    topology_preference="",
                    priority=4,
                ),
                queue_name=small_queue,
                cluster_queue=small_cluster_queue,
                fairshare_group="starve-small",
            )
        )
    return workloads[:num_jobs]


def _build_gang_topology_provisioning_suite(
    *,
    seed: int,
    num_jobs: int,
    trace_split: str,
    train_fraction: float,
) -> list[KueueWorkload]:
    rng = random.Random(seed)
    pool = list(load_lingjun_workloads(trace_split=trace_split, train_fraction=train_fraction))
    if not pool:
        return []
    gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4 and _live_feasible_trace_item(item)]
    single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4]
    if not gang_pool:
        gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4] or pool
    if not single_pool:
        single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4] or pool

    gang_queue = "lq-etp-gang"
    gang_cluster_queue = "training-cluster-queue-gang"
    small_queue = "lq-etp-small"
    small_cluster_queue = "training-cluster-queue-small"

    workloads = [
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="etp-blocker-c",
                arrival_time=0.0,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="etp-gang-critical",
                runtime_seconds=720.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="etp-gang-critical",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="0-etp-critical-c-head",
                arrival_time=1.0,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="etp-gang-critical",
                runtime_seconds=720.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=9,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="etp-gang-critical",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="1-etp-flex-c-0",
                arrival_time=5.0,
                candidate_flavors=("rf-4gpu-c", "rf-4gpu-a"),
                fairshare_group="etp-small-flex",
                runtime_seconds=1680.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=5,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="etp-small-flex",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="2-etp-b-gang",
                arrival_time=0.5,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="etp-gang-b",
                runtime_seconds=960.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="etp-gang-b",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="3-etp-noise-b-0",
                arrival_time=5.2,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="etp-small-b0",
                runtime_seconds=540.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="etp-small-a0",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="4-etp-noise-b-1",
                arrival_time=5.4,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="etp-small-b1",
                runtime_seconds=600.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="etp-small-a1",
        ),
    ]

    extra_jobs = max(0, num_jobs - len(workloads))
    for index in range(extra_jobs):
        template = _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool)
        workloads.append(
            _with_queue_binding(
                _workload_from_template(
                    template,
                    workload_id=f"5-etp-noise-b-{index}",
                    arrival_time=6.0 + (index * 0.35),
                    candidate_flavors=("rf-4gpu-b",),
                    fairshare_group="etp-small-b",
                    runtime_seconds=max(360.0, min(660.0, template.runtime_seconds)),
                    per_worker_gpu=4,
                    topology_aware=False,
                    topology_preference="",
                    priority=4,
                ),
                queue_name=small_queue,
                cluster_queue=small_cluster_queue,
                fairshare_group="etp-small-b",
            )
        )
    return workloads[:num_jobs]


def _build_gang_elastic_topology_suite(
    *,
    seed: int,
    num_jobs: int,
    trace_split: str,
    train_fraction: float,
) -> list[KueueWorkload]:
    rng = random.Random(seed)
    pool = list(load_lingjun_workloads(trace_split=trace_split, train_fraction=train_fraction))
    if not pool:
        return []
    gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4 and _live_feasible_trace_item(item)]
    single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4]
    if not gang_pool:
        gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4] or pool
    if not single_pool:
        single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4] or pool

    gang_queue = "lq-eet-gang"
    gang_cluster_queue = "training-cluster-queue-gang"
    small_queue = "lq-eet-small"
    small_cluster_queue = "training-cluster-queue-small"

    workloads = [
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="eet-blocker-c",
                arrival_time=0.0,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="eet-gang-critical",
                runtime_seconds=900.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eet-gang-critical",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="0-eet-critical-c-head",
                arrival_time=0.05,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="eet-gang-critical",
                runtime_seconds=900.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=9,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eet-gang-critical",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="1-eet-flex-elastic",
                arrival_time=0.06,
                candidate_flavors=("rf-4gpu-c", "rf-4gpu-a"),
                fairshare_group="eet-elastic-flex",
                runtime_seconds=1080.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=6,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eet-elastic-flex",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="2-eet-b-gang",
                arrival_time=0.03,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="eet-gang-b",
                runtime_seconds=960.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eet-gang-b",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="3-eet-noise-b-0",
                arrival_time=1.0,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="eet-small-b0",
                runtime_seconds=540.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eet-small-b0",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="4-eet-noise-a-0",
                arrival_time=1.1,
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="eet-small-a0",
                runtime_seconds=600.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eet-small-a0",
        ),
    ]

    extension_jobs = [
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="5-eet-blocker-c-1",
                arrival_time=1.45,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="eet-gang-critical-1",
                runtime_seconds=900.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eet-gang-critical-1",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="6-eet-b-gang-1",
                arrival_time=1.48,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="eet-gang-b1",
                runtime_seconds=900.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eet-gang-b1",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="7-eet-critical-c-head-1",
                arrival_time=1.52,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="eet-gang-critical-1",
                runtime_seconds=900.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=9,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eet-gang-critical-1",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="8-eet-flex-elastic-1",
                arrival_time=1.53,
                candidate_flavors=("rf-4gpu-c", "rf-4gpu-a"),
                fairshare_group="eet-elastic-flex-1",
                runtime_seconds=1080.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=6,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eet-elastic-flex-1",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="9-eet-noise-a-1",
                arrival_time=2.2,
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="eet-small-a1",
                runtime_seconds=540.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eet-small-a1",
        ),
    ]
    for workload in extension_jobs:
        if len(workloads) >= num_jobs:
            break
        workloads.append(workload)

    extra_jobs = max(0, num_jobs - len(workloads))
    for index in range(extra_jobs):
        template = _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool)
        workloads.append(
            _with_queue_binding(
                _workload_from_template(
                    template,
                    workload_id=f"10-eet-noise-b-{index}",
                    arrival_time=2.3 + (index * 0.1),
                    candidate_flavors=("rf-4gpu-b",),
                    fairshare_group="eet-small-b",
                    runtime_seconds=max(360.0, min(660.0, template.runtime_seconds)),
                    per_worker_gpu=4,
                    topology_aware=False,
                    topology_preference="",
                    priority=4,
                ),
                queue_name=small_queue,
                cluster_queue=small_cluster_queue,
                fairshare_group="eet-small-b",
            )
        )
    return workloads[:num_jobs]


def _build_gang_elastic_profile_cohort_suite(
    *,
    seed: int,
    num_jobs: int,
    trace_split: str,
    train_fraction: float,
) -> list[KueueWorkload]:
    rng = random.Random(seed)
    pool = list(load_lingjun_workloads(trace_split=trace_split, train_fraction=train_fraction))
    if not pool:
        return []
    gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4 and _live_feasible_trace_item(item)]
    single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4]
    if not gang_pool:
        gang_pool = [item for item in pool if item.worker_count >= 2 and item.per_worker_gpu >= 4] or pool
    if not single_pool:
        single_pool = [item for item in pool if item.worker_count == 1 and item.per_worker_gpu >= 4] or pool

    gang_queue = "lq-eep-gang"
    gang_cluster_queue = "training-cluster-queue-gang"
    elastic_queue = "lq-eep-flex"
    elastic_cluster_queue = "training-cluster-queue-flex"
    small_queue = "lq-eep-small"
    small_cluster_queue = "training-cluster-queue-small"

    workloads = [
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="eep-blocker-c",
                arrival_time=0.0,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="eep-gang-critical",
                runtime_seconds=900.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eep-gang-critical",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="0-eep-critical-c-head",
                arrival_time=0.05,
                candidate_flavors=("rf-4gpu-c",),
                fairshare_group="eep-gang-critical",
                runtime_seconds=900.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=9,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eep-gang-critical",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="1-eep-flex-elastic-c-0",
                arrival_time=0.06,
                candidate_flavors=("rf-4gpu-c", "rf-4gpu-a"),
                fairshare_group="eep-elastic-c",
                runtime_seconds=1140.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=6,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=elastic_queue,
            cluster_queue=elastic_cluster_queue,
            fairshare_group="eep-elastic-c",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="2-eep-flex-elastic-c-aux",
                arrival_time=0.40,
                candidate_flavors=("rf-4gpu-c", "rf-4gpu-a"),
                fairshare_group="eep-elastic-caux",
                runtime_seconds=1020.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=5,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=elastic_queue,
            cluster_queue=elastic_cluster_queue,
            fairshare_group="eep-elastic-caux",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="3-eep-b-gang",
                arrival_time=0.04,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="eep-gang-b",
                runtime_seconds=900.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=7,
            ),
            queue_name=gang_queue,
            cluster_queue=gang_cluster_queue,
            fairshare_group="eep-gang-b",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="4-eep-noise-a-0",
                arrival_time=3.20,
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="eep-small-a0",
                runtime_seconds=540.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eep-small-a0",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="5-eep-noise-a-1",
                arrival_time=3.40,
                candidate_flavors=("rf-4gpu-a",),
                fairshare_group="eep-small-a1",
                runtime_seconds=600.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eep-small-a1",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool),
                workload_id="6-eep-noise-b-0",
                arrival_time=0.95,
                candidate_flavors=("rf-4gpu-b",),
                fairshare_group="eep-small-b0",
                runtime_seconds=540.0,
                worker_count=1,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
            ),
            queue_name=small_queue,
            cluster_queue=small_cluster_queue,
            fairshare_group="eep-small-b0",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="7-eep-flex-elastic-c-1",
                arrival_time=1.20,
                candidate_flavors=("rf-4gpu-c", "rf-4gpu-a"),
                fairshare_group="eep-elastic-c1",
                runtime_seconds=1200.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=True,
                topology_preference="C",
                priority=5,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=elastic_queue,
            cluster_queue=elastic_cluster_queue,
            fairshare_group="eep-elastic-c1",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="8-goodput-c-elastic-0",
                arrival_time=2.30,
                candidate_flavors=("rf-4gpu-a", "rf-4gpu-c"),
                fairshare_group="eep-goodput-c0",
                runtime_seconds=900.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=elastic_queue,
            cluster_queue=elastic_cluster_queue,
            fairshare_group="eep-goodput-c0",
        ),
        _with_queue_binding(
            _workload_from_template(
                _pick_template(gang_pool, rng, lambda item: item.worker_count >= 2 and item.per_worker_gpu >= 4, pool),
                workload_id="9-goodput-a-elastic-0",
                arrival_time=2.90,
                candidate_flavors=("rf-4gpu-a", "rf-4gpu-c"),
                fairshare_group="eep-goodput-a0",
                runtime_seconds=720.0,
                worker_count=2,
                per_worker_gpu=4,
                topology_aware=False,
                topology_preference="",
                priority=4,
                elastic_enabled=True,
                min_worker_count=1,
                preferred_worker_count=2,
                max_worker_count=2,
            ),
            queue_name=elastic_queue,
            cluster_queue=elastic_cluster_queue,
            fairshare_group="eep-goodput-a0",
        ),
    ]

    extra_jobs = max(0, num_jobs - len(workloads))
    for index in range(extra_jobs):
        template = _pick_template(single_pool, rng, lambda item: item.worker_count == 1 and item.per_worker_gpu >= 4, pool)
        workloads.append(
            _with_queue_binding(
                _workload_from_template(
                    template,
                    workload_id=f"10-eep-noise-b-{index}",
                    arrival_time=1.4 + (index * 0.25),
                    candidate_flavors=("rf-4gpu-b",),
                    fairshare_group="eep-small-b",
                    runtime_seconds=max(360.0, min(660.0, template.runtime_seconds)),
                    per_worker_gpu=4,
                    topology_aware=False,
                    topology_preference="",
                    priority=4,
                ),
                queue_name=small_queue,
                cluster_queue=small_cluster_queue,
                fairshare_group="eep-small-b",
            )
        )
    return workloads[:num_jobs]


def build_kueue_workloads(
    *,
    seed: int,
    num_jobs: int,
    arrival_span: float,
    workload_preset: str,
    trace_split: str = "all",
    trace_train_fraction: float = 0.75,
) -> list[KueueWorkload]:
    canonical = canonical_kueue_preset(workload_preset)
    if canonical == "kueue-lingjun-gang-starvation":
        workloads = _build_gang_starvation_suite(
            seed=seed,
            num_jobs=num_jobs,
            trace_split=trace_split,
            train_fraction=trace_train_fraction,
        )
        workloads.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
        return _rescale_arrivals(workloads[:num_jobs], arrival_span)
    if canonical == "kueue-lingjun-gang-starvation-cohort":
        workloads = _build_gang_starvation_cohort_suite(
            seed=seed,
            num_jobs=num_jobs,
            trace_split=trace_split,
            train_fraction=trace_train_fraction,
        )
        workloads.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
        return _rescale_arrivals(workloads[:num_jobs], arrival_span)
    if canonical == "kueue-lingjun-gang-topology-provisioning":
        workloads = _build_gang_topology_provisioning_suite(
            seed=seed,
            num_jobs=num_jobs,
            trace_split=trace_split,
            train_fraction=trace_train_fraction,
        )
        workloads.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
        return _rescale_arrivals(workloads[:num_jobs], arrival_span)
    if canonical == "kueue-lingjun-gang-elastic-topology":
        workloads = _build_gang_elastic_topology_suite(
            seed=seed,
            num_jobs=num_jobs,
            trace_split=trace_split,
            train_fraction=trace_train_fraction,
        )
        workloads.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
        return _rescale_arrivals(workloads[:num_jobs], arrival_span)
    if canonical == "kueue-lingjun-gang-elastic-profile-cohort":
        workloads = _build_gang_elastic_profile_cohort_suite(
            seed=seed,
            num_jobs=num_jobs,
            trace_split=trace_split,
            train_fraction=trace_train_fraction,
        )
        workloads.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
        return _rescale_arrivals(workloads[:num_jobs], arrival_span)
    raise ValueError(f"unsupported Kueue workload preset: {workload_preset}")


def _node_flavor(node: NodeState) -> str:
    return f"rf-{node.gpu_total}gpu-{node.domain.lower()}"


def _nodes_for_flavor(nodes: Iterable[NodeState], flavor_name: str) -> list[NodeState]:
    return [node for node in nodes if _node_flavor(node) == flavor_name]


def _flavor_gpu_size(flavor_name: str) -> int:
    token = flavor_name.split("-", 2)[1] if flavor_name.startswith("rf-") and "-" in flavor_name else flavor_name
    token = token.removesuffix("gpu")
    try:
        return max(0, int(token))
    except ValueError:
        return 0


def _live_feasible_trace_item(spec: LingjunWorkloadSpec) -> bool:
    return not (spec.worker_count >= 2 and spec.per_worker_gpu >= 8)


def elastic_scale_options(workload: KueueWorkload) -> list[tuple[str, int]]:
    if not workload.is_elastic:
        return [("fixed", workload.worker_count)]
    options = [
        ("min", workload.min_worker_count),
        ("preferred", workload.preferred_worker_count),
        ("max", workload.max_worker_count),
    ]
    deduped: list[tuple[str, int]] = []
    seen: set[int] = set()
    for scale_tag, worker_count in options:
        count = max(1, int(worker_count))
        if count in seen:
            continue
        seen.add(count)
        deduped.append((scale_tag, count))
    return deduped


def workload_for_scale(workload: KueueWorkload, selected_worker_count: int) -> KueueWorkload:
    count = max(1, int(selected_worker_count))
    return replace(
        workload,
        worker_count=count,
        total_gpu=count * workload.per_worker_gpu,
        queue_class="gang" if count >= 2 else "small",
    )


def elastic_speedup(worker_count: int, max_worker_count: int) -> float:
    workers = max(1, int(worker_count))
    max_workers = max(1, int(max_worker_count))
    if max_workers <= 1:
        return 1.0
    additional = max(0, workers - 1)
    return 1.0 + (0.80 * additional)


def elastic_total_work_units(workload: KueueWorkload, flavor_name: str) -> float:
    base_runtime = float(workload.runtime_seconds) * workload_runtime_multiplier(
        workload.workload_id,
        flavor_name,
        workload.profile_name,
    )
    return base_runtime * elastic_speedup(workload.max_worker_count, workload.max_worker_count)


def elastic_expected_runtime(workload: KueueWorkload, flavor_name: str, selected_worker_count: int) -> float:
    work_units = elastic_total_work_units(workload, flavor_name)
    speed = elastic_speedup(selected_worker_count, workload.max_worker_count)
    return max(60.0, work_units / max(speed, 1e-6))


def _take_diverse(
    *,
    pool: list[LingjunWorkloadSpec],
    target: int,
    rng: random.Random,
    used_ids: set[str],
    preferred_groups: set[str] | None = None,
) -> list[LingjunWorkloadSpec]:
    if target <= 0:
        return []
    remaining = [item for item in pool if item.workload_id not in used_ids]
    remaining.sort(key=lambda item: (item.fairshare_group, item.arrival_time, -item.total_gpu, item.workload_id))
    grouped: dict[str, list[LingjunWorkloadSpec]] = {}
    for item in remaining:
        grouped.setdefault(item.fairshare_group, []).append(item)

    group_order = list(grouped)
    if preferred_groups:
        preferred = [group for group in group_order if group in preferred_groups]
        other = [group for group in group_order if group not in preferred_groups]
        rng.shuffle(preferred)
        rng.shuffle(other)
        group_order = preferred + other
    else:
        rng.shuffle(group_order)

    chosen: list[LingjunWorkloadSpec] = []
    while group_order and len(chosen) < target:
        next_round: list[str] = []
        for group in group_order:
            items = grouped[group]
            if not items:
                continue
            item = items.pop(0)
            chosen.append(item)
            used_ids.add(item.workload_id)
            if len(chosen) >= target:
                break
            if items:
                next_round.append(group)
        group_order = next_round
    return chosen


def _protective_pressure(workload: KueueWorkload, flavor_name: str, waiting: list[KueueWorkload]) -> float:
    pressure = 0.0
    flavor_gpu = _flavor_gpu_size(flavor_name)
    flavor_domain = flavor_name.rsplit("-", 1)[-1].upper()
    for other in waiting:
        if other.workload_id == workload.workload_id:
            continue
        if other.arrival_time > workload.arrival_time:
            same_burst_topology_gang = (
                other.topology_aware
                and other.topology_preference == flavor_domain
                and (other.arrival_time - workload.arrival_time) <= 1.0
            )
            if not same_burst_topology_gang:
                continue
        if flavor_name not in other.candidate_flavors or not other.is_gang:
            continue
        if other.per_worker_gpu >= flavor_gpu or other.total_gpu > workload.total_gpu or other.worker_count > workload.worker_count:
            pressure += 1.0 + min(float(other.worker_count) / 4.0, 2.0)
    return pressure


def _feasible_allocations(workload: KueueWorkload, flavor_name: str, nodes: list[NodeState]) -> list[str] | None:
    candidates = [node for node in _nodes_for_flavor(nodes, flavor_name) if node.free_gpu >= workload.per_worker_gpu]
    candidates.sort(key=lambda node: (node.free_gpu, node.name))
    allocations: list[str] = []
    remaining = workload.worker_count
    temp_free = {node.name: node.free_gpu for node in candidates}
    while remaining > 0:
        placed = False
        for node in candidates:
            if temp_free[node.name] >= workload.per_worker_gpu:
                temp_free[node.name] -= workload.per_worker_gpu
                allocations.append(node.name)
                remaining -= 1
                placed = True
                break
        if not placed:
            return None
    return allocations


def _score_node_tight_fit(node: dict, pod_gpu_request: int, topology_hint: str) -> int:
    cpu_fit = float(node.get("cpu_fit_ratio", 0.0) or 0.0)
    mem_fit = float(node.get("mem_fit_ratio", 0.0) or 0.0)
    can_fit_gpu = bool(node.get("can_fit_gpu", True))
    if cpu_fit > 1.0 or mem_fit > 1.0 or (pod_gpu_request > 0 and not can_fit_gpu):
        return 0

    score = 20
    score += int(min(cpu_fit, 1.0) * 18)
    score += int(min(mem_fit, 1.0) * 18)

    gpu_total = int(node.get("gpu_total", 0) or 0)
    gpu_free = int(node.get("gpu_free", 0) or 0)
    gpu_free_ratio = float(node.get("gpu_free_ratio", 0.0) or 0.0)

    if pod_gpu_request > 0:
        residual_gpu = max(0, gpu_free - pod_gpu_request)
        score += 35 - min(residual_gpu * 8, 35)
        score += int((1.0 - gpu_free_ratio) * 10)
        if gpu_free < gpu_total:
            score += 6
        if pod_gpu_request > 1:
            topology_hint_match = bool(node.get("topology_hint_match", True))
            if topology_hint:
                score += 14 if topology_hint_match else -12
            elif int(node.get("nvlink_domain_id", -1) or -1) >= 0:
                score += 4
    else:
        score += 10 if gpu_total == 0 else -8

    return max(0, min(100, score))


def _score_node_heuristically(
    *,
    node: dict,
    pod_gpu_request: int,
    topology_hint: str,
    cluster_gpu_frag: float,
    cluster_gpu_free_ratio: float,
    smallest_feasible_gpu_total: int,
    matching_feasible_count: int,
    scarce_large_blocks: int,
    scarce_huge_blocks: int,
) -> int:
    gpu_free = int(node.get("gpu_free", 0) or 0)
    gpu_total = int(node.get("gpu_total", 0) or 0)
    can_fit_gpu = bool(node.get("can_fit_gpu", True))
    gpu_free_ratio = float(node.get("gpu_free_ratio", 0.0) or 0.0)
    topology_hint_match = bool(node.get("topology_hint_match", True))
    cpu_fit = float(node.get("cpu_fit_ratio", 0.0) or 0.0)
    mem_fit = float(node.get("mem_fit_ratio", 0.0) or 0.0)

    if cpu_fit > 1.0 or mem_fit > 1.0 or (pod_gpu_request > 0 and not can_fit_gpu):
        return 0

    score = 50

    if pod_gpu_request <= 0:
        return score + (8 if gpu_total == 0 else -5)

    residual_gpu = max(0, gpu_free - pod_gpu_request)
    total_surplus_gpu = max(0, gpu_total - pod_gpu_request)

    if pod_gpu_request >= 4:
        score = _score_node_tight_fit(node, pod_gpu_request, topology_hint)
        if smallest_feasible_gpu_total:
            if gpu_total == smallest_feasible_gpu_total:
                score += 10
            else:
                score -= min(12, max(0, gpu_total - smallest_feasible_gpu_total) * 4)
        if scarce_huge_blocks and pod_gpu_request < 8 and gpu_free >= 8:
            score -= 18
        if scarce_large_blocks and pod_gpu_request < 6 and gpu_free >= 6:
            score -= 10
        if residual_gpu == 0:
            score += 10
        elif residual_gpu == 1:
            score += 4
        return score

    score += 20
    score += max(0, 22 - (residual_gpu * 8))
    score += max(0, 18 - (total_surplus_gpu * 6))
    score += int(min(cpu_fit, 1.0) * 8)
    score += int(min(mem_fit, 1.0) * 8)

    if gpu_free == pod_gpu_request:
        score += 18
    elif residual_gpu == 1:
        score += 8

    if smallest_feasible_gpu_total:
        if gpu_total == smallest_feasible_gpu_total:
            score += 16
        else:
            score -= min(18, max(0, gpu_total - smallest_feasible_gpu_total) * 6)

    if pod_gpu_request <= 2:
        if gpu_total >= 6:
            score -= 28
        if gpu_free >= 6:
            score -= 18
    elif pod_gpu_request <= 4:
        if gpu_total >= 8:
            score -= 24
        if gpu_free >= 8:
            score -= 16

    if scarce_large_blocks and pod_gpu_request < 6 and gpu_free >= 6:
        score -= 18
    if scarce_huge_blocks and pod_gpu_request < 8 and gpu_free >= 8:
        score -= 26

    if pod_gpu_request >= 6:
        if gpu_total == pod_gpu_request:
            score += 20
        elif gpu_total > pod_gpu_request:
            score -= min(10, residual_gpu * 3)
        if gpu_free == pod_gpu_request:
            score += 14

    nvlink_domain_id = int(node.get("nvlink_domain_id", -1) or -1)
    if pod_gpu_request > 1 and nvlink_domain_id >= 0:
        if topology_hint:
            if topology_hint_match:
                score += 20 if pod_gpu_request <= 2 else 28
            else:
                score -= 6 if matching_feasible_count == 0 else (18 if pod_gpu_request <= 2 else 28)
        else:
            score += 4

    if pod_gpu_request > 0 and cluster_gpu_frag > 0.5:
        if residual_gpu <= 2:
            score += 8
        elif gpu_free >= 6 and pod_gpu_request < 6:
            score -= 8

    if cluster_gpu_free_ratio < 0.25 and residual_gpu == 0:
        score += 8
    elif cluster_gpu_free_ratio < 0.25 and gpu_free >= 6 and pod_gpu_request < 6:
        score -= 10

    return score


def _heuristic_candidate_node_features(
    node: NodeState,
    *,
    worker_gpu: int,
    worker_cpu_milli: int,
    worker_mem_bytes: int,
    topology_hint: str,
) -> dict[str, float | int | bool | str]:
    free_cpu = max(0, node.free_cpu)
    free_mem = max(0, node.free_mem)
    return {
        "name": node.name,
        "gpu_total": node.gpu_total,
        "gpu_free": node.free_gpu,
        "gpu_free_ratio": (float(node.free_gpu) / float(node.gpu_total)) if node.gpu_total > 0 else 0.0,
        "cpu_fit_ratio": _feature_ratio(worker_cpu_milli, max(1, free_cpu)) if worker_cpu_milli > 0 else 0.0,
        "mem_fit_ratio": _feature_ratio(worker_mem_bytes, max(1, free_mem)) if worker_mem_bytes > 0 else 0.0,
        "can_fit_gpu": node.free_gpu >= worker_gpu,
        "topology_hint_match": bool(topology_hint) and node.domain == topology_hint,
        "nvlink_domain_id": {"A": 0, "B": 1, "C": 2, "D": 3}.get(node.domain, -1),
    }


def _heuristic_worker_allocations(workload: KueueWorkload, flavor_name: str, nodes: list[NodeState]) -> list[str] | None:
    candidate_nodes = [node.snapshot() for node in _nodes_for_flavor(nodes, flavor_name)]
    allocations: list[str] = []
    topology_hint = workload.topology_preference if workload.topology_aware else ""

    for _ in range(workload.worker_count):
        feasible_nodes = [node for node in candidate_nodes if node.free_gpu >= workload.per_worker_gpu]
        if not feasible_nodes:
            return None

        cluster_gpu_frag = cluster_gpu_fragmentation(feasible_nodes)
        total_gpu = max(1, sum(max(0, node.gpu_total) for node in feasible_nodes))
        total_free_gpu = sum(max(0, node.free_gpu) for node in feasible_nodes)
        cluster_gpu_free_ratio = float(total_free_gpu) / float(total_gpu)
        smallest_feasible_gpu_total = min((node.gpu_total for node in feasible_nodes), default=0)

        feature_nodes = [
            _heuristic_candidate_node_features(
                node,
                worker_gpu=workload.per_worker_gpu,
                worker_cpu_milli=workload.per_worker_cpu_milli,
                worker_mem_bytes=workload.per_worker_mem_bytes,
                topology_hint=topology_hint,
            )
            for node in feasible_nodes
        ]
        matching_feasible_count = sum(1 for node in feature_nodes if bool(node.get("topology_hint_match", False)))
        scarce_large_blocks = 1 if sum(1 for node in feasible_nodes if node.free_gpu >= 6) <= 4 else 0
        scarce_huge_blocks = 1 if sum(1 for node in feasible_nodes if node.free_gpu >= 8) <= 2 else 0

        ranked = []
        for feature_node, backing_node in zip(feature_nodes, feasible_nodes):
            score = _score_node_heuristically(
                node=feature_node,
                pod_gpu_request=workload.per_worker_gpu,
                topology_hint=topology_hint,
                cluster_gpu_frag=cluster_gpu_frag,
                cluster_gpu_free_ratio=cluster_gpu_free_ratio,
                smallest_feasible_gpu_total=smallest_feasible_gpu_total,
                matching_feasible_count=matching_feasible_count,
                scarce_large_blocks=scarce_large_blocks,
                scarce_huge_blocks=scarce_huge_blocks,
            )
            ranked.append((score, backing_node.free_gpu, backing_node.name, backing_node))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen = ranked[0][3]
        allocations.append(chosen.name)
        chosen.allocate(
            type(
                "_Tmp",
                (),
                {
                    "cpu_milli": workload.per_worker_cpu_milli,
                    "mem_bytes": workload.per_worker_mem_bytes,
                    "gpu": workload.per_worker_gpu,
                },
            )()
        )

    return allocations


def _free_gpu_by_flavor(nodes: list[NodeState], flavor_name: str) -> int:
    return sum(max(0, node.free_gpu) for node in _nodes_for_flavor(nodes, flavor_name))


def _total_gpu_by_flavor(nodes: list[NodeState], flavor_name: str) -> int:
    return sum(node.gpu_total for node in _nodes_for_flavor(nodes, flavor_name))


def _fairshare_debt(waiting: list[KueueWorkload], fairshare: KueueFairShareState) -> dict[str, float]:
    groups = {workload.fairshare_group for workload in waiting}
    if not groups:
        return {}
    active_counts = {group: fairshare.launched.get(group, 0) - fairshare.completed.get(group, 0) for group in groups}
    mean_active = sum(active_counts.values()) / max(len(active_counts), 1)
    debt = {}
    for group, active in active_counts.items():
        debt[group] = max(-1.0, min(1.0, (active - mean_active) / max(mean_active + 1.0, 1.0)))
    return debt


def candidate_one_block_short(candidate: CandidateAction) -> bool:
    return (
        (not candidate.immediate_fit)
        and candidate.provisionable
        and candidate.available_gpu < candidate.total_gpu
        and (candidate.available_gpu + max(candidate.flavor_gpu_size, candidate.per_worker_gpu)) >= candidate.total_gpu
    )


def group_candidates_by_workload(candidates: list[CandidateAction]) -> dict[str, list[CandidateAction]]:
    grouped: dict[str, list[CandidateAction]] = {}
    for item in candidates:
        grouped.setdefault(item.workload_id, []).append(item)
    return grouped


def blocked_head_context(candidates: list[CandidateAction]) -> dict[str, object]:
    grouped = group_candidates_by_workload(candidates)
    blocked_heads = []
    blocked_provisionable = 0
    for workload_id, items in grouped.items():
        immediate_any = any(item.immediate_fit for item in items)
        provisionable_any = any(item.provisionable for item in items)
        if immediate_any:
            continue
        if provisionable_any:
            blocked_provisionable += 1
        items_sorted = sorted(
            items,
            key=lambda item: (
                -float(item.priority),
                -float(item.wait_seconds),
                -float(item.worker_count),
                -float(item.total_gpu),
                item.action_id,
            ),
        )
        blocked_heads.append((workload_id, items_sorted[0]))

    blocked_heads.sort(
        key=lambda item: (
            -float(item[1].priority),
            -float(item[1].wait_seconds),
            -float(item[1].worker_count),
            -float(item[1].total_gpu),
            item[0],
        )
    )
    blocked_workload_id = blocked_heads[0][0] if blocked_heads else ""
    blocked_flavors = {item.flavor_name for item in grouped.get(blocked_workload_id, [])}
    disjoint_immediate_fit = 0
    for workload_id, items in grouped.items():
        if workload_id == blocked_workload_id:
            continue
        flavor_names = {item.flavor_name for item in items}
        if blocked_flavors and flavor_names.isdisjoint(blocked_flavors) and any(item.immediate_fit for item in items):
            disjoint_immediate_fit += 1
    return {
        "blocked_workload_id": blocked_workload_id,
        "blocked_flavors": blocked_flavors,
        "blocked_provisionable": blocked_provisionable,
        "disjoint_immediate_fit": disjoint_immediate_fit,
        "workload_groups": grouped,
    }


def heuristic_candidate_priority(candidate: CandidateAction) -> float:
    priority = float(candidate.priority) * 20.0
    priority += min(candidate.wait_seconds, MAX_WAIT_TIME) / 15.0
    priority += float(candidate.total_gpu) * 1.5
    if candidate.elastic_enabled:
        if candidate.scale_tag == "max":
            priority += 6.0
        elif candidate.scale_tag == "preferred":
            priority += 3.0
        else:
            priority -= 2.0
        if candidate.competing_older_pressure > 0.0 and candidate.scale_tag == "min":
            priority += 8.0
        if candidate.competing_older_pressure > 0.0 and candidate.scale_tag == "max":
            priority -= 8.0
    if candidate.queue_class == "gang":
        priority += 18.0 + float(candidate.worker_count)
    topology_match = candidate.topology_aware and candidate.topology_preference and candidate.flavor_domain == candidate.topology_preference
    if candidate.topology_aware and candidate.topology_preference:
        if topology_match:
            priority += 18.0
            if candidate.queue_class == "gang":
                priority += 12.0 + (float(candidate.worker_count) * 1.5)
            if candidate.immediate_fit:
                priority += 6.0
        else:
            priority -= 18.0
            if candidate.queue_class == "gang":
                priority -= 10.0
    if candidate.immediate_fit:
        priority += 9.0
    elif candidate.provisionable:
        priority += 4.0
    if candidate.elastic_enabled and candidate.immediate_fit and candidate.scale_tag == "min":
        priority += 2.0
    if candidate.flavor_gpu_size == candidate.per_worker_gpu:
        priority += 5.0
    oversize_penalty = float(candidate.oversize_gpu) * 1.2
    if topology_match:
        oversize_penalty *= 0.35
    priority -= oversize_penalty
    if candidate.queue_class != "gang" and candidate.competing_older_pressure > 0:
        priority -= min(20.0, candidate.competing_older_pressure * 4.0)
        if candidate.oversize_gpu > 0:
            priority -= min(10.0, candidate.competing_older_pressure * 2.0)
    elif candidate.oversize_gpu > 0 and candidate.competing_older_pressure > 0 and not topology_match:
        priority -= min(18.0, candidate.competing_older_pressure * 4.0)
    priority -= max(0.0, candidate.fairshare_debt) * 12.0
    return priority


def build_candidate_actions(
    waiting: list[KueueWorkload],
    nodes: list[NodeState],
    time_now: float,
    fairshare: KueueFairShareState,
    limit: int = MAX_QUEUE_JOBS,
) -> list[CandidateAction]:
    debts = _fairshare_debt(waiting, fairshare)
    candidates: list[CandidateAction] = []
    for workload in waiting:
        wait_seconds = max(0.0, time_now - workload.arrival_time)
        for scale_tag, selected_workers in elastic_scale_options(workload):
            scaled_workload = workload_for_scale(workload, selected_workers)
            for flavor_name in workload.candidate_flavors:
                flavor_nodes = _nodes_for_flavor(nodes, flavor_name)
                if not flavor_nodes:
                    continue
                flavor_gpu_size = _flavor_gpu_size(flavor_name)
                immediate_fit = _feasible_allocations(scaled_workload, flavor_name, nodes) is not None
                total_gpu_capacity = _total_gpu_by_flavor(nodes, flavor_name)
                available_gpu = _free_gpu_by_flavor(nodes, flavor_name)
                provisionable = (not immediate_fit) and total_gpu_capacity >= scaled_workload.total_gpu
                if provisionable and workload_external_provision_only(workload.workload_id, workload.profile_name):
                    provisionable = False
                competing_older_pressure = _protective_pressure(scaled_workload, flavor_name, waiting)
                candidates.append(
                    CandidateAction(
                        action_id=f"{workload.workload_id}@{flavor_name}@w{selected_workers}",
                        workload_id=workload.workload_id,
                        profile_name=workload.profile_name or workload.workload_id,
                        flavor_name=flavor_name,
                        queue_name=workload.queue_name,
                        cluster_queue=workload.cluster_queue,
                        fairshare_group=workload.fairshare_group,
                        priority=workload.priority,
                        wait_seconds=wait_seconds,
                        runtime_seconds=elastic_expected_runtime(workload, flavor_name, selected_workers),
                        worker_count=scaled_workload.worker_count,
                        total_gpu=scaled_workload.total_gpu,
                        per_worker_gpu=workload.per_worker_gpu,
                        topology_aware=workload.topology_aware,
                        topology_preference=workload.topology_preference,
                        flavor_domain=flavor_name.rsplit("-", 1)[-1].upper(),
                        immediate_fit=immediate_fit,
                        provisionable=provisionable,
                        available_gpu=available_gpu,
                        total_gpu_capacity=total_gpu_capacity,
                        fairshare_debt=debts.get(workload.fairshare_group, 0.0),
                        requeue_count=workload.restart_count,
                        queue_class=scaled_workload.queue_class,
                        flavor_gpu_size=flavor_gpu_size,
                        oversize_gpu=max(0, flavor_gpu_size - workload.per_worker_gpu),
                        competing_older_pressure=competing_older_pressure,
                        elastic_enabled=workload.is_elastic,
                        min_worker_count=workload.min_worker_count,
                        preferred_worker_count=workload.preferred_worker_count,
                        max_worker_count=workload.max_worker_count,
                        scale_tag=scale_tag,
                        scale_fraction=float(selected_workers) / float(max(workload.max_worker_count, 1)),
                    )
                )
    # Keep the action slate deterministic but policy-neutral. PPO should learn
    # over the full candidate set rather than inheriting a heuristic pre-rank
    # from the environment itself.
    candidates.sort(
        key=lambda item: (
            -item.wait_seconds,
            -item.priority,
            item.workload_id,
            item.flavor_name,
            item.worker_count,
            item.scale_tag,
            item.action_id,
        )
    )
    return candidates[:limit]


def _feature_ratio(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(denom)))


def workload_profile_token(workload_id: str, profile_name: str = "") -> str:
    token = str(profile_name or workload_id or "").strip().lower()
    return token


def workload_profile_anchor_scalar(workload_id: str, profile_name: str = "") -> float:
    token = workload_profile_token(workload_id, profile_name)
    return 1.0 if token.startswith("reserve-anchor-") else 0.0


def workload_profile_prefer_c_scalar(workload_id: str, profile_name: str = "") -> float:
    token = workload_profile_token(workload_id, profile_name)
    return 1.0 if "goodput-c-" in token or "etp-flex-c-" in token or "eep-flex-elastic-c-" in token else 0.0


def workload_profile_prefer_a_scalar(workload_id: str, profile_name: str = "") -> float:
    token = workload_profile_token(workload_id, profile_name)
    return 1.0 if "goodput-a-" in token else 0.0


def workload_runtime_multiplier(workload_id: str, flavor_name: str, profile_name: str = "") -> float:
    flavor = str(flavor_name or "").strip().lower()
    token = workload_profile_token(workload_id, profile_name)
    if "etp-flex-c-" in token:
        if flavor.endswith("-c"):
            return 1.0
        if flavor.endswith("-a"):
            return 1.2
    if "eet-flex-elastic" in token:
        if flavor.endswith("-c"):
            return 1.0
        if flavor.endswith("-a"):
            return 1.12
    if "eep-flex-elastic-c-" in token:
        if flavor.endswith("-c"):
            return 1.0
        if flavor.endswith("-a"):
            return 1.15
    if workload_profile_prefer_c_scalar(workload_id, profile_name) > 0.0:
        if flavor.endswith("-c"):
            return 1.0
        if flavor.endswith("-a"):
            return 4.0
    if workload_profile_prefer_a_scalar(workload_id, profile_name) > 0.0:
        if flavor.endswith("-a"):
            return 1.0
        if flavor.endswith("-c"):
            return 4.0
    return 1.0


def workload_profile_flavor_match_scalar(workload_id: str, flavor_name: str, profile_name: str = "") -> float:
    has_profile = (
        workload_profile_prefer_c_scalar(workload_id, profile_name) > 0.0
        or workload_profile_prefer_a_scalar(workload_id, profile_name) > 0.0
    )
    if not has_profile:
        return 0.0
    return 1.0 if workload_runtime_multiplier(workload_id, flavor_name, profile_name) <= 1.0 else 0.0


def workload_profile_flavor_mismatch_scalar(workload_id: str, flavor_name: str, profile_name: str = "") -> float:
    has_profile = (
        workload_profile_prefer_c_scalar(workload_id, profile_name) > 0.0
        or workload_profile_prefer_a_scalar(workload_id, profile_name) > 0.0
    )
    if not has_profile:
        return 0.0
    return 1.0 if workload_runtime_multiplier(workload_id, flavor_name, profile_name) > 1.0 else 0.0


def workload_profile_critical_scalar(workload_id: str, profile_name: str = "") -> float:
    token = workload_profile_token(workload_id, profile_name)
    return 1.0 if "critical" in token else 0.0


def workload_external_provision_only(workload_id: str, profile_name: str = "") -> bool:
    token = workload_profile_token(workload_id, profile_name)
    return "etp-critical-c-head" in token or "eet-critical-c-head" in token or "eep-critical-c-head" in token


def kueue_state_vector(
    *,
    candidates: list[CandidateAction],
    waiting: list[KueueWorkload],
    running: list[RunningWorkload],
    future: list[KueueWorkload],
    nodes: list[NodeState],
    time_now: float,
    blocked_seconds: float,
    idle_quota_while_blocked: float,
    fair_share_violations: int,
) -> np.ndarray:
    context = blocked_head_context(candidates)
    queue_vec: list[float] = []
    for index in range(MAX_QUEUE_JOBS):
        if index >= len(candidates):
            queue_vec.extend([0.0] * JOB_FEAT_DIM)
            continue
        item = candidates[index]
        queue_vec.extend(
            [
                _feature_ratio(item.total_gpu, MAX_JOB_GPU),
                _feature_ratio(item.per_worker_gpu, MAX_NODE_GPU),
                _feature_ratio(item.worker_count, 16.0),
                _feature_ratio(item.runtime_seconds, MAX_JOB_DURATION),
                _feature_ratio(item.wait_seconds, MAX_WAIT_TIME),
                _feature_ratio(item.priority, 10.0),
                1.0 if item.queue_class == "gang" else 0.0,
                1.0 if item.topology_aware else 0.0,
                1.0 if item.immediate_fit else 0.5 if item.provisionable else 0.0,
                _feature_ratio(item.flavor_gpu_size, MAX_NODE_GPU),
                _feature_ratio(item.oversize_gpu, MAX_NODE_GPU),
                _feature_ratio(item.competing_older_pressure, 8.0),
                _feature_ratio(item.available_gpu, MAX_JOB_GPU),
                _feature_ratio(item.total_gpu_capacity, MAX_JOB_GPU),
                _feature_ratio(max(0, item.total_gpu_capacity - item.available_gpu), MAX_JOB_GPU),
                1.0 if candidate_one_block_short(item) else 0.0,
                workload_profile_anchor_scalar(item.workload_id, item.profile_name),
                workload_profile_prefer_c_scalar(item.workload_id, item.profile_name),
                workload_profile_prefer_a_scalar(item.workload_id, item.profile_name),
                topology_scalar(item.flavor_domain),
                workload_profile_flavor_match_scalar(item.workload_id, item.flavor_name, item.profile_name),
                workload_profile_flavor_mismatch_scalar(item.workload_id, item.flavor_name, item.profile_name),
                1.0 if item.elastic_enabled else 0.0,
                max(0.0, min(1.0, item.scale_fraction)),
                _feature_ratio(item.min_worker_count, 16.0),
                _feature_ratio(item.preferred_worker_count, 16.0),
                _feature_ratio(item.max_worker_count, 16.0),
            ]
        )

    node_vec: list[float] = []
    runnable_domains = {item.flavor_domain for item in candidates if item.immediate_fit}
    for index in range(MAX_CLUSTER_NODES):
        if index >= len(nodes):
            node_vec.extend([0.0] * NODE_FEAT_DIM)
            continue
        node = nodes[index]
        node_vec.extend(
            [
                _feature_ratio(node.free_cpu, max(node.cpu_total, 1)),
                _feature_ratio(node.free_mem, max(node.mem_total, 1)),
                _feature_ratio(node.free_gpu, max(node.gpu_total, 1)),
                _feature_ratio(node.gpu_total, MAX_NODE_GPU),
                _feature_ratio(node.free_gpu, MAX_NODE_GPU),
                topology_scalar(node.domain),
                1.0 if node.domain in runnable_domains else 0.0,
            ]
        )

    blocked_workloads = sum(1 for item in waiting if item.is_gang)
    workload_group_count = max(len(context.get("workload_groups", {}) or {}), 1)
    cluster_vec = [
        _feature_ratio(time_now, MAX_TIME_HORIZON),
        _feature_ratio(len(waiting), max(len(waiting) + len(future) + len(running), 1)),
        _feature_ratio(len(running), max(len(waiting) + len(future) + len(running), 1)),
        _feature_ratio(blocked_workloads, max(len(waiting), 1)),
        cluster_gpu_fragmentation(nodes),
        _feature_ratio(idle_quota_while_blocked, MAX_NODE_GPU * max(len(nodes), 1)),
        _feature_ratio(fair_share_violations, 10.0),
        _feature_ratio(float(context.get("blocked_provisionable", 0) or 0), float(workload_group_count)),
        _feature_ratio(float(context.get("disjoint_immediate_fit", 0) or 0), float(workload_group_count)),
    ]
    state = np.array(queue_vec + node_vec + cluster_vec, dtype=np.float32)
    assert state.shape == (STATE_DIM,), f"kueue state dim mismatch: {state.shape} != {(STATE_DIM,)}"
    return state


class KueueAdmissionEnv:
    def __init__(
        self,
        seed: int,
        num_jobs: int,
        arrival_span: float,
        workload_preset: str = "kueue-lingjun-gang-starvation-cohort",
        cluster_layout: str | None = None,
        trace_split: str = "all",
        trace_train_fraction: float = 0.75,
    ):
        self.seed = seed
        self.num_jobs = num_jobs
        self.arrival_span = arrival_span
        self.workload_preset = canonical_kueue_preset(workload_preset)
        self.cluster_layout = cluster_layout or default_cluster_layout_for_kueue_preset(self.workload_preset)
        self.trace_split = trace_split
        self.trace_train_fraction = trace_train_fraction
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.time = 0.0
        self.nodes = kueue_nodes_for_layout(self.cluster_layout)
        self.future_workloads = build_kueue_workloads(
            seed=self.seed,
            num_jobs=self.num_jobs,
            arrival_span=self.arrival_span,
            workload_preset=self.workload_preset,
            trace_split=self.trace_split,
            trace_train_fraction=self.trace_train_fraction,
        )
        self.all_workloads = list(self.future_workloads)
        self.waiting: list[KueueWorkload] = []
        self.running: list[RunningWorkload] = []
        self.completed: list[RunningWorkload] = []
        self.pending_provision: list[ProvisionEvent] = []
        self.total_reward = 0.0
        self.time_weighted_fragmentation = 0.0
        self.time_weighted_utilization = 0.0
        self.time_weighted_largest_block = 0.0
        self.blocked_seconds = 0.0
        self.idle_quota_while_blocked = 0.0
        self.fair_share_violation_count = 0
        self.topology_requests = 0
        self.topology_hits = 0
        self.provisioning_delays: list[float] = []
        self.fairshare = KueueFairShareState()
        self._seed_benchmark_sparse_capacity()
        self._release_arrivals()
        self._auto_advance_until_runnable()
        return self.observe()

    def _seed_benchmark_sparse_capacity(self):
        if self.workload_preset not in {
            "kueue-lingjun-gang-topology-provisioning",
            "kueue-lingjun-gang-elastic-topology",
            "kueue-lingjun-gang-elastic-profile-cohort",
        }:
            return
        c_nodes = [node for node in self.nodes if _node_flavor(node) == "rf-4gpu-c"]
        if self.workload_preset == "kueue-lingjun-gang-topology-provisioning":
            if len(c_nodes) <= 1:
                return
            retained = c_nodes[0]
            self.nodes = [node for node in self.nodes if _node_flavor(node) != "rf-4gpu-c" or node.name == retained.name]
        elif len(c_nodes) <= 0:
            return
        first_arrival = min((item.arrival_time for item in self.future_workloads), default=0.0)
        delay = 8.0
        self.pending_provision.append(
            ProvisionEvent(
                complete_at=first_arrival + delay,
                flavor_name="rf-4gpu-c",
                nodes_to_add=1,
            )
        )
        self.provisioning_delays.append(delay)

    def done(self) -> bool:
        return not self.future_workloads and not self.waiting and not self.running and not self.pending_provision

    def _release_arrivals(self):
        while self.future_workloads and self.future_workloads[0].arrival_time <= self.time:
            self.waiting.append(self.future_workloads.pop(0))

    def _update_integrals(self, dt: float):
        if dt <= 0:
            return
        self.time_weighted_fragmentation += cluster_gpu_fragmentation(self.nodes) * dt
        self.time_weighted_utilization += cluster_gpu_utilization(self.nodes) * dt
        self.time_weighted_largest_block += largest_free_gpu_block(self.nodes) * dt
        if self.waiting:
            gang_waiting = [item for item in self.waiting if item.is_gang]
            if gang_waiting:
                self.blocked_seconds += dt
                self.idle_quota_while_blocked += max(0.0, float(sum(node.free_gpu for node in self.nodes))) * dt

    def _running_speed(self, run: RunningWorkload) -> float:
        return elastic_speedup(run.admitted_worker_count, run.workload.max_worker_count)

    def _refresh_running_progress(self, run: RunningWorkload, until_time: float) -> None:
        elapsed = max(0.0, float(until_time) - float(run.last_progress_at))
        if elapsed <= 0:
            return
        run.remaining_work = max(0.0, float(run.remaining_work) - (self._running_speed(run) * elapsed))
        run.last_progress_at = float(until_time)
        run.end_time = float(until_time) + (run.remaining_work / max(self._running_speed(run), 1e-6))

    def _try_expand_elastic_running(self) -> float:
        reward = 0.0
        for run in list(self.running):
            if run.expanded or not run.workload.is_elastic:
                continue
            target_workers = max(run.workload.preferred_worker_count, run.workload.max_worker_count)
            if run.admitted_worker_count >= target_workers:
                run.expanded = True
                continue
            delta_workers = max(0, target_workers - run.admitted_worker_count)
            if delta_workers <= 0:
                run.expanded = True
                continue
            delta_shape = workload_for_scale(run.workload, delta_workers)
            extra_allocations = _heuristic_worker_allocations(delta_shape, run.flavor_name, self.nodes)
            if extra_allocations is None:
                continue
            self._refresh_running_progress(run, self.time)
            for node_name in extra_allocations:
                node = next(node for node in self.nodes if node.name == node_name)
                node.allocate(
                    type(
                        "_Tmp",
                        (),
                        {
                            "cpu_milli": run.workload.per_worker_cpu_milli,
                            "mem_bytes": run.workload.per_worker_mem_bytes,
                            "gpu": run.workload.per_worker_gpu,
                        },
                    )()
                )
            run.allocations.extend(extra_allocations)
            run.admitted_worker_count = target_workers
            run.expanded = True
            run.end_time = self.time + (run.remaining_work / max(self._running_speed(run), 1e-6))
            reward += 0.20
        return reward

    def _advance_time(self, next_time: float) -> float:
        dt = max(0.0, next_time - self.time)
        self._update_integrals(dt)
        wait_pressure = 0.0
        for item in self.waiting:
            item_pressure = 1.10 if item.is_gang else 0.60
            if item.is_elastic:
                item_pressure += 0.80
            if item.topology_aware:
                item_pressure += 0.25
            if workload_profile_critical_scalar(item.workload_id, item.profile_name) > 0.0:
                item_pressure += 0.35
            wait_pressure += item_pressure
        reward = -(dt * (wait_pressure + (0.20 * len(self.future_workloads)) + (0.35 * len(self.running)))) / max(
            float(self.num_jobs), 1.0
        )
        reward += dt * cluster_gpu_utilization(self.nodes) * 0.08
        reward -= dt * cluster_gpu_fragmentation(self.nodes) * 0.03
        self.time = next_time

        completed: list[RunningWorkload] = []
        for run in list(self.running):
            self._refresh_running_progress(run, self.time)
            if run.end_time <= self.time:
                for node_name in run.allocations:
                    node = next(node for node in self.nodes if node.name == node_name)
                    node.release(
                        type(
                            "_Tmp",
                            (),
                            {
                                "cpu_milli": run.workload.per_worker_cpu_milli,
                                "mem_bytes": run.workload.per_worker_mem_bytes,
                                "gpu": run.workload.per_worker_gpu,
                            },
                        )()
                    )
                self.running.remove(run)
                self.completed.append(run)
                self.fairshare.completed[run.workload.fairshare_group] = self.fairshare.completed.get(run.workload.fairshare_group, 0) + 1
                completed.append(run)
        for run in completed:
            completion_seconds = max(0.0, run.end_time - run.workload.arrival_time)
            completion_urgency = max(0.0, 1.0 - _feature_ratio(completion_seconds, MAX_TIME_HORIZON))
            runtime_shortness = max(0.0, 1.0 - _feature_ratio(run.workload.runtime_seconds, MAX_JOB_DURATION))
            reward += 1.2
            if run.workload.is_gang:
                reward += 0.6
            else:
                reward += 0.20
            reward += _feature_ratio(run.workload.total_gpu, MAX_NODE_GPU) * 0.25
            if run.provisioning_delay <= 0.0:
                reward += 0.12
            if self.workload_preset == "kueue-lingjun-gang-topology-provisioning":
                reward += completion_urgency * (0.25 if run.workload.is_gang else 0.10)

        for event in list(self.pending_provision):
            if event.complete_at <= self.time:
                sample = next((node for node in self.nodes if _node_flavor(node) == event.flavor_name), None)
                if sample is not None:
                    for index in range(event.nodes_to_add):
                        new_node = NodeState(
                            name=f"{event.flavor_name}-prov-{int(self.time)}-{index}",
                            domain=sample.domain,
                            cpu_total=sample.cpu_total,
                            mem_total=sample.mem_total,
                            gpu_total=sample.gpu_total,
                        )
                        self.nodes.append(new_node)
                self.pending_provision.remove(event)
                reward -= 0.05 * event.nodes_to_add

        self._release_arrivals()
        reward += self._try_expand_elastic_running()
        return reward

    def _next_event_time(self) -> float | None:
        times: list[float] = []
        if self.future_workloads:
            times.append(self.future_workloads[0].arrival_time)
        if self.running:
            times.append(min(item.end_time for item in self.running))
        if self.pending_provision:
            times.append(min(item.complete_at for item in self.pending_provision))
        if not times:
            return None
        return min(times)

    def _runnable_actions(self) -> list[CandidateAction]:
        return [item for item in self.candidate_actions() if item.immediate_fit or item.provisionable]

    def _drop_permanently_blocked_workload(self) -> float:
        if not self.waiting:
            return 0.0
        runnable_ids = {item.workload_id for item in self._runnable_actions()}
        blocked = [item for item in self.waiting if item.workload_id not in runnable_ids]
        if not blocked:
            return 0.0
        blocked.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
        victim = blocked[0]
        self.waiting.remove(victim)
        penalty = 1.5 if victim.is_gang else 1.0
        self.total_reward -= penalty
        return -penalty

    def _auto_advance_until_runnable(self) -> float:
        reward = 0.0
        while not self.done():
            if self._runnable_actions():
                break
            next_time = self._next_event_time()
            if next_time is None or next_time <= self.time:
                drop_reward = self._drop_permanently_blocked_workload()
                reward += drop_reward
                if drop_reward == 0.0:
                    break
                continue
            reward += self._advance_time(next_time)
        return reward

    def candidate_actions(self) -> list[CandidateAction]:
        return build_candidate_actions(self.waiting, self.nodes, self.time, self.fairshare, limit=MAX_QUEUE_JOBS)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(MAX_QUEUE_JOBS, dtype=np.float32)
        candidates = self.candidate_actions()
        for index, item in enumerate(candidates):
            if item.immediate_fit or item.provisionable:
                mask[index] = 1.0
        return mask

    def observe(self) -> np.ndarray:
        return kueue_state_vector(
            candidates=self.candidate_actions(),
            waiting=self.waiting,
            running=self.running,
            future=self.future_workloads,
            nodes=self.nodes,
            time_now=self.time,
            blocked_seconds=self.blocked_seconds,
            idle_quota_while_blocked=self.idle_quota_while_blocked,
            fair_share_violations=self.fair_share_violation_count,
        )

    def _provision_delay(self, workload: KueueWorkload, flavor_name: str) -> float:
        base = 90.0 if workload.queue_class == "small" else 180.0
        if flavor_name.startswith("rf-8gpu"):
            base += 120.0
        elif flavor_name.startswith("rf-6gpu"):
            base += 60.0
        return base

    def _allocate_workload(self, workload: KueueWorkload, flavor_name: str) -> tuple[list[str], float, float]:
        allocations = _heuristic_worker_allocations(workload, flavor_name, self.nodes)
        delay = 0.0
        reward = 0.0
        if allocations is None:
            delay = self._provision_delay(workload, flavor_name)
            sample = next((node for node in self.nodes if _node_flavor(node) == flavor_name), None)
            if sample is None:
                raise ValueError(f"no nodes for flavor {flavor_name}")
            additional_nodes = 0
            while allocations is None:
                additional_nodes += 1
                event = ProvisionEvent(complete_at=self.time + delay, flavor_name=flavor_name, nodes_to_add=1)
                self.pending_provision.append(event)
                self.provisioning_delays.append(delay)
                reward -= 0.1
                reward += self._advance_time(event.complete_at)
                allocations = _heuristic_worker_allocations(workload, flavor_name, self.nodes)
                if allocations is not None:
                    break
                if additional_nodes >= max(32, workload.worker_count * 2):
                    raise ValueError(f"provisioned flavor {flavor_name} repeatedly but workload {workload.workload_id} still does not fit")
        for node_name in allocations:
            node = next(node for node in self.nodes if node.name == node_name)
            node.allocate(
                type(
                    "_Tmp",
                    (),
                    {
                        "cpu_milli": workload.per_worker_cpu_milli,
                        "mem_bytes": workload.per_worker_mem_bytes,
                        "gpu": workload.per_worker_gpu,
                    },
                )()
            )
        return allocations, delay, reward

    def schedule_job(self, queue_index: int) -> tuple[float, bool, dict]:
        candidates = self.candidate_actions()
        if queue_index >= len(candidates):
            raise IndexError(queue_index)
        selected = candidates[queue_index]
        workload_index = next((index for index, item in enumerate(self.waiting) if item.workload_id == selected.workload_id), -1)
        if workload_index < 0:
            raise ValueError(f"selected workload {selected.workload_id} no longer waiting")
        workload = self.waiting.pop(workload_index)
        admitted_workload = workload_for_scale(workload, selected.worker_count)
        wait_seconds = max(0.0, self.time - workload.arrival_time)
        allocations, provisioning_delay, alloc_reward = self._allocate_workload(admitted_workload, selected.flavor_name)
        topology_hit = workload.topology_aware and workload.topology_preference and selected.flavor_domain == workload.topology_preference
        start_time = self.time
        effective_runtime_seconds = elastic_expected_runtime(workload, selected.flavor_name, selected.worker_count)
        remaining_work = elastic_total_work_units(workload, selected.flavor_name)
        end_time = start_time + (remaining_work / max(elastic_speedup(selected.worker_count, workload.max_worker_count), 1e-6))
        run = RunningWorkload(
            workload=workload,
            flavor_name=selected.flavor_name,
            allocations=allocations,
            start_time=start_time,
            end_time=end_time,
            topology_hit=topology_hit,
            provisioning_delay=provisioning_delay,
            admitted_worker_count=selected.worker_count,
            remaining_work=remaining_work,
            last_progress_at=start_time,
        )
        self.running.append(run)
        self.fairshare.launched[workload.fairshare_group] = self.fairshare.launched.get(workload.fairshare_group, 0) + 1
        if workload.topology_aware:
            self.topology_requests += 1
            if topology_hit:
                self.topology_hits += 1

        active_by_group = {
            group: self.fairshare.launched.get(group, 0) - self.fairshare.completed.get(group, 0)
            for group in {item.fairshare_group for item in self.waiting} | {workload.fairshare_group}
        }
        if active_by_group:
            mean_active = sum(active_by_group.values()) / max(len(active_by_group), 1)
            if active_by_group.get(workload.fairshare_group, 0) > mean_active + 2.0:
                self.fair_share_violation_count += 1

        reward = alloc_reward
        reward -= _feature_ratio(wait_seconds, MAX_WAIT_TIME) * (1.05 if workload.is_gang else 0.90)
        reward -= _feature_ratio(provisioning_delay, MAX_WAIT_TIME) * 0.4
        reward += 0.25 if selected.immediate_fit else 0.0
        reward += 0.30 if admitted_workload.is_gang else 0.10
        reward += 0.18 if selected.flavor_gpu_size == admitted_workload.per_worker_gpu else 0.0
        reward -= _feature_ratio(selected.oversize_gpu, MAX_NODE_GPU) * 0.12
        reward -= _feature_ratio(selected.competing_older_pressure, 8.0) * _feature_ratio(selected.oversize_gpu, MAX_NODE_GPU) * 0.75
        if workload.is_elastic:
            if selected.scale_tag == "min":
                reward += 0.35 if selected.competing_older_pressure > 0.0 else -0.08
            elif selected.scale_tag == "max" and selected.competing_older_pressure > 0.0:
                reward -= 0.85
            reward += (selected.scale_fraction - 0.5) * (0.14 if selected.competing_older_pressure <= 0.0 else 0.06)
        profile_prefers_specific_flavor = (
            workload_profile_prefer_a_scalar(workload.workload_id, workload.profile_name) > 0.0
            or workload_profile_prefer_c_scalar(workload.workload_id, workload.profile_name) > 0.0
        )
        profile_multiplier = workload_runtime_multiplier(
            workload.workload_id,
            selected.flavor_name,
            workload.profile_name,
        )
        if workload.is_elastic and profile_prefers_specific_flavor:
            if profile_multiplier <= 1.0:
                reward += 0.75
                if selected.scale_tag in {"preferred", "max"}:
                    reward += 0.18
            else:
                reward -= min(2.2, 0.70 * (profile_multiplier - 1.0))
                if selected.scale_tag == "min":
                    reward -= 0.30
        if workload.is_elastic:
            runtime_gain = max(0.0, 1.0 - (effective_runtime_seconds / max(workload.runtime_seconds, 1e-6)))
            reward += runtime_gain * (0.22 if selected.competing_older_pressure > 0.0 else 0.38)
        if workload_profile_prefer_a_scalar(workload.workload_id, workload.profile_name) > 0.0:
            if selected.scale_tag == "min":
                reward -= 0.55
            elif selected.scale_tag == "max":
                reward += 0.20
            if selected.flavor_domain == "A":
                reward += 0.30
            elif selected.flavor_domain == "C":
                reward -= 0.45
        if not admitted_workload.is_gang:
            reward -= _feature_ratio(selected.competing_older_pressure, 8.0) * 0.25
            if selected.immediate_fit and selected.flavor_gpu_size == workload.per_worker_gpu:
                reward += 0.12
            if provisioning_delay <= 0.0:
                reward += 0.08
        profile_token = workload_profile_token(workload.workload_id, workload.profile_name)
        if "etp-flex-c-" in profile_token and selected.competing_older_pressure > 0.0:
            pressure = _feature_ratio(selected.competing_older_pressure, 4.0)
            if selected.flavor_domain == "C":
                reward -= 1.10 + (0.55 * pressure)
            elif selected.flavor_domain == "A":
                reward += 0.90 + (0.30 * pressure)
        if "eet-flex-elastic" in profile_token and selected.competing_older_pressure > 0.0:
            pressure = _feature_ratio(selected.competing_older_pressure, 4.0)
            if selected.flavor_domain == "C":
                reward -= 1.60 + (0.75 * pressure)
                if selected.scale_tag == "max":
                    reward -= 0.55
            elif selected.flavor_domain == "A":
                reward += 1.20 + (0.45 * pressure)
                if selected.scale_tag == "min":
                    reward += 0.35
        if "eep-flex-elastic-c-" in profile_token and selected.competing_older_pressure > 0.0:
            pressure = _feature_ratio(selected.competing_older_pressure, 4.0)
            if selected.flavor_domain == "C":
                reward -= 1.60 + (0.70 * pressure)
                if selected.scale_tag == "max":
                    reward -= 0.50
            elif selected.flavor_domain == "A":
                reward += 1.25 + (0.45 * pressure)
                if selected.scale_tag == "min":
                    reward += 0.30
        if "etp-critical-c-head" in profile_token and selected.immediate_fit:
            reward += 0.85
        if "eet-critical-c-head" in profile_token and selected.immediate_fit:
            reward += 1.00
        if "eep-critical-c-head" in profile_token and selected.immediate_fit:
            reward += 1.00
        if topology_hit:
            reward += 0.35 + (0.05 * workload.worker_count)
            if workload.is_gang and workload.topology_aware:
                reward += 0.25
        elif workload.topology_aware:
            reward -= 0.20 + (0.15 if workload.is_gang else 0.0)

        reward += self._auto_advance_until_runnable()
        self.total_reward += reward
        return reward, self.done(), {
            "workload_id": workload.workload_id,
            "flavor_name": selected.flavor_name,
            "worker_count": admitted_workload.worker_count,
            "queue_class": admitted_workload.queue_class,
            "topology_hit": topology_hit,
            "scale_tag": selected.scale_tag,
        }

    def summary(self) -> dict:
        finished = self.completed
        wait_values = [max(0.0, item.start_time - item.workload.arrival_time) for item in finished]
        completion_values = [max(0.0, item.end_time - item.workload.arrival_time) for item in finished]
        gang_wait_values = [max(0.0, item.start_time - item.workload.arrival_time) for item in finished if item.workload.is_gang]
        small_wait_values = [max(0.0, item.start_time - item.workload.arrival_time) for item in finished if not item.workload.is_gang]
        gang_completion_values = [max(0.0, item.end_time - item.workload.arrival_time) for item in finished if item.workload.is_gang]
        small_completion_values = [max(0.0, item.end_time - item.workload.arrival_time) for item in finished if not item.workload.is_gang]
        topology_wait_values = [
            max(0.0, item.start_time - item.workload.arrival_time)
            for item in finished
            if item.workload.topology_aware
        ]
        topology_completion_values = [
            max(0.0, item.end_time - item.workload.arrival_time)
            for item in finished
            if item.workload.topology_aware
        ]
        critical_wait_values = [
            max(0.0, item.start_time - item.workload.arrival_time)
            for item in finished
            if workload_profile_critical_scalar(item.workload.workload_id, item.workload.profile_name) > 0.0
        ]
        critical_completion_values = [
            max(0.0, item.end_time - item.workload.arrival_time)
            for item in finished
            if workload_profile_critical_scalar(item.workload.workload_id, item.workload.profile_name) > 0.0
        ]
        elastic_wait_values = [
            max(0.0, item.start_time - item.workload.arrival_time)
            for item in finished
            if item.workload.is_elastic
        ]
        elastic_completion_values = [
            max(0.0, item.end_time - item.workload.arrival_time)
            for item in finished
            if item.workload.is_elastic
        ]
        critical_total = sum(
            1 for item in self.all_workloads if workload_profile_critical_scalar(item.workload_id, item.profile_name) > 0.0
        )
        critical_finished = sum(
            1
            for item in finished
            if workload_profile_critical_scalar(item.workload.workload_id, item.workload.profile_name) > 0.0
        )
        gang_total = sum(1 for item in self.all_workloads if item.is_gang)
        gang_finished = sum(1 for item in finished if item.workload.is_gang)
        elastic_total = sum(1 for item in self.all_workloads if item.is_elastic)
        elastic_finished = sum(1 for item in finished if item.workload.is_elastic)
        elastic_initial_scale_fraction = [
            float(item.admitted_worker_count) / float(max(item.workload.max_worker_count, 1))
            for item in finished
            if item.workload.is_elastic
        ]
        elastic_expanded = sum(1 for item in finished if item.workload.is_elastic and item.expanded)
        topology_hit_rate = (self.topology_hits / self.topology_requests) if self.topology_requests else 0.0
        duration = max(self.time, 1.0)
        total_gpu_completed = float(sum(item.workload.total_gpu for item in finished))
        return {
            "jobs_total": len(self.all_workloads),
            "jobs_completed": len(finished),
            "avg_workload_wait_seconds": float(np.mean(wait_values)) if wait_values else 0.0,
            "avg_gang_wait_seconds": float(np.mean(gang_wait_values)) if gang_wait_values else 0.0,
            "p95_gang_wait_seconds": percentile(gang_wait_values, 95),
            "avg_small_wait_seconds": float(np.mean(small_wait_values)) if small_wait_values else 0.0,
            "avg_topology_aware_wait_seconds": float(np.mean(topology_wait_values)) if topology_wait_values else 0.0,
            "avg_critical_wait_seconds": float(np.mean(critical_wait_values)) if critical_wait_values else 0.0,
            "avg_elastic_wait_seconds": float(np.mean(elastic_wait_values)) if elastic_wait_values else 0.0,
            "p95_elastic_wait_seconds": percentile(elastic_wait_values, 95),
            "p95_workload_wait_seconds": percentile(wait_values, 95),
            "p99_workload_wait_seconds": percentile(wait_values, 99),
            "avg_job_completion_seconds": float(np.mean(completion_values)) if completion_values else 0.0,
            "avg_gang_completion_seconds": float(np.mean(gang_completion_values)) if gang_completion_values else 0.0,
            "p95_gang_completion_seconds": percentile(gang_completion_values, 95),
            "avg_small_completion_seconds": float(np.mean(small_completion_values)) if small_completion_values else 0.0,
            "avg_topology_aware_completion_seconds": float(np.mean(topology_completion_values)) if topology_completion_values else 0.0,
            "avg_critical_completion_seconds": float(np.mean(critical_completion_values)) if critical_completion_values else 0.0,
            "avg_elastic_completion_seconds": float(np.mean(elastic_completion_values)) if elastic_completion_values else 0.0,
            "p95_elastic_completion_seconds": percentile(elastic_completion_values, 95),
            "p95_job_completion_seconds": percentile(completion_values, 95),
            "p99_job_completion_seconds": percentile(completion_values, 99),
            "gang_admission_ratio": (gang_finished / gang_total) if gang_total else 0.0,
            "critical_completion_ratio": (critical_finished / critical_total) if critical_total else 0.0,
            "elastic_completion_ratio": (elastic_finished / elastic_total) if elastic_total else 0.0,
            "avg_elastic_initial_scale_fraction": float(np.mean(elastic_initial_scale_fraction)) if elastic_initial_scale_fraction else 0.0,
            "elastic_jobs_expanded": elastic_expanded,
            "topology_hit_rate": topology_hit_rate,
            "flavor_head_blocking_seconds": self.blocked_seconds,
            "idle_quota_while_blocked": self.idle_quota_while_blocked / duration,
            "avg_provisioning_delay_seconds": float(np.mean(self.provisioning_delays)) if self.provisioning_delays else 0.0,
            "makespan_seconds": duration,
            "throughput_jobs_per_minute": (len(finished) * 60.0) / duration,
            "throughput_gpu_per_minute": (total_gpu_completed * 60.0) / duration,
            "largest_free_gpu_block": largest_free_gpu_block(self.nodes),
            "gpu_fragmentation": cluster_gpu_fragmentation(self.nodes),
            "avg_gpu_fragmentation": self.time_weighted_fragmentation / duration,
            "avg_gpu_utilization": self.time_weighted_utilization / duration,
            "fair_share_violation_count": self.fair_share_violation_count,
            "class_mix": class_mix(self.all_workloads),
            "total_reward": self.total_reward,
        }
