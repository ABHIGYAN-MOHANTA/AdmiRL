from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable


TRACE_ROOT = Path(__file__).resolve().parents[2] / "data" / "alibaba_lingjun_2023"
JOB_CSV = TRACE_ROOT / "job.csv"
WORKER_CSV = TRACE_ROOT / "worker.csv"
TOPO_CSV = TRACE_ROOT / "topo.csv"

GPU_RESOURCE_KEYS = ("nvidia.com/gpu", "admirl.ai/gpu", "gscore/gpu")
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


@dataclass(frozen=True)
class LingjunWorkerSpec:
    worker_name: str
    replica_type: str
    host_ip: str
    created_at: float
    started_at: float
    finished_at: float
    cpu_milli: int
    mem_bytes: int
    gpu: int
    rdma: int
    topology_bucket: str


@dataclass(frozen=True)
class LingjunWorkloadSpec:
    workload_id: str
    job_uid: str
    namespace: str
    kind: str
    model: str
    queue_class: str
    fairshare_group: str
    priority: int
    topology_aware: bool
    topology_preference: str
    arrival_time: float
    runtime_seconds: float
    worker_count: int
    total_cpu_milli: int
    total_mem_bytes: int
    total_gpu: int
    per_worker_cpu_milli: int
    per_worker_mem_bytes: int
    per_worker_gpu: int
    restart_count: int
    max_runtime_seconds: float
    candidate_flavors: tuple[str, ...] = ()


def _parse_timestamp(raw: str) -> float:
    value = (raw or "").strip()
    if not value:
        return 0.0
    return datetime.strptime(value, TIMESTAMP_FORMAT).timestamp()


def _parse_int(raw: str, default: int = 0) -> int:
    value = (raw or "").strip()
    if not value:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _parse_quantity(raw: str) -> int:
    value = (raw or "").strip()
    if not value:
        return 0
    suffixes = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
    }
    for suffix, scale in suffixes.items():
        if value.endswith(suffix):
            number = value[: -len(suffix)]
            return int(float(number) * scale)
    return int(float(value))


def _parse_worker_resources(raw: str) -> tuple[int, int, int, int]:
    payload = (raw or "").strip()
    if not payload:
        return 0, 0, 0, 0
    data = json.loads(payload)
    cpu_value = data.get("cpu", "0")
    mem_value = data.get("memory", "0")
    rdma_value = data.get("koordinator.sh/rdma", "0")
    gpu_value = 0
    for key in GPU_RESOURCE_KEYS:
        if key in data:
            gpu_value = _parse_int(str(data[key]))
            break
    cpu_milli = int(float(cpu_value) * 1000.0)
    mem_bytes = _parse_quantity(str(mem_value))
    rdma = _parse_int(str(rdma_value))
    return cpu_milli, mem_bytes, gpu_value, rdma


def _topology_bucket(host_ip: str, topo_map: dict[str, tuple[str, str, str]]) -> str:
    dsw, psw, asw = topo_map.get(host_ip, ("", "", ""))
    seed = next((item for item in (psw, asw, dsw) if item), "")
    if not seed:
        return ""
    return "ABCD"[sum(ord(ch) for ch in seed) % 4]


def _row_is_deleted(row: dict[str, str]) -> bool:
    return (row.get("is_deleted", "") or "").strip() == "1"


def _row_gpu_topology_enabled(row: dict[str, str]) -> bool:
    return (row.get("is_enable_gpu_topo_aware", "") or "").strip() == "1"


def _job_runtime_seconds(row: dict[str, str], workers: list[LingjunWorkerSpec]) -> float:
    submit = _parse_timestamp(row.get("gmt_job_submitted", ""))
    started = _parse_timestamp(row.get("gmt_job_running", ""))
    stopped = _parse_timestamp(row.get("gmt_job_finished", "")) or _parse_timestamp(row.get("gmt_job_stopped", ""))
    if submit and stopped and stopped >= submit:
        return max(60.0, stopped - submit)
    if started and stopped and stopped >= started:
        return max(60.0, stopped - started)
    worker_created = min((worker.created_at for worker in workers if worker.created_at > 0), default=0.0)
    worker_finished = max((worker.finished_at for worker in workers if worker.finished_at > 0), default=0.0)
    if worker_created and worker_finished and worker_finished >= worker_created:
        return max(60.0, worker_finished - worker_created)
    max_runtime_minutes = _parse_int(row.get("job_max_running_time_minutes", ""), 0)
    if max_runtime_minutes > 0:
        return max(60.0, float(max_runtime_minutes) * 60.0)
    return 600.0


def _job_arrival_time(row: dict[str, str], workers: list[LingjunWorkerSpec]) -> float:
    submitted = _parse_timestamp(row.get("gmt_job_submitted", ""))
    created = _parse_timestamp(row.get("gmt_created", ""))
    if submitted > 0:
        return submitted
    if created > 0:
        return created
    worker_created = min((worker.created_at for worker in workers if worker.created_at > 0), default=0.0)
    if worker_created > 0:
        return worker_created
    return 0.0


def _candidate_flavors(per_worker_gpu: int, worker_count: int, topology_bucket: str) -> tuple[str, ...]:
    flavors: list[str] = []
    for gpu in (2, 4, 6, 8):
        if per_worker_gpu <= gpu:
            domains = ("A", "B", "C", "D") if gpu <= 4 else ("C", "D")
            for domain in domains:
                flavors.append(f"rf-{gpu}gpu-{domain.lower()}")
    if topology_bucket:
        preferred = [name for name in flavors if name.endswith(f"-{topology_bucket.lower()}")]
        flavors = preferred + [name for name in flavors if name not in preferred]
    if worker_count >= 8:
        flavors = [name for name in flavors if name.startswith(("rf-6gpu", "rf-8gpu"))] or flavors
    return tuple(dict.fromkeys(flavors))


def _split_workloads(
    workloads: list[LingjunWorkloadSpec],
    trace_split: str,
    train_fraction: float,
) -> list[LingjunWorkloadSpec]:
    if trace_split == "all" or not workloads:
        return list(workloads)
    split_index = int(len(workloads) * train_fraction)
    split_index = max(1, min(len(workloads) - 1, split_index))
    if trace_split == "train":
        return workloads[:split_index]
    if trace_split == "test":
        return workloads[split_index:]
    raise ValueError(f"unsupported trace split: {trace_split}")


@lru_cache(maxsize=2)
def load_topology_map() -> dict[str, tuple[str, str, str]]:
    topo: dict[str, tuple[str, str, str]] = {}
    with TOPO_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            topo[row["ip"]] = (
                (row.get("DSW", "") or "").strip(),
                (row.get("PSW", "") or "").strip(),
                (row.get("ASW", "") or "").strip(),
            )
    return topo


@lru_cache(maxsize=2)
def load_worker_specs() -> dict[str, tuple[LingjunWorkerSpec, ...]]:
    topo_map = load_topology_map()
    workers_by_job: dict[str, list[LingjunWorkerSpec]] = {}
    with WORKER_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            job_name = (row.get("job_name", "") or "").strip()
            if not job_name:
                continue
            cpu_milli, mem_bytes, gpu, rdma = _parse_worker_resources(row.get("RES", ""))
            host_ip = (row.get("host_ip", "") or "").strip()
            workers_by_job.setdefault(job_name, []).append(
                LingjunWorkerSpec(
                    worker_name=(row.get("worker_name", "") or "").strip(),
                    replica_type=(row.get("replica_type", "") or "").strip().lower(),
                    host_ip=host_ip,
                    created_at=_parse_timestamp(row.get("gmt_created", "")),
                    started_at=_parse_timestamp(row.get("gmt_pod_running", "")),
                    finished_at=_parse_timestamp(row.get("gmt_pod_finished", "")),
                    cpu_milli=cpu_milli,
                    mem_bytes=mem_bytes,
                    gpu=gpu,
                    rdma=rdma,
                    topology_bucket=_topology_bucket(host_ip, topo_map),
                )
            )
    return {job_name: tuple(workers) for job_name, workers in workers_by_job.items()}


@lru_cache(maxsize=1)
def _load_all_lingjun_workloads() -> tuple[LingjunWorkloadSpec, ...]:
    workers_by_job = load_worker_specs()
    workloads: list[LingjunWorkloadSpec] = []
    with JOB_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if _row_is_deleted(row):
                continue
            job_name = (row.get("job_name", "") or "").strip()
            if not job_name:
                continue
            workers = list(workers_by_job.get(job_name, ()))
            if not workers:
                continue
            worker_count = len(workers)
            total_gpu = sum(worker.gpu for worker in workers)
            if total_gpu <= 0 or total_gpu > 64 or worker_count > 16:
                continue
            total_cpu = sum(worker.cpu_milli for worker in workers)
            total_mem = sum(worker.mem_bytes for worker in workers)
            per_worker_gpu = max(worker.gpu for worker in workers)
            per_worker_cpu = max(worker.cpu_milli for worker in workers)
            per_worker_mem = max(worker.mem_bytes for worker in workers)
            topology_votes = [worker.topology_bucket for worker in workers if worker.topology_bucket]
            topology_preference = max(topology_votes, key=topology_votes.count) if topology_votes else ""
            topology_aware = _row_gpu_topology_enabled(row) or (worker_count >= 2 and per_worker_gpu >= 4)
            queue_class = "gang" if worker_count >= 2 else "small"
            fairshare_group = (row.get("workspace_name", "") or row.get("namespace", "") or "default").strip() or "default"
            arrival_time = _job_arrival_time(row, workers)
            runtime_seconds = _job_runtime_seconds(row, workers)
            max_runtime_seconds = max(runtime_seconds, float(_parse_int(row.get("job_max_running_time_minutes", ""), 0) * 60))
            priority = _parse_int(row.get("priority", ""), 0)
            restart_count = _parse_int(row.get("job_restart_times", ""), 0)
            workloads.append(
                LingjunWorkloadSpec(
                    workload_id=job_name,
                    job_uid=(row.get("uid", "") or "").strip(),
                    namespace=(row.get("namespace", "") or "").strip(),
                    kind=(row.get("kind", "") or "").strip(),
                    model=(row.get("model", "") or "").strip(),
                    queue_class=queue_class,
                    fairshare_group=fairshare_group,
                    priority=priority,
                    topology_aware=topology_aware,
                    topology_preference=topology_preference,
                    arrival_time=arrival_time,
                    runtime_seconds=runtime_seconds,
                    worker_count=worker_count,
                    total_cpu_milli=total_cpu,
                    total_mem_bytes=total_mem,
                    total_gpu=total_gpu,
                    per_worker_cpu_milli=per_worker_cpu,
                    per_worker_mem_bytes=per_worker_mem,
                    per_worker_gpu=per_worker_gpu,
                    restart_count=restart_count,
                    max_runtime_seconds=max_runtime_seconds,
                    candidate_flavors=_candidate_flavors(per_worker_gpu, worker_count, topology_preference if topology_aware else ""),
                )
            )
    workloads.sort(key=lambda item: (item.arrival_time, -item.total_gpu, -item.worker_count, item.workload_id))
    return tuple(workloads)


@lru_cache(maxsize=8)
def load_lingjun_workloads(trace_split: str = "all", train_fraction: float = 0.75) -> tuple[LingjunWorkloadSpec, ...]:
    workloads = list(_load_all_lingjun_workloads())
    selected = _split_workloads(workloads, trace_split, train_fraction)
    return tuple(selected)


def sample_lingjun_workloads(
    *,
    seed: int,
    num_jobs: int,
    trace_split: str = "all",
    train_fraction: float = 0.75,
) -> list[LingjunWorkloadSpec]:
    workloads = list(load_lingjun_workloads(trace_split=trace_split, train_fraction=train_fraction))
    if not workloads:
        return []
    if len(workloads) <= num_jobs:
        base = workloads
    else:
        rng = random.Random(seed)
        window = max(num_jobs, min(len(workloads), int(num_jobs * 2.25)))
        start = rng.randint(0, max(0, len(workloads) - window))
        candidates = workloads[start : start + window]
        small = [item for item in candidates if item.queue_class == "small"]
        gang = [item for item in candidates if item.queue_class == "gang"]
        target_gang = min(max(1, num_jobs // 3), len(gang))
        target_small = min(num_jobs - target_gang, len(small))
        chosen = []
        if target_gang > 0:
            chosen.extend(rng.sample(gang, target_gang))
        remaining = num_jobs - len(chosen)
        pool = [item for item in candidates if item not in chosen]
        if remaining > 0:
            chosen.extend(rng.sample(pool, min(remaining, len(pool))))
        chosen.sort(key=lambda item: (item.arrival_time, -item.total_gpu, item.workload_id))
        base = chosen[:num_jobs]

    first_arrival = min((item.arrival_time for item in base), default=0.0)
    normalized: list[LingjunWorkloadSpec] = []
    for item in base:
        normalized.append(
            LingjunWorkloadSpec(
                **{
                    **item.__dict__,
                    "arrival_time": max(0.0, item.arrival_time - first_arrival),
                }
            )
        )
    return normalized


def class_mix(workloads: Iterable[LingjunWorkloadSpec]) -> dict[str, int]:
    summary = {"small": 0, "gang": 0}
    for workload in workloads:
        summary.setdefault(workload.queue_class, 0)
        summary[workload.queue_class] += 1
    return summary
