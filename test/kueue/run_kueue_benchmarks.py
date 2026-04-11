from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_server.kueue_rl.kueue_admission import (
    KUEUE_CLUSTER_LAYOUTS,
    KueueWorkload,
    build_kueue_workloads,
    canonical_kueue_preset,
    is_kueue_preset,
    workload_for_scale,
)
from test.kueue.manifests import benchmark_setup_docs, layout_for_preset, live_worker_resource_requests, pod_group_docs, render_yaml


@dataclass(frozen=True)
class BenchmarkArm:
    name: str
    kueue_mode: str
    queueing_strategy: str
    runtime_policy: str = ""
    elastic_policy: str = "disabled"


ARMS = [
    BenchmarkArm("stock-best-effort-default", "stock", "BestEffortFIFO", runtime_policy="", elastic_policy="disabled"),
    BenchmarkArm("learned-best-effort-default", "learned", "BestEffortFIFO", runtime_policy="learned_multi_objective", elastic_policy="disabled"),
    BenchmarkArm("heuristic-elastic-default", "learned", "BestEffortFIFO", runtime_policy="blocked_guard", elastic_policy="blocked_guard"),
    BenchmarkArm("learned-elastic-default", "learned", "BestEffortFIFO", runtime_policy="learned_multi_objective", elastic_policy="learned_multi_objective"),
    BenchmarkArm("strict-default-sensitivity", "stock", "StrictFIFO", runtime_policy="", elastic_policy="disabled"),
]


def _write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def meta_row_to_workload(row: dict) -> KueueWorkload:
    return KueueWorkload(
        workload_id=str(row.get("workload_id", "")),
        queue_name=str(row.get("queue_name", "")),
        cluster_queue=str(row.get("cluster_queue", "training-cluster-queue")),
        fairshare_group=str(row.get("fairshare_group", "")),
        queue_class=str(row.get("queue_class", "small")),
        priority=int(row.get("priority", 0) or 0),
        worker_count=int(row.get("worker_count", 1) or 1),
        per_worker_gpu=int(row.get("per_worker_gpu", 1) or 1),
        per_worker_cpu_milli=int(row.get("per_worker_cpu_milli", 1000) or 1000),
        per_worker_mem_bytes=int(row.get("per_worker_mem_bytes", 2 * (1024**3)) or (2 * (1024**3))),
        total_gpu=int(row.get("total_gpu", 0) or 0),
        runtime_seconds=float(row.get("runtime_seconds", 60.0) or 60.0),
        arrival_time=float(row.get("arrival_time", 0.0) or 0.0),
        topology_aware=bool(row.get("topology_aware", False)),
        topology_preference=str(row.get("topology_preference", "")),
        restart_count=int(row.get("restart_count", 0) or 0),
        candidate_flavors=tuple(row.get("candidate_flavors", []) or []),
        elastic_enabled=bool(row.get("elastic_enabled", False)),
        min_worker_count=int(row.get("min_worker_count", row.get("worker_count", 1)) or 1),
        preferred_worker_count=int(row.get("preferred_worker_count", row.get("worker_count", 1)) or 1),
        max_worker_count=int(row.get("max_worker_count", row.get("worker_count", 1)) or 1),
        profile_name=str(row.get("profile_name", row.get("workload_id", ""))),
    )


def _flavor_name(group: dict) -> str:
    return f"rf-{group['gpus']}gpu-{group['domain'].lower()}"


def _live_feasible(workload, cluster_layout: str) -> bool:
    # Plain pod-group + TAS live runs currently fail to admit 2x8-GPU groups
    # reliably, even on clean clusters with enough nominal quota. Keep the live
    # benchmark on the subset of workload shapes that the current Kueue/TAS
    # integration can execute end to end.
    if workload.worker_count >= 2 and workload.per_worker_gpu >= 8:
        return False
    if workload.worker_count == 1 and workload.per_worker_gpu >= 8:
        return False
    request_cpu, request_mem_gi = live_worker_resource_requests(workload)
    total_cpu = request_cpu * workload.worker_count
    total_mem_gi = request_mem_gi * workload.worker_count
    total_gpu = workload.per_worker_gpu * workload.worker_count
    quotas = {
        _flavor_name(group): {
            "cpu": max(1, int(group["count"] * group["cpu_milli"] // 1000)),
            "memory_gi": max(1, int(group["count"] * group["mem_bytes"] // (1024**3))),
            "gpu": max(1, int(group["count"] * group["gpus"])),
        }
        for group in KUEUE_CLUSTER_LAYOUTS[cluster_layout]
    }
    for flavor in workload.candidate_flavors:
        quota = quotas.get(flavor)
        if quota is None:
            continue
        if quota["gpu"] >= total_gpu and quota["cpu"] >= total_cpu and quota["memory_gi"] >= total_mem_gi:
            return True
    return False


def generate_arm_artifacts(
    *,
    output_root: Path,
    arm: BenchmarkArm,
    workload_preset: str,
    cluster_layout: str,
    seed: int,
    num_jobs: int,
    arrival_span: float,
    trace_split: str,
    trace_train_fraction: float,
    runtime_scale: float,
    include_provisioning: bool,
    namespace: str,
    live_feasible_only: bool = False,
    oversample_factor: int = 1,
):
    arm_dir = output_root / arm.name
    cluster_queue_name = f"cq-{namespace}"[:63]
    cohort_name = f"cohort-{namespace}"[:63]
    workloads = build_kueue_workloads(
        seed=seed,
        num_jobs=max(num_jobs, num_jobs * max(1, oversample_factor)),
        arrival_span=arrival_span,
        workload_preset=workload_preset,
        trace_split=trace_split,
        trace_train_fraction=trace_train_fraction,
    )
    if live_feasible_only:
        workloads = [workload for workload in workloads if _live_feasible(workload, cluster_layout)]
    workloads = workloads[:num_jobs]
    setup_yaml = render_yaml(
        benchmark_setup_docs(
            workloads,
            layout=cluster_layout,
            queueing_strategy=arm.queueing_strategy,
            namespace=namespace,
            include_provisioning=include_provisioning,
            cluster_queue_name=cluster_queue_name,
            cohort_name=cohort_name,
        )
    )
    _write_text(arm_dir / "setup.yaml", setup_yaml)

    workload_index = []
    use_initial_scale_override = any(workload.elastic_enabled for workload in workloads)
    for workload in workloads:
        if use_initial_scale_override:
            initial_workers = workload.worker_count
            elastic_enabled = False
            if workload.elastic_enabled:
                if arm.elastic_policy == "disabled":
                    initial_workers = workload.max_worker_count
                    elastic_enabled = False
                else:
                    initial_workers = workload.min_worker_count
                    elastic_enabled = True
            docs = pod_group_docs(
                workload_for_scale(workload, initial_workers),
                namespace=namespace,
                runtime_scale=runtime_scale,
            )
        else:
            docs = pod_group_docs(
                workload,
                namespace=namespace,
                runtime_scale=runtime_scale,
            )
        file_name = f"{workload.workload_id[:48].replace('/', '-')}.yaml"
        _write_text(arm_dir / "workloads" / file_name, render_yaml(docs))
        workload_index.append(
            {
                "file": file_name,
                "workload_id": workload.workload_id,
                "arrival_time": workload.arrival_time,
                "queue_name": workload.queue_name,
                "cluster_queue": workload.cluster_queue,
                "fairshare_group": workload.fairshare_group,
                "queue_class": workload.queue_class,
                "priority": workload.priority,
                "worker_count": workload.worker_count,
                "per_worker_gpu": workload.per_worker_gpu,
                "per_worker_cpu_milli": workload.per_worker_cpu_milli,
                "per_worker_mem_bytes": workload.per_worker_mem_bytes,
                "total_gpu": workload.total_gpu,
                "runtime_seconds": workload.runtime_seconds,
                "topology_aware": workload.topology_aware,
                "topology_preference": workload.topology_preference,
                "candidate_flavors": list(workload.candidate_flavors),
                "restart_count": workload.restart_count,
                "elastic_enabled": workload.elastic_enabled,
                "min_worker_count": workload.min_worker_count,
                "preferred_worker_count": workload.preferred_worker_count,
                "max_worker_count": workload.max_worker_count,
                "initial_worker_count": initial_workers if use_initial_scale_override else workload.worker_count,
                "initial_elastic_enabled": elastic_enabled if use_initial_scale_override else workload.elastic_enabled,
                "profile_name": workload.profile_name or workload.workload_id,
            }
        )

    meta = {
        "arm": asdict(arm),
        "workload_preset": workload_preset,
        "cluster_layout": cluster_layout,
        "seed": seed,
        "num_jobs": num_jobs,
        "arrival_span": arrival_span,
        "trace_split": trace_split,
        "trace_train_fraction": trace_train_fraction,
        "runtime_scale": runtime_scale,
        "include_provisioning": include_provisioning,
        "namespace": namespace,
        "cluster_queue_name": cluster_queue_name,
        "cohort_name": cohort_name,
        "workloads": workload_index,
    }
    _write_text(arm_dir / "meta.json", json.dumps(meta, indent=2))
    return arm_dir, meta


def _kubectl_apply(path: Path):
    subprocess.run(["kubectl", "apply", "-f", str(path)], check=True)


def submit_workloads_live(arm_dir: Path, meta: dict, time_scale: float):
    last_arrival = 0.0
    for item in sorted(meta["workloads"], key=lambda row: (row["arrival_time"], -row["total_gpu"], row["workload_id"])):
        sleep_seconds = max(0.0, (float(item["arrival_time"]) - last_arrival) / max(time_scale, 1e-6))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        _kubectl_apply(arm_dir / "workloads" / item["file"])
        last_arrival = float(item["arrival_time"])


def main():
    parser = argparse.ArgumentParser(description="Generate or run Kueue admission benchmark arms on a KWOK cluster")
    parser.add_argument("--workload-preset", default="kueue-lingjun-gang-starvation-cohort")
    parser.add_argument("--cluster-layout", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-jobs", type=int, default=32)
    parser.add_argument("--arrival-span", type=float, default=0.0)
    parser.add_argument("--trace-split", default="all")
    parser.add_argument("--trace-train-fraction", type=float, default=0.75)
    parser.add_argument("--output-root", type=Path, default=Path("test/results/kueue-benchmarks"))
    parser.add_argument("--arm", action="append", dest="arms", default=[], help="Optional arm name filter")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Replay divisor when --apply is used")
    parser.add_argument("--runtime-scale", type=float, default=60.0, help="Divide simulated runtime seconds by this value for live pod execution")
    parser.add_argument("--disable-provisioning", action="store_true", help="Omit provisioning admission checks for fixed-capacity live runs")
    parser.add_argument("--namespace", default="default", help="Namespace used for LocalQueues and plain pod groups")
    args = parser.parse_args()

    if not is_kueue_preset(args.workload_preset):
        raise SystemExit(f"unsupported Kueue workload preset: {args.workload_preset}")
    selected_arms = [arm for arm in ARMS if not args.arms or arm.name in set(args.arms)]
    if not selected_arms:
        raise SystemExit("no benchmark arms selected")
    if args.apply and len(selected_arms) != 1:
        raise SystemExit("--apply currently supports exactly one arm at a time to avoid Kueue object collisions")
    cluster_layout = layout_for_preset(canonical_kueue_preset(args.workload_preset), args.cluster_layout or None)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for arm in selected_arms:
        arm_dir, meta = generate_arm_artifacts(
            output_root=args.output_root,
            arm=arm,
            workload_preset=args.workload_preset,
            cluster_layout=cluster_layout,
            seed=args.seed,
            num_jobs=args.num_jobs,
            arrival_span=args.arrival_span,
            trace_split=args.trace_split,
            trace_train_fraction=args.trace_train_fraction,
            runtime_scale=args.runtime_scale,
            include_provisioning=not args.disable_provisioning,
            namespace=args.namespace,
        )
        summary[arm.name] = meta
        if args.apply:
            _kubectl_apply(arm_dir / "setup.yaml")
            submit_workloads_live(arm_dir, meta, time_scale=args.time_scale)

    _write_text(args.output_root / "index.json", json.dumps(summary, indent=2))
    print(json.dumps({"output_root": str(args.output_root), "arms": list(summary)}, indent=2))


if __name__ == "__main__":
    main()
