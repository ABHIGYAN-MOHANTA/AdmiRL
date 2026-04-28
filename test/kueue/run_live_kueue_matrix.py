from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_server.kueue_rl.cluster import NodeState
from model_server.kueue_rl.kueue_admission import (
    build_candidate_actions,
    canonical_kueue_preset,
    is_kueue_preset,
    workload_for_scale,
)
from model_server.admirl_server.kueue_runtime import build_kueue_admission_response, load_runtime_policy
from test.kueue.manifests import layout_for_preset, pod_group_docs, render_yaml
from test.kueue.run_kueue_benchmarks import ARMS, BenchmarkArm, generate_arm_artifacts, meta_row_to_workload, submit_workloads_live


MODEL_SERVER_URL = "http://127.0.0.1:5050"
MODEL_SERVER_CLUSTER_URL = "http://host.docker.internal:5050"
KUEUE_IMAGE_REPO = "kueue-admirl"
KUEUE_NAMESPACE = "kueue-system"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "test" / "results" / "kueue-live-matrix"
KWOK_CLUSTER_ALIAS = "admirl-kwok"
DEFAULT_KUEUE_FORK_URL = "https://github.com/ABHIGYAN-MOHANTA/kueue.git"
DEFAULT_KUEUE_FORK_REF = "main"
DEFAULT_KUEUE_CACHE_DIR = Path(tempfile.gettempdir()) / "admirl-kueue-source"
LOCAL_KUEUE_SOURCE_DIR = Path(
    os.environ.get("ADMIRL_LOCAL_KUEUE_SOURCE_DIR", str(Path.home() / "Desktop" / "kueue"))
).expanduser().resolve()

PRESET_TO_KWOK_LAYOUT = {
    "kueue-lingjun-gang-starvation": "training-gang-starvation",
    "kueue-lingjun-gang-starvation-cohort": "training-gang-starvation",
    "kueue-lingjun-gang-topology-provisioning": "training-gang-topology-provisioning",
    "kueue-lingjun-gang-elastic-topology": "training-gang-elastic-topology",
    "kueue-lingjun-gang-elastic-profile-cohort": "training-gang-elastic-profile-cohort",
}

_KUEUE_IMAGE_CACHE: str | None = None
_KUEUE_BUILD_DATE_CACHE: str | None = None
_KUEUE_SOURCE_DIR_CACHE: Path | None = None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _kueue_cache_dir() -> Path:
    override = os.environ.get("ADMIRL_KUEUE_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_KUEUE_CACHE_DIR


def _kueue_fork_url() -> str:
    return os.environ.get("ADMIRL_KUEUE_GIT_URL", DEFAULT_KUEUE_FORK_URL).strip() or DEFAULT_KUEUE_FORK_URL


def _kueue_fork_ref() -> str:
    return os.environ.get("ADMIRL_KUEUE_GIT_REF", DEFAULT_KUEUE_FORK_REF).strip() or DEFAULT_KUEUE_FORK_REF


def _prepare_kueue_fork_source() -> Path:
    cache_dir = _kueue_cache_dir()
    repo_url = _kueue_fork_url()
    repo_ref = _kueue_fork_ref()
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    if not (cache_dir / ".git").exists():
        _run(
            ["git", "clone", "--branch", repo_ref, "--single-branch", repo_url, str(cache_dir)],
            cwd=cache_dir.parent,
            capture_output=False,
        )
        return cache_dir

    origin_url = _run(["git", "remote", "get-url", "origin"], cwd=cache_dir, check=False).stdout.strip()
    if origin_url and origin_url != repo_url:
        raise RuntimeError(
            f"cached Kueue source at {cache_dir} points to {origin_url}, expected {repo_url}; "
            "set ADMIRL_KUEUE_SOURCE_DIR or ADMIRL_KUEUE_CACHE_DIR to use a different checkout"
        )

    checkout = _run(["git", "checkout", "--force", repo_ref], cwd=cache_dir, check=False)
    if checkout.returncode == 0 and not _truthy_env("ADMIRL_KUEUE_REFRESH"):
        return cache_dir

    fetch = _run(["git", "fetch", "origin", repo_ref], cwd=cache_dir, check=False, capture_output=False)
    if fetch.returncode != 0:
        raise RuntimeError(f"failed to fetch Kueue ref {repo_ref!r} from {repo_url}")
    _run(["git", "checkout", "--force", "FETCH_HEAD"], cwd=cache_dir, capture_output=False)
    return cache_dir


def _kueue_source_metadata(source_dir: Path) -> dict[str, str]:
    metadata = {"path": str(source_dir)}
    remote = _run(["git", "remote", "get-url", "origin"], cwd=source_dir, check=False).stdout.strip()
    commit = _run(["git", "rev-parse", "HEAD"], cwd=source_dir, check=False).stdout.strip()
    ref = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=source_dir, check=False).stdout.strip()
    if remote:
        metadata["git_url"] = remote
    if commit:
        metadata["git_commit"] = commit
    if ref:
        metadata["git_ref"] = ref
    return metadata


def kueue_source_dir() -> Path:
    global _KUEUE_SOURCE_DIR_CACHE
    if _KUEUE_SOURCE_DIR_CACHE is not None:
        return _KUEUE_SOURCE_DIR_CACHE

    override = os.environ.get("ADMIRL_KUEUE_SOURCE_DIR", "").strip()
    if override:
        _KUEUE_SOURCE_DIR_CACHE = Path(override).expanduser().resolve()
        return _KUEUE_SOURCE_DIR_CACHE

    mode = os.environ.get("ADMIRL_KUEUE_SOURCE_MODE", "auto").strip().lower()
    if mode == "local":
        if not LOCAL_KUEUE_SOURCE_DIR.exists():
            raise SystemExit(
                f"local Kueue source dir {LOCAL_KUEUE_SOURCE_DIR} does not exist; "
                "set ADMIRL_KUEUE_SOURCE_DIR or remove ADMIRL_KUEUE_SOURCE_MODE=local"
            )
        _KUEUE_SOURCE_DIR_CACHE = LOCAL_KUEUE_SOURCE_DIR.resolve()
        return _KUEUE_SOURCE_DIR_CACHE

    if mode in {"auto", "fork", "git", ""}:
        try:
            _KUEUE_SOURCE_DIR_CACHE = _prepare_kueue_fork_source()
            return _KUEUE_SOURCE_DIR_CACHE
        except Exception as exc:
            if mode == "auto" and LOCAL_KUEUE_SOURCE_DIR.exists():
                _KUEUE_SOURCE_DIR_CACHE = LOCAL_KUEUE_SOURCE_DIR.resolve()
                return _KUEUE_SOURCE_DIR_CACHE
            raise SystemExit(str(exc))

    raise SystemExit(f"unsupported ADMIRL_KUEUE_SOURCE_MODE={mode!r}")


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        input=input_text,
        text=True,
        check=check,
        capture_output=capture_output,
    )


def _http_json(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=data, headers=headers, method=method)
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body or "{}")


def _kubectl_json(args: list[str]) -> dict:
    result = _run(["kubectl", *args, "-o", "json"])
    return json.loads(result.stdout or "{}")


def _kubectl_json_optional(args: list[str]) -> dict:
    result = _run(["kubectl", *args, "-o", "json"], check=False)
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def _workload_name_from_object(item: dict) -> str:
    metadata = item.get("metadata", {}) or {}
    annotations = metadata.get("annotations", {}) or {}
    labels = metadata.get("labels", {}) or {}
    return str(
        annotations.get("admirl.ai/workload-name")
        or labels.get("admirl.ai/workload-name")
        or metadata.get("name")
        or ""
    )


def _job_complete(job: dict) -> bool:
    status = job.get("status", {}) or {}
    conditions = status.get("conditions", []) or []
    for cond in conditions:
        if cond.get("type") in {"Complete", "Failed"} and str(cond.get("status")) == "True":
            return True
    succeeded = int(status.get("succeeded", 0) or 0)
    completions = int(((job.get("spec", {}) or {}).get("completions", 0) or 0))
    return completions > 0 and succeeded >= completions


def _live_node_states(namespace: str, meta_rows: dict[str, dict]) -> list[NodeState]:
    nodes_json = _kubectl_json(["get", "nodes"]).get("items", [])
    nodes_by_name: dict[str, NodeState] = {}
    for item in nodes_json:
        metadata = item.get("metadata", {}) or {}
        labels = metadata.get("labels", {}) or {}
        status = item.get("status", {}) or {}
        allocatable = status.get("allocatable", {}) or {}
        name = str(metadata.get("name") or "")
        node = NodeState(
            name=name,
            domain=str(labels.get("nvlink-domain", "")),
            cpu_total=int(float(str(allocatable.get("cpu", "0")).rstrip("m") or "0") * (1000 if "m" not in str(allocatable.get("cpu", "0")) else 1)),
            mem_total=_parse_k8s_memory_bytes(str(allocatable.get("memory", "0"))),
            gpu_total=int(str(allocatable.get("admirl.ai/gpu", "0") or "0")),
        )
        nodes_by_name[name] = node

    pods = _kubectl_json_optional(["get", "pods", "-n", namespace]).get("items", [])
    for pod in pods:
        spec = pod.get("spec", {}) or {}
        status = pod.get("status", {}) or {}
        node_name = str(spec.get("nodeName", "") or "")
        if not node_name or node_name not in nodes_by_name:
            continue
        phase = str(status.get("phase", "") or "")
        if phase in {"Succeeded", "Failed"}:
            continue
        workload_name = _workload_name_from_object(pod)
        meta_row = meta_rows.get(workload_name)
        if meta_row is None:
            continue
        node = nodes_by_name[node_name]
        node.allocate(
            type(
                "_Tmp",
                (),
                {
                    "cpu_milli": int(meta_row.get("per_worker_cpu_milli", 1000) or 1000),
                    "mem_bytes": int(meta_row.get("per_worker_mem_bytes", 2 * (1024**3)) or (2 * (1024**3))),
                    "gpu": int(meta_row.get("per_worker_gpu", 1) or 1),
                },
            )()
        )
    return list(nodes_by_name.values())


def _parse_k8s_memory_bytes(raw: str) -> int:
    token = str(raw or "0").strip()
    if not token:
        return 0
    suffixes = {
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
    }
    for suffix, factor in suffixes.items():
        if token.endswith(suffix):
            return int(float(token[:-len(suffix)] or 0.0) * factor)
    return int(float(token))


def _live_waiting_and_running(
    namespace: str,
    meta_rows: dict[str, dict],
    *,
    submitted_workload_ids: set[str] | None = None,
) -> tuple[list, int]:
    jobs = _kubectl_json_optional(["get", "jobs.batch", "-n", namespace]).get("items", [])
    waiting_by_workload: dict[str, dict] = {}
    running_workloads: set[str] = set()
    running_count = 0

    for workload_id in sorted(submitted_workload_ids or set()):
        meta_row = meta_rows.get(workload_id)
        if meta_row is not None:
            waiting_by_workload[workload_id] = meta_row

    for job in jobs:
        workload_name = _workload_name_from_object(job)
        meta_row = meta_rows.get(workload_name)
        if meta_row is None or _job_complete(job):
            continue
        active = int((job.get("status", {}) or {}).get("active", 0) or 0)
        if active > 0:
            running_workloads.add(workload_name)
            continue
        waiting_by_workload[workload_name] = meta_row

    pods = _kubectl_json_optional(["get", "pods", "-n", namespace]).get("items", [])
    for pod in pods:
        workload_name = _workload_name_from_object(pod)
        meta_row = meta_rows.get(workload_name)
        if meta_row is None:
            continue
        status = pod.get("status", {}) or {}
        spec = pod.get("spec", {}) or {}
        phase = str(status.get("phase", "") or "")
        if phase in {"Succeeded", "Failed"}:
            continue
        if str(spec.get("nodeName", "") or ""):
            running_workloads.add(workload_name)
            waiting_by_workload.pop(workload_name, None)
        else:
            if workload_name not in running_workloads:
                waiting_by_workload[workload_name] = meta_row

    waiting = [meta_row_to_workload(row) for row in waiting_by_workload.values()]
    running_count = len(running_workloads)
    return waiting, running_count


def _node_payload(node: NodeState) -> dict[str, object]:
    return {
        "name": node.name,
        "domain": node.domain,
        "cpu_total": node.cpu_total,
        "mem_total": node.mem_total,
        "gpu_total": node.gpu_total,
        "free_cpu": node.free_cpu,
        "free_mem": node.free_mem,
        "free_gpu": node.free_gpu,
    }


def _select_elastic_submission_workers(
    *,
    namespace: str,
    meta_row: dict,
    meta_rows: dict[str, dict],
    submitted_workload_ids: set[str],
    time_now: float,
    policy_mode: str,
    loaded_policy,
) -> int:
    workload = meta_row_to_workload(meta_row)
    if not workload.elastic_enabled or policy_mode == "disabled":
        return int(meta_row.get("initial_worker_count", workload.worker_count) or workload.worker_count)

    waiting, running_count = _live_waiting_and_running(
        namespace,
        meta_rows,
        submitted_workload_ids=submitted_workload_ids,
    )
    waiting.append(workload)
    nodes = _live_node_states(namespace, meta_rows)
    candidates = build_candidate_actions(waiting, nodes, time_now, fairshare=type("_FS", (), {"launched": {}, "completed": {}})())
    request_state = {
        "request_mode": "kueue-admission",
        "time": time_now,
        "candidates": [candidate.__dict__ for candidate in candidates],
        "nodes": [_node_payload(node) for node in nodes],
        "fair_share_violation_count": 0,
        "blocked_seconds": 0.0,
        "idle_quota_while_blocked": 0.0,
    }
    response = build_kueue_admission_response(
        request_state=request_state,
        policy=loaded_policy,
        policy_mode=policy_mode,
    )
    pair_scores = response.get("pair_scores", {}) or {}
    current_candidates = [
        candidate
        for candidate in candidates
        if candidate.workload_id == workload.workload_id and (candidate.immediate_fit or candidate.provisionable)
    ]
    if not current_candidates:
        return workload.min_worker_count
    best = max(
        current_candidates,
        key=lambda candidate: (
            float(pair_scores.get(candidate.action_id, float("-inf"))),
            candidate.worker_count,
            candidate.action_id,
        ),
    )
    return int(best.worker_count)


def _kubectl_apply_docs(docs: list[dict]) -> None:
    _run(["kubectl", "apply", "-f", "-"], input_text=render_yaml(docs), capture_output=False)


def submit_workloads_live_with_policy(
    *,
    arm: BenchmarkArm,
    arm_dir: Path,
    meta: dict,
    time_scale: float,
    runtime_scale: float,
    namespace: str,
    elastic_policy_mode: str,
    loaded_policy,
) -> None:
    meta_rows = {row["workload_id"]: row for row in meta.get("workloads", [])}
    benchmark_start = time.time()
    last_arrival = 0.0
    preset = canonical_kueue_preset(str(meta.get("workload_preset", "")))
    use_initial_scale_override = any(bool(row.get("elastic_enabled", False)) for row in meta.get("workloads", []))
    submitted_workload_ids: set[str] = set()

    for item in sorted(meta.get("workloads", []), key=lambda row: (row["arrival_time"], -row["total_gpu"], row["workload_id"])):
        sleep_seconds = max(0.0, (float(item["arrival_time"]) - last_arrival) / max(time_scale, 1e-6))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if use_initial_scale_override:
            workload = meta_row_to_workload(item)
            selected_workers = _select_elastic_submission_workers(
                namespace=namespace,
                meta_row=item,
                meta_rows=meta_rows,
                submitted_workload_ids=submitted_workload_ids,
                time_now=max(0.0, time.time() - benchmark_start),
                policy_mode=elastic_policy_mode,
                loaded_policy=loaded_policy,
            )
            item["selected_worker_count"] = selected_workers
            item["selected_total_gpu"] = selected_workers * int(item.get("per_worker_gpu", 1) or 1)
            docs = pod_group_docs(
                workload_for_scale(workload, selected_workers),
                namespace=namespace,
                runtime_scale=runtime_scale,
            )
            _kubectl_apply_docs(docs)
        else:
            _run(["kubectl", "apply", "-f", str(arm_dir / "workloads" / item["file"])], capture_output=False)
        submitted_workload_ids.add(str(item["workload_id"]))
        last_arrival = float(item["arrival_time"])


def _iso_to_ts(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil((pct / 100.0) * len(ordered)) - 1)))
    return float(ordered[index])


def _format_build_date(timestamp: float) -> str:
    return datetime.fromtimestamp(max(timestamp, 0.0), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_tree_identity(root: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    latest_mtime = 0.0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if rel_path.startswith(".git/") or rel_path.endswith(".pyc"):
            continue
        stat = path.stat()
        latest_mtime = max(latest_mtime, stat.st_mtime)
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest(), _format_build_date(latest_mtime)


def _kueue_image_name(source_hash: str) -> str:
    return f"{KUEUE_IMAGE_REPO}:{source_hash[:12]}"


def kueue_image() -> str:
    global _KUEUE_IMAGE_CACHE, _KUEUE_BUILD_DATE_CACHE
    if _KUEUE_IMAGE_CACHE is None or _KUEUE_BUILD_DATE_CACHE is None:
        source_hash, build_date = _source_tree_identity(kueue_source_dir())
        _KUEUE_IMAGE_CACHE = _kueue_image_name(source_hash)
        _KUEUE_BUILD_DATE_CACHE = build_date
    return _KUEUE_IMAGE_CACHE


def _kueue_build_date() -> str:
    kueue_image()
    assert _KUEUE_BUILD_DATE_CACHE is not None
    return _KUEUE_BUILD_DATE_CACHE


def _docker_image_exists(image: str) -> bool:
    result = _run(["docker", "image", "inspect", image], check=False)
    return result.returncode == 0


PAPER_METRIC_DIRECTIONS = {
    "head_gang_blocked_seconds": "lower",
    "avg_gang_wait_seconds": "lower",
    "p95_gang_wait_seconds": "lower",
    "p99_gang_wait_seconds": "lower",
    "avg_gang_completion_seconds": "lower",
    "avg_topology_aware_wait_seconds": "lower",
    "avg_topology_aware_completion_seconds": "lower",
    "avg_critical_wait_seconds": "lower",
    "avg_critical_completion_seconds": "lower",
    "avg_elastic_wait_seconds": "lower",
    "avg_elastic_completion_seconds": "lower",
    "elastic_completion_ratio": "higher",
    "critical_completion_ratio": "higher",
    "small_job_bypass_count_while_gang_pending": "lower",
    "small_job_bypass_fraction_while_gang_pending": "lower",
    "small_job_head_flavor_admissions_while_gang_pending": "lower",
    "small_job_head_flavor_gpu_while_gang_pending": "lower",
    "gang_slowdown_vs_isolated": "lower",
    "avg_small_wait_seconds": "lower",
    "avg_small_completion_seconds": "lower",
    "avg_small_slowdown_vs_isolated": "lower",
    "makespan_seconds": "lower",
    "throughput_jobs_per_minute": "higher",
    "throughput_gpu_per_minute": "higher",
    "job_completion_ratio": "higher",
    "gang_completion_ratio": "higher",
    "topology_hit_rate": "higher",
}

IMPORTANT_GANG_METRICS = [
    "head_gang_blocked_seconds",
    "head_gang_wait_to_runtime_ratio",
    "avg_gang_wait_seconds",
    "p95_gang_wait_seconds",
    "p99_gang_wait_seconds",
    "avg_gang_completion_seconds",
    "avg_topology_aware_wait_seconds",
    "avg_topology_aware_completion_seconds",
    "avg_critical_wait_seconds",
    "avg_critical_completion_seconds",
    "avg_elastic_wait_seconds",
    "avg_elastic_completion_seconds",
    "elastic_completion_ratio",
    "critical_completion_ratio",
    "small_job_bypass_count_while_gang_pending",
    "small_job_bypass_fraction_while_gang_pending",
    "small_job_bypass_gpu_while_gang_pending",
    "small_job_head_flavor_admissions_while_gang_pending",
    "small_job_head_flavor_gpu_while_gang_pending",
    "gang_slowdown_vs_isolated",
    "gang_completion_ratio",
    "avg_small_wait_seconds",
    "avg_small_completion_seconds",
    "avg_small_slowdown_vs_isolated",
    "makespan_seconds",
    "throughput_jobs_per_minute",
    "throughput_gpu_per_minute",
    "topology_hit_rate",
]

LOWER_IS_BETTER = {metric for metric, direction in PAPER_METRIC_DIRECTIONS.items() if direction == "lower"}


def _average(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _series_stats(values: list[float]) -> dict[str, float]:
    return {
        "avg": _average(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else 0.0,
    }


def _apply_series_stats(metrics: dict, prefix: str, values: list[float]) -> None:
    stats = _series_stats(values)
    for label, value in stats.items():
        metrics[f"{label}_{prefix}"] = float(value)


def _completion_end_time(row: dict) -> float | None:
    arrival_time = row.get("scaled_arrival_time")
    completion_seconds = row.get("completion_seconds")
    if arrival_time is None or completion_seconds is None:
        return None
    return float(arrival_time) + float(completion_seconds)


def _admission_time(row: dict) -> float | None:
    arrival_time = row.get("scaled_arrival_time")
    wait_seconds = row.get("wait_seconds")
    if arrival_time is None or wait_seconds is None:
        return None
    return float(arrival_time) + float(wait_seconds)


def _slowdowns(rows: list[dict]) -> list[float]:
    values: list[float] = []
    for row in rows:
        runtime_seconds = row.get("runtime_seconds")
        completion_seconds = row.get("completion_seconds")
        if runtime_seconds and completion_seconds:
            values.append(float(completion_seconds) / max(float(runtime_seconds), 1.0))
    return values


def _gpu_weighted_slowdown(rows: list[dict]) -> float:
    weighted_runtime = 0.0
    weighted_completion = 0.0
    for row in rows:
        runtime_seconds = row.get("runtime_seconds")
        completion_seconds = row.get("completion_seconds")
        total_gpu = row.get("total_gpu")
        if runtime_seconds is None or completion_seconds is None or total_gpu is None:
            continue
        weight = float(total_gpu)
        weighted_runtime += weight * float(runtime_seconds)
        weighted_completion += weight * float(completion_seconds)
    if weighted_runtime <= 0:
        return 0.0
    return weighted_completion / weighted_runtime


def _candidate_flavor_set(row: dict) -> set[str]:
    flavors = row.get("candidate_flavors") or []
    return {str(item) for item in flavors if str(item).strip()}


def _admitted_flavor_set(row: dict) -> set[str]:
    flavors = row.get("admitted_flavors") or []
    return {str(item) for item in flavors if str(item).strip()}


def _is_critical_workload_name(name: str) -> bool:
    token = str(name or "").strip().lower()
    return "critical" in token


def _resource_overlaps_head(row: dict, head_gang: dict) -> bool:
    head_flavors = _candidate_flavor_set(head_gang)
    row_flavors = _candidate_flavor_set(row)
    if head_flavors and row_flavors:
        return not row_flavors.isdisjoint(head_flavors)
    return True


def _augment_arm_metrics(metrics: dict) -> dict:
    raw_groups = list(metrics.get("raw_groups", []) or [])
    enriched = copy.deepcopy(metrics)
    if not raw_groups:
        return enriched

    all_rows = raw_groups
    gang_rows = [row for row in raw_groups if row.get("queue_class") == "gang"]
    small_rows = [row for row in raw_groups if row.get("queue_class") != "gang"]
    topology_rows = [row for row in raw_groups if str(row.get("topology_preference") or "").strip()]
    critical_rows = [row for row in raw_groups if _is_critical_workload_name(str(row.get("workload_name") or ""))]
    elastic_rows = [
        row
        for row in raw_groups
        if bool(row.get("elastic_enabled"))
        or ("elastic" in str(row.get("workload_name") or "").strip().lower())
    ]
    provisioned_rows = [row for row in raw_groups if row.get("used_spare_capacity")]

    all_waits = [float(row["wait_seconds"]) for row in all_rows if row.get("wait_seconds") is not None]
    gang_waits = [float(row["wait_seconds"]) for row in gang_rows if row.get("wait_seconds") is not None]
    small_waits = [float(row["wait_seconds"]) for row in small_rows if row.get("wait_seconds") is not None]
    topology_waits = [float(row["wait_seconds"]) for row in topology_rows if row.get("wait_seconds") is not None]
    critical_waits = [float(row["wait_seconds"]) for row in critical_rows if row.get("wait_seconds") is not None]
    elastic_waits = [float(row["wait_seconds"]) for row in elastic_rows if row.get("wait_seconds") is not None]
    provisioned_waits = [float(row["wait_seconds"]) for row in provisioned_rows if row.get("wait_seconds") is not None]
    all_completions = [float(row["completion_seconds"]) for row in all_rows if row.get("completion_seconds") is not None]
    gang_completions = [float(row["completion_seconds"]) for row in gang_rows if row.get("completion_seconds") is not None]
    small_completions = [float(row["completion_seconds"]) for row in small_rows if row.get("completion_seconds") is not None]
    topology_completions = [float(row["completion_seconds"]) for row in topology_rows if row.get("completion_seconds") is not None]
    critical_completions = [float(row["completion_seconds"]) for row in critical_rows if row.get("completion_seconds") is not None]
    elastic_completions = [float(row["completion_seconds"]) for row in elastic_rows if row.get("completion_seconds") is not None]
    provisioned_completions = [float(row["completion_seconds"]) for row in provisioned_rows if row.get("completion_seconds") is not None]
    all_slowdowns = _slowdowns(all_rows)
    gang_slowdowns = _slowdowns(gang_rows)
    small_slowdowns = _slowdowns(small_rows)

    _apply_series_stats(enriched, "workload_wait_seconds", all_waits)
    _apply_series_stats(enriched, "job_completion_seconds", all_completions)
    _apply_series_stats(enriched, "gang_wait_seconds", gang_waits)
    _apply_series_stats(enriched, "gang_completion_seconds", gang_completions)
    _apply_series_stats(enriched, "small_wait_seconds", small_waits)
    _apply_series_stats(enriched, "small_completion_seconds", small_completions)
    _apply_series_stats(enriched, "topology_aware_wait_seconds", topology_waits)
    _apply_series_stats(enriched, "topology_aware_completion_seconds", topology_completions)
    _apply_series_stats(enriched, "critical_wait_seconds", critical_waits)
    _apply_series_stats(enriched, "critical_completion_seconds", critical_completions)
    _apply_series_stats(enriched, "elastic_wait_seconds", elastic_waits)
    _apply_series_stats(enriched, "elastic_completion_seconds", elastic_completions)
    _apply_series_stats(enriched, "provisioned_wait_seconds", provisioned_waits)
    _apply_series_stats(enriched, "provisioned_completion_seconds", provisioned_completions)
    _apply_series_stats(enriched, "workload_slowdown_vs_isolated", all_slowdowns)
    _apply_series_stats(enriched, "gang_slowdown_vs_isolated", gang_slowdowns)
    _apply_series_stats(enriched, "small_slowdown_vs_isolated", small_slowdowns)

    enriched["gang_slowdown_vs_isolated"] = float(enriched.get("avg_gang_slowdown_vs_isolated", 0.0))

    jobs_total = int(enriched.get("jobs_total", len(all_rows)) or 0)
    jobs_completed = int(enriched.get("jobs_completed", sum(1 for row in all_rows if row.get("completed"))) or 0)
    jobs_running = sum(
        1
        for row in all_rows
        if not row.get("completed")
        and any(row.get(key) is not None for key in ("admitted_at", "scheduled_at", "started_at"))
    )
    jobs_pending = max(0, jobs_total - jobs_completed - jobs_running)
    gang_completed = sum(1 for row in gang_rows if row.get("completed"))
    small_completed = sum(1 for row in small_rows if row.get("completed"))
    topology_completed = sum(1 for row in topology_rows if row.get("completed"))
    critical_completed = sum(1 for row in critical_rows if row.get("completed"))
    elastic_completed = sum(1 for row in elastic_rows if row.get("completed"))
    provisioned_completed = sum(1 for row in provisioned_rows if row.get("completed"))
    enriched["job_completion_ratio"] = (jobs_completed / jobs_total) if jobs_total else 0.0
    enriched["progress_ratio"] = enriched["job_completion_ratio"]
    enriched["jobs_running"] = jobs_running
    enriched["jobs_pending"] = jobs_pending
    enriched["gang_jobs_total"] = len(gang_rows)
    enriched["gang_jobs_completed"] = gang_completed
    enriched["small_jobs_total"] = len(small_rows)
    enriched["small_jobs_completed"] = small_completed
    enriched["topology_aware_jobs_total"] = len(topology_rows)
    enriched["topology_aware_jobs_completed"] = topology_completed
    enriched["critical_jobs_total"] = len(critical_rows)
    enriched["critical_jobs_completed"] = critical_completed
    enriched["elastic_jobs_total"] = len(elastic_rows)
    enriched["elastic_jobs_completed"] = elastic_completed
    enriched["provisioned_jobs_total"] = len(provisioned_rows)
    enriched["provisioned_jobs_completed"] = provisioned_completed
    enriched["gang_completion_ratio"] = (gang_completed / len(gang_rows)) if gang_rows else 0.0
    enriched["small_completion_ratio"] = (small_completed / len(small_rows)) if small_rows else 0.0
    enriched["topology_aware_completion_ratio"] = (topology_completed / len(topology_rows)) if topology_rows else 0.0
    enriched["critical_completion_ratio"] = (critical_completed / len(critical_rows)) if critical_rows else 0.0
    enriched["elastic_completion_ratio"] = (elastic_completed / len(elastic_rows)) if elastic_rows else 0.0
    enriched["provisioned_completion_ratio"] = (provisioned_completed / len(provisioned_rows)) if provisioned_rows else 0.0

    arrival_values = [float(row["scaled_arrival_time"]) for row in all_rows if row.get("scaled_arrival_time") is not None]
    completion_end_values = [end_time for row in all_rows if (end_time := _completion_end_time(row)) is not None]
    makespan_seconds = 0.0
    if arrival_values and completion_end_values:
        makespan_seconds = max(0.0, max(completion_end_values) - min(arrival_values))
    enriched["makespan_seconds"] = makespan_seconds
    enriched["throughput_jobs_per_minute"] = (jobs_completed * 60.0 / makespan_seconds) if makespan_seconds > 0 else 0.0
    completed_gpu = sum(float(row.get("total_gpu") or 0.0) for row in all_rows if row.get("completed"))
    enriched["throughput_gpu_per_minute"] = (completed_gpu * 60.0 / makespan_seconds) if makespan_seconds > 0 else 0.0
    enriched["requested_gpu_seconds_total"] = sum(
        float(row.get("total_gpu") or 0.0) * float(row.get("runtime_seconds") or 0.0)
        for row in all_rows
    )
    enriched["completed_gpu_seconds_total"] = sum(
        float(row.get("total_gpu") or 0.0) * float(row.get("runtime_seconds") or 0.0)
        for row in all_rows
        if row.get("completed")
    )
    enriched["gpu_weighted_workload_slowdown_vs_isolated"] = _gpu_weighted_slowdown(all_rows)
    enriched["gpu_weighted_gang_slowdown_vs_isolated"] = _gpu_weighted_slowdown(gang_rows)
    enriched["gpu_weighted_small_slowdown_vs_isolated"] = _gpu_weighted_slowdown(small_rows)

    avg_gang_wait = float(enriched.get("avg_gang_wait_seconds", 0.0))
    avg_small_wait = float(enriched.get("avg_small_wait_seconds", 0.0))
    enriched["avg_gang_minus_small_wait_seconds"] = avg_gang_wait - avg_small_wait
    enriched["avg_gang_to_small_wait_ratio"] = (avg_gang_wait / avg_small_wait) if avg_small_wait > 0 else 0.0

    head_gang = None
    for row in gang_rows:
        arrival_time = row.get("scaled_arrival_time")
        if arrival_time is None:
            continue
        if head_gang is None or float(arrival_time) < float(head_gang.get("scaled_arrival_time")):
            head_gang = row
    if head_gang is not None:
        enriched["head_gang_workload"] = head_gang.get("workload_name", enriched.get("head_gang_workload", ""))
        enriched["head_gang_blocked_seconds"] = float(head_gang.get("wait_seconds") or 0.0)
        head_runtime = head_gang.get("runtime_seconds")
        head_completion = head_gang.get("completion_seconds")
        enriched["head_gang_wait_to_runtime_ratio"] = (
            float(head_gang.get("wait_seconds") or 0.0) / max(float(head_runtime), 1.0)
            if head_runtime
            else 0.0
        )
        enriched["head_gang_completion_to_runtime_ratio"] = (
            float(head_completion) / max(float(head_runtime), 1.0)
            if head_runtime and head_completion is not None
            else 0.0
        )
        head_arrival = head_gang.get("scaled_arrival_time")
        head_admit = _admission_time(head_gang)
        younger_small_rows = [
            row
            for row in small_rows
            if head_arrival is not None
            and row.get("scaled_arrival_time") is not None
            and float(row["scaled_arrival_time"]) > float(head_arrival)
        ]
        competing_younger_small_rows = [row for row in younger_small_rows if _resource_overlaps_head(row, head_gang)]
        bypass_rows = [
            row
            for row in competing_younger_small_rows
            if head_admit is not None and (admit_time := _admission_time(row)) is not None and float(admit_time) < float(head_admit)
        ]
        enriched["small_jobs_arriving_while_head_gang_pending"] = len(younger_small_rows)
        enriched["small_job_bypass_count_while_gang_pending"] = len(bypass_rows)
        enriched["small_job_bypass_fraction_while_gang_pending"] = (
            len(bypass_rows) / len(competing_younger_small_rows)
            if competing_younger_small_rows
            else 0.0
        )
        enriched["small_job_bypass_gpu_while_gang_pending"] = sum(
            float(row.get("total_gpu") or 0.0) for row in bypass_rows
        )
        head_flavor_bypass_rows = [
            row
            for row in bypass_rows
            if not _admitted_flavor_set(row).isdisjoint(_candidate_flavor_set(head_gang))
        ]
        enriched["small_job_head_flavor_admissions_while_gang_pending"] = len(head_flavor_bypass_rows)
        enriched["small_job_head_flavor_gpu_while_gang_pending"] = sum(
            float(row.get("total_gpu") or 0.0) for row in head_flavor_bypass_rows
        )
    else:
        enriched["head_gang_wait_to_runtime_ratio"] = 0.0
        enriched["head_gang_completion_to_runtime_ratio"] = 0.0
        enriched["small_jobs_arriving_while_head_gang_pending"] = 0
        enriched["small_job_bypass_count_while_gang_pending"] = 0
        enriched["small_job_bypass_fraction_while_gang_pending"] = 0.0
        enriched["small_job_bypass_gpu_while_gang_pending"] = 0.0
        enriched["small_job_head_flavor_admissions_while_gang_pending"] = 0
        enriched["small_job_head_flavor_gpu_while_gang_pending"] = 0.0

    return enriched


def _compare_metric_values(baseline_value: float, candidate_value: float, *, direction: str) -> dict:
    baseline_value = float(baseline_value)
    candidate_value = float(candidate_value)
    delta = candidate_value - baseline_value
    if direction == "lower":
        improvement_fraction = ((baseline_value - candidate_value) / baseline_value) if baseline_value > 0 else None
        winner = "candidate" if candidate_value < baseline_value else "baseline" if candidate_value > baseline_value else "tie"
    else:
        improvement_fraction = ((candidate_value - baseline_value) / baseline_value) if baseline_value > 0 else None
        winner = "candidate" if candidate_value > baseline_value else "baseline" if candidate_value < baseline_value else "tie"
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "direction": direction,
        "improvement_fraction": improvement_fraction,
        "winner": winner,
    }


def _build_arm_comparison(baseline_name: str, candidate_name: str, baseline: dict, candidate: dict) -> dict:
    metric_comparisons = {}
    for metric_name, direction in PAPER_METRIC_DIRECTIONS.items():
        if metric_name not in baseline or metric_name not in candidate:
            continue
        metric_comparisons[metric_name] = _compare_metric_values(
            baseline.get(metric_name, 0.0),
            candidate.get(metric_name, 0.0),
            direction=direction,
        )
    return {
        "baseline_arm": baseline_name,
        "candidate_arm": candidate_name,
        "metrics": metric_comparisons,
    }


def _augment_result_pack(result_pack: dict) -> dict:
    enriched = copy.deepcopy(result_pack)
    arms = {name: _augment_arm_metrics(metrics) for name, metrics in (enriched.get("arms", {}) or {}).items()}
    enriched["arms"] = arms

    comparisons = {}
    arm_names = list(arms)
    if "stock-best-effort-default" in arms:
        baseline_name = "stock-best-effort-default"
        for candidate_name in arm_names:
            if candidate_name == baseline_name:
                continue
            key = f"{baseline_name}__vs__{candidate_name}"
            comparisons[key] = _build_arm_comparison(baseline_name, candidate_name, arms[baseline_name], arms[candidate_name])
    elif len(arm_names) == 2:
        baseline_name, candidate_name = arm_names
        key = f"{baseline_name}__vs__{candidate_name}"
        comparisons[key] = _build_arm_comparison(baseline_name, candidate_name, arms[baseline_name], arms[candidate_name])

    enriched["comparisons"] = comparisons
    paper_summary = {}
    for key, comparison in comparisons.items():
        metrics = comparison.get("metrics", {})
        paper_summary[key] = {
            "headline_metric": metrics.get("head_gang_blocked_seconds", {}),
            "gang_wait": metrics.get("avg_gang_wait_seconds", {}),
            "small_wait": metrics.get("avg_small_wait_seconds", {}),
            "bypass_count": metrics.get("small_job_bypass_count_while_gang_pending", {}),
            "makespan": metrics.get("makespan_seconds", {}),
            "throughput_jobs_per_minute": metrics.get("throughput_jobs_per_minute", {}),
        }
    enriched["paper_summary"] = paper_summary
    return enriched


def _load_existing_result_pack(output_root: Path) -> dict:
    summary_path = output_root / "live-matrix-summary.json"
    result_pack = {}
    if summary_path.exists():
        result_pack = json.loads(summary_path.read_text(encoding="utf-8"))
    result_pack.setdefault("arms", {})
    for live_result_path in sorted(output_root.glob("*/live-results.json")):
        metrics = json.loads(live_result_path.read_text(encoding="utf-8"))
        meta_path = live_result_path.parent / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            workload_meta = {row.get("workload_id"): row for row in meta.get("workloads", []) or []}
            for row in metrics.get("raw_groups", []) or []:
                meta_row = workload_meta.get(row.get("workload_name"))
                if meta_row and not row.get("candidate_flavors"):
                    row["candidate_flavors"] = list(meta_row.get("candidate_flavors", []) or [])
        result_pack["arms"][live_result_path.parent.name] = metrics
    if not result_pack.get("arms"):
        raise SystemExit(f"no live-results.json files found under {output_root}")
    return result_pack


def _write_result_pack(output_root: Path, result_pack: dict) -> Path:
    enriched = _augment_result_pack(result_pack)
    output_root.mkdir(parents=True, exist_ok=True)
    for arm_name, metrics in (enriched.get("arms", {}) or {}).items():
        arm_dir = output_root / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "live-results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_path = output_root / "live-matrix-summary.json"
    summary_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    paper_summary_path = output_root / "live-matrix-paper-summary.json"
    paper_summary_path.write_text(
        json.dumps(
            {
                "workload_preset": enriched.get("workload_preset", ""),
                "seed": enriched.get("seed"),
                "num_jobs": enriched.get("num_jobs"),
                "comparisons": enriched.get("comparisons", {}),
                "paper_summary": enriched.get("paper_summary", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _count_small_job_bypass_while_gang_pending(
    group_rows: list[dict],
    *,
    head_gang_arrival: float | None,
    head_gang_admit: float | None,
    head_gang_candidate_flavors: list[str] | None = None,
) -> int:
    if head_gang_arrival is None or head_gang_admit is None:
        return 0
    head_flavors = {str(item) for item in (head_gang_candidate_flavors or []) if str(item).strip()}
    bypass_count = 0
    for row in group_rows:
        if row.get("queue_class") == "gang":
            continue
        row_flavors = _candidate_flavor_set(row)
        if head_flavors and row_flavors and row_flavors.isdisjoint(head_flavors):
            continue
        row_arrival = row.get("scaled_arrival_time", row.get("arrival_time"))
        row_wait = row.get("wait_seconds")
        if row_arrival is None or row_wait is None:
            continue
        row_arrival = float(row_arrival)
        if row_arrival <= float(head_gang_arrival):
            continue
        row_admit = row_arrival + float(row_wait)
        if row_admit < float(head_gang_admit):
            bypass_count += 1
    return bypass_count


def _small_job_head_flavor_admissions_while_gang_pending(
    group_rows: list[dict],
    *,
    head_gang_arrival: float | None,
    head_gang_admit: float | None,
    head_gang_candidate_flavors: list[str] | None = None,
) -> tuple[int, float]:
    if head_gang_arrival is None or head_gang_admit is None:
        return 0, 0.0
    head_flavors = {str(item) for item in (head_gang_candidate_flavors or []) if str(item).strip()}
    if not head_flavors:
        return 0, 0.0
    count = 0
    total_gpu = 0.0
    for row in group_rows:
        if row.get("queue_class") == "gang":
            continue
        admitted_flavors = _admitted_flavor_set(row)
        if not admitted_flavors or admitted_flavors.isdisjoint(head_flavors):
            continue
        row_arrival = row.get("scaled_arrival_time", row.get("arrival_time"))
        row_wait = row.get("wait_seconds")
        if row_arrival is None or row_wait is None:
            continue
        row_arrival = float(row_arrival)
        if row_arrival <= float(head_gang_arrival):
            continue
        row_admit = row_arrival + float(row_wait)
        if row_admit < float(head_gang_admit):
            count += 1
            total_gpu += float(row.get("total_gpu") or 0.0)
    return count, total_gpu


def _sanitize_namespace(name: str) -> str:
    lowered = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in name.lower())
    lowered = lowered.strip("-")
    if not lowered:
        lowered = "kueue-bench"
    return lowered[:63]


def resolve_kind_cluster_name(alias: str = KWOK_CLUSTER_ALIAS) -> str:
    result = _run(["kind", "get", "clusters"])
    clusters = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        current_context = _run(["kubectl", "config", "current-context"]).stdout.strip()
    except Exception:
        current_context = ""
    if current_context:
        context_cluster = current_context.removeprefix("kind-").removeprefix("kwok-")
        if current_context in clusters:
            return current_context
        if context_cluster in clusters:
            return context_cluster
        kwok_context_cluster = f"kwok-{context_cluster}"
        if kwok_context_cluster in clusters:
            return kwok_context_cluster
    if alias in clusters:
        return alias
    kwok_style = f"kwok-{alias}"
    if kwok_style in clusters:
        return kwok_style
    if len(clusters) == 1:
        return clusters[0]
    raise RuntimeError(
        f"unable to resolve kind cluster name for alias {alias!r}; found clusters: {clusters}"
    )


def resolve_control_plane_node_name(alias: str = KWOK_CLUSTER_ALIAS) -> str:
    try:
        result = _run(["kubectl", "get", "nodes", "-o", "name"])
        nodes = [line.strip().removeprefix("node/") for line in result.stdout.splitlines() if line.strip()]
        control_planes = [name for name in nodes if name.endswith("-control-plane")]
        if len(control_planes) == 1:
            return control_planes[0]
    except Exception:
        pass
    return f"{resolve_kind_cluster_name(alias)}-control-plane"


def _controller_overlay(learned_enabled: bool) -> str:
    enabled = "true" if learned_enabled else "false"
    return f"""apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ./kueue-default
configMapGenerator:
  - name: manager-config
    behavior: replace
    files:
      - controller_manager_config.yaml=./controller_manager_config.yaml
patches:
  - target:
      kind: Deployment
      name: kueue-controller-manager
      namespace: {KUEUE_NAMESPACE}
    patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: kueue-controller-manager
        namespace: {KUEUE_NAMESPACE}
      spec:
        template:
          spec:
            nodeSelector:
              node-role.kubernetes.io/control-plane: ""
            tolerations:
              - key: node-role.kubernetes.io/control-plane
                operator: Exists
                effect: NoSchedule
              - key: node-role.kubernetes.io/master
                operator: Exists
                effect: NoSchedule
            containers:
              - name: manager
                image: {kueue_image()}
                imagePullPolicy: IfNotPresent
                env:
                  - name: ADMIRL_KUEUE_ENABLE
                    value: "{enabled}"
                  - name: ADMIRL_KUEUE_MODEL_SERVER_URL
                    value: "{MODEL_SERVER_CLUSTER_URL}"
"""


def ensure_model_server() -> None:
    try:
        payload = _http_json(f"{MODEL_SERVER_URL}/api/policy/status")
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("model server is not reachable on http://127.0.0.1:5050") from exc
    if "effective_policy" not in payload:
        raise RuntimeError(f"unexpected model server status response: {json.dumps(payload)}")


def configure_model_server(*, runtime_policy: str, checkpoint_path: str | None) -> None:
    if checkpoint_path:
        _http_json(
            f"{MODEL_SERVER_URL}/api/policy/load-checkpoint",
            method="POST",
            payload={"path": str(Path(checkpoint_path).resolve())},
        )
    _http_json(
        f"{MODEL_SERVER_URL}/api/policy/runtime-policy",
        method="POST",
        payload={"policy": runtime_policy},
    )


def reset_model_server_runtime_metrics() -> None:
    try:
        _http_json(
            f"{MODEL_SERVER_URL}/api/runtime-metrics/reset",
            method="POST",
            payload={},
        )
    except urllib_error.HTTPError as exc:
        if exc.code != 404:
            raise


def model_server_status() -> dict:
    try:
        return _http_json(f"{MODEL_SERVER_URL}/api/policy/status")
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def publish_model_server_benchmark_snapshot(payload: dict) -> None:
    try:
        _http_json(
            f"{MODEL_SERVER_URL}/api/benchmark/progress",
            method="POST",
            payload=payload,
            timeout=5.0,
        )
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, json.JSONDecodeError):
        return


def _build_benchmark_progress_payload(
    metrics: dict,
    *,
    workload_preset: str,
    arm_name: str,
    seed: int,
    namespace: str,
    phase: str,
    elapsed_seconds: float,
    timeout_seconds: float,
    active: bool,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preset": canonical_kueue_preset(workload_preset),
        "arm": arm_name,
        "seed": int(seed),
        "namespace": namespace,
        "phase": phase,
        "active": bool(active),
        "elapsed_seconds": float(max(0.0, elapsed_seconds)),
        "timeout_seconds": float(max(0.0, timeout_seconds)),
        "progress_ratio": float(metrics.get("job_completion_ratio", 0.0) or 0.0),
        "jobs_total": int(metrics.get("jobs_total", 0) or 0),
        "jobs_completed": int(metrics.get("jobs_completed", 0) or 0),
        "jobs_running": int(metrics.get("jobs_running", 0) or 0),
        "jobs_pending": int(metrics.get("jobs_pending", 0) or 0),
        "avg_gang_wait_seconds": float(metrics.get("avg_gang_wait_seconds", 0.0) or 0.0),
        "avg_small_wait_seconds": float(metrics.get("avg_small_wait_seconds", 0.0) or 0.0),
        "avg_elastic_wait_seconds": float(metrics.get("avg_elastic_wait_seconds", 0.0) or 0.0),
        "head_gang_blocked_seconds": float(metrics.get("head_gang_blocked_seconds", 0.0) or 0.0),
        "head_gang_current_blocked_seconds": float(
            metrics.get(
                "head_gang_current_blocked_seconds",
                metrics.get("head_gang_blocked_seconds", 0.0),
            )
            or 0.0
        ),
        "throughput_jobs_per_minute": float(metrics.get("throughput_jobs_per_minute", 0.0) or 0.0),
        "throughput_gpu_per_minute": float(metrics.get("throughput_gpu_per_minute", 0.0) or 0.0),
        "job_completion_ratio": float(metrics.get("job_completion_ratio", 0.0) or 0.0),
        "gang_completion_ratio": float(metrics.get("gang_completion_ratio", 0.0) or 0.0),
        "elastic_completion_ratio": float(metrics.get("elastic_completion_ratio", 0.0) or 0.0),
        "topology_hit_rate": float(metrics.get("topology_hit_rate", 0.0) or 0.0),
        "small_job_bypass_count": float(metrics.get("small_job_bypass_count_while_gang_pending", 0.0) or 0.0),
        "head_gang_workload": str(metrics.get("head_gang_workload", "") or ""),
        "cluster_nodes_total": int(metrics.get("cluster_nodes_total", 0) or 0),
        "namespace_pods_total": int(metrics.get("namespace_pods_total", 0) or 0),
        "namespace_workloads_total": int(metrics.get("namespace_workloads_total", 0) or 0),
        "clusterqueues_total": int(metrics.get("clusterqueues_total", 0) or 0),
        "localqueues_total": int(metrics.get("localqueues_total", 0) or 0),
        "resourceflavors_total": int(metrics.get("resourceflavors_total", 0) or 0),
        "topologies_total": int(metrics.get("topologies_total", 0) or 0),
    }


def start_provisioners(
    arm_dir: Path,
    workload_preset: str,
    cluster_layout: str,
    *,
    workloads: list[dict] | None = None,
    time_scale: float = 1.0,
) -> list[tuple[subprocess.Popen[str], object]]:
    canonical = canonical_kueue_preset(workload_preset)
    if canonical not in {
        "kueue-lingjun-gang-topology-provisioning",
        "kueue-lingjun-gang-elastic-topology",
        "kueue-lingjun-gang-elastic-profile-cohort",
    }:
        return []

    first_scaled_arrival = 0.0
    if workloads:
        arrival_values = [float(item.get("arrival_time", 0.0) or 0.0) / max(time_scale, 1.0) for item in workloads]
        if arrival_values:
            # Keep the live sparse-capacity event aligned with the simulator,
            # which injects the spare C node 8s after the first workload arrives.
            first_scaled_arrival = min(arrival_values)

    launched: list[tuple[subprocess.Popen[str], object]] = []
    for index, (flavor_name, count, delay_seconds) in enumerate(
        [
            ("rf-4gpu-c", 1, 8.0),
        ]
    ):
        log_path = arm_dir / f"provisioner-{index}-{flavor_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [
                sys.executable,
                "test/kueue/provisioner.py",
                "--layout",
                cluster_layout,
                "--flavor",
                flavor_name,
                "--count",
                str(count),
                "--delay-seconds",
                str(first_scaled_arrival + delay_seconds),
            ],
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launched.append((proc, log_file))
    return launched


def stop_process(proc: subprocess.Popen[str] | None, log_file: object | None = None) -> None:
    if proc is None:
        if log_file is not None:
            log_file.close()
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if log_file is not None:
        log_file.close()


def build_kueue_image() -> None:
    image = kueue_image()
    if not _docker_image_exists(image) or os.environ.get("ADMIRL_FORCE_KUEUE_REBUILD") == "1":
        build = _run(
            [
                "docker",
                "build",
                "-t",
                image,
                "--build-arg",
                f"BUILD_DATE={_kueue_build_date()}",
                ".",
            ],
            cwd=kueue_source_dir(),
            capture_output=False,
            check=False,
        )
        if build.returncode != 0 and not _docker_image_exists(image):
            raise subprocess.CalledProcessError(build.returncode, build.args)
    cluster_name = resolve_kind_cluster_name()
    _run(["kind", "load", "docker-image", image, "--name", cluster_name], capture_output=False)


def apply_kueue_controller(*, learned_enabled: bool) -> None:
    control_plane_node = resolve_control_plane_node_name()
    _run(["kubectl", "uncordon", control_plane_node], capture_output=False)
    with tempfile.TemporaryDirectory(prefix="kueue-overlay-") as tmpdir:
        tmp_path = Path(tmpdir)
        overlay_target = tmp_path / "kueue-default"
        overlay_target.symlink_to(kueue_source_dir() / "config" / "default", target_is_directory=True)
        (tmp_path / "controller_manager_config.yaml").write_text(
            (REPO_ROOT / "test" / "kueue" / "controller_manager_config.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (tmp_path / "kustomization.yaml").write_text(_controller_overlay(learned_enabled), encoding="utf-8")
        _run(
            [
                "kubectl",
                "apply",
                "--server-side",
                "--force-conflicts",
                "-k",
                str(tmp_path),
            ],
            capture_output=False,
        )
    _run(
        [
            "kubectl",
            "rollout",
            "restart",
            "deployment/kueue-controller-manager",
            "-n",
            KUEUE_NAMESPACE,
        ],
        capture_output=False,
    )
    _run(
        [
            "kubectl",
            "wait",
            "deployment/kueue-controller-manager",
            "-n",
            KUEUE_NAMESPACE,
            "--for=condition=available",
            "--timeout=300s",
        ],
        capture_output=False,
    )
    # The deployment can be Ready slightly before the admission webhooks and
    # aggregated APIs are fully responsive for CRD-backed setup objects.
    time.sleep(20)


def apply_kwok_training_nodes(layout: str) -> None:
    _run(["kubectl", "delete", "node", "-l", "type=kwok", "--ignore-not-found=true"], capture_output=False)
    generated = _run(
        ["python3", "test/kwok/generate_fake_nodes.py", "--layout", layout],
        capture_output=True,
    )
    _run(["kubectl", "apply", "-f", "-"], input_text=generated.stdout, capture_output=False)
    time.sleep(2)


def cleanup_arm(arm_dir: Path) -> None:
    workload_dir = arm_dir / "workloads"
    if workload_dir.exists():
        _run(["kubectl", "delete", "-f", str(workload_dir), "--ignore-not-found=true", "--wait=false"], capture_output=False, check=False)
    setup_path = arm_dir / "setup.yaml"
    if setup_path.exists():
        _run(["kubectl", "delete", "-f", str(setup_path), "--ignore-not-found=true", "--wait=false"], capture_output=False, check=False)


def _delete_resource(kind: str, name: str, *, namespace: str | None = None) -> None:
    cmd = ["kubectl", "delete", kind, name, "--ignore-not-found=true", "--wait=false"]
    if namespace:
        cmd.extend(["-n", namespace])
    _run(cmd, capture_output=False, check=False)


def _clear_finalizers(kind: str, name: str, *, namespace: str | None = None) -> None:
    cmd = [
        "kubectl",
        "patch",
        kind,
        name,
        "--type=merge",
        "-p",
        '{"metadata":{"finalizers":[]}}',
    ]
    if namespace:
        cmd.extend(["-n", namespace])
    _run(cmd, capture_output=False, check=False)


def cleanup_kueue_test_state() -> None:
    _run(
        ["kubectl", "delete", "node", "-l", "admirl.ai/spare=true", "--ignore-not-found=true", "--wait=false"],
        capture_output=False,
        check=False,
    )

    namespaces = _kubectl_json_optional(["get", "namespaces"]).get("items", [])
    test_namespaces = []
    for item in namespaces:
        metadata = item.get("metadata", {}) or {}
        name = str(metadata.get("name") or "")
        if name == KUEUE_NAMESPACE:
            continue
        if not (name.startswith("kueue-") or name.startswith("demo-")):
            continue
        test_namespaces.append(name)

    for namespace in test_namespaces:
        pods = _kubectl_json_optional(["get", "pods", "-n", namespace]).get("items", [])
        for item in pods:
            name = str((item.get("metadata", {}) or {}).get("name") or "")
            if not name:
                continue
            _delete_resource("pod", name, namespace=namespace)

        workloads = _kubectl_json_optional(["get", "workloads.kueue.x-k8s.io", "-n", namespace]).get("items", [])
        for item in workloads:
            name = str((item.get("metadata", {}) or {}).get("name") or "")
            if not name:
                continue
            _clear_finalizers("workload.kueue.x-k8s.io", name, namespace=namespace)
            _delete_resource("workload.kueue.x-k8s.io", name, namespace=namespace)

        localqueues = _kubectl_json_optional(["get", "localqueues.kueue.x-k8s.io", "-n", namespace]).get("items", [])
        for item in localqueues:
            name = str((item.get("metadata", {}) or {}).get("name") or "")
            if not name:
                continue
            _delete_resource("localqueue.kueue.x-k8s.io", name, namespace=namespace)

    clusterqueues = _kubectl_json_optional(["get", "clusterqueues.kueue.x-k8s.io"]).get("items", [])
    for item in clusterqueues:
        metadata = item.get("metadata", {}) or {}
        name = str(metadata.get("name") or "")
        if not name or (
            name != "training-cluster-queue"
            and not name.startswith("cq-kueue-")
            and not name.startswith("cq-demo-")
        ):
            continue
        _clear_finalizers("clusterqueue.kueue.x-k8s.io", name)
        _delete_resource("clusterqueue.kueue.x-k8s.io", name)

    cohorts = _kubectl_json_optional(["get", "cohorts.kueue.x-k8s.io"]).get("items", [])
    for item in cohorts:
        metadata = item.get("metadata", {}) or {}
        name = str(metadata.get("name") or "")
        if not name.startswith("cohort-"):
            continue
        _clear_finalizers("cohort.kueue.x-k8s.io", name)
        _delete_resource("cohort.kueue.x-k8s.io", name)

    resource_flavors = _kubectl_json_optional(["get", "resourceflavors.kueue.x-k8s.io"]).get("items", [])
    for item in resource_flavors:
        metadata = item.get("metadata", {}) or {}
        name = str(metadata.get("name") or "")
        if not name.startswith("rf-"):
            continue
        _clear_finalizers("resourceflavor.kueue.x-k8s.io", name)
        _delete_resource("resourceflavor.kueue.x-k8s.io", name)

    topologies = _kubectl_json_optional(["get", "topologies.kueue.x-k8s.io"]).get("items", [])
    for item in topologies:
        metadata = item.get("metadata", {}) or {}
        name = str(metadata.get("name") or "")
        if not name:
            continue
        _clear_finalizers("topology.kueue.x-k8s.io", name)
        _delete_resource("topology.kueue.x-k8s.io", name)

    for name in test_namespaces:
        _delete_resource("namespace", name)

    time.sleep(3)


def _pod_key(pod: dict) -> str:
    metadata = pod.get("metadata", {}) or {}
    name = str(metadata.get("name") or "")
    namespace = str(metadata.get("namespace") or "")
    uid = str(metadata.get("uid") or "")
    return uid or f"{namespace}/{name}"


def _delete_pod(namespace: str, name: str) -> None:
    _run(
        ["kubectl", "delete", "pod", name, "-n", namespace, "--ignore-not-found=true", "--wait=false"],
        capture_output=False,
        check=False,
    )


def _runtime_annotation_flavor_keys(annotations: dict) -> list[str]:
    prefix = "admirl.ai/scaled-runtime-"
    flavors = []
    for key in annotations or {}:
        if not str(key).startswith(prefix):
            continue
        flavor = str(key)[len(prefix):].strip()
        if flavor:
            flavors.append(flavor)
    return sorted(flavors, key=len, reverse=True)


def _runtime_flavor_for_node(node_name: str, annotations: dict) -> str:
    text = str(node_name or "")
    if not text:
        return ""
    for flavor in _runtime_annotation_flavor_keys(annotations):
        if flavor in text:
            return flavor
    return ""


def _resolve_synthetic_runtime_annotations(namespace: str, pods: list[dict]) -> None:
    for pod in pods:
        metadata = pod.get("metadata", {}) or {}
        annotations = metadata.get("annotations", {}) or {}
        spec = pod.get("spec", {}) or {}
        node_name = str(spec.get("nodeName", "") or "")
        if not node_name:
            continue
        resolved_flavor = _runtime_flavor_for_node(node_name, annotations)
        if not resolved_flavor:
            continue
        runtime_key = f"admirl.ai/scaled-runtime-{resolved_flavor}"
        runtime_value = annotations.get(runtime_key)
        if runtime_value is None:
            continue
        current_value = annotations.get("admirl.ai/scaled-runtime-seconds")
        current_flavor = annotations.get("admirl.ai/resolved-runtime-flavor")
        if str(current_value) == str(runtime_value) and str(current_flavor) == resolved_flavor:
            continue
        name = str(metadata.get("name") or "")
        if not name:
            continue
        _run(
            [
                "kubectl",
                "annotate",
                "pod",
                name,
                "-n",
                namespace,
                f"admirl.ai/scaled-runtime-seconds={runtime_value}",
                f"admirl.ai/resolved-runtime-flavor={resolved_flavor}",
                "--overwrite",
            ],
            capture_output=False,
            check=False,
        )


def _reap_synthetic_completions(namespace: str, pods: list[dict], finalized_pods: dict[str, dict]) -> None:
    now = time.time()
    for pod in pods:
        key = _pod_key(pod)
        if key in finalized_pods:
            continue
        synthetic_finished_at = _synthetic_finish_time(pod)
        if synthetic_finished_at is None or synthetic_finished_at > now:
            continue
        snapshot = copy.deepcopy(pod)
        snapshot.setdefault("status", {})
        snapshot["status"]["phase"] = "Succeeded"
        finalized_pods[key] = snapshot
        name = str((pod.get("metadata", {}) or {}).get("name") or "")
        if name:
            _delete_pod(namespace, name)


def wait_for_arm_completion(
    namespace: str,
    expected_pods: int,
    timeout_seconds: float,
    finalized_pods: dict[str, dict],
    *,
    meta: dict,
    time_scale: float,
    workload_preset: str,
    arm_name: str,
    seed: int,
) -> None:
    run_started_at = time.time()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        pods = _kubectl_json(["get", "pods", "-n", namespace]).get("items", [])
        _resolve_synthetic_runtime_annotations(namespace, pods)
        pods = _kubectl_json(["get", "pods", "-n", namespace]).get("items", [])
        _reap_synthetic_completions(namespace, pods, finalized_pods)
        now = time.time()
        completed_keys = set(finalized_pods)
        for item in pods:
            key = _pod_key(item)
            if key in completed_keys:
                continue
            phase = ((item.get("status", {}) or {}).get("phase", "")) or ""
            synthetic_finished_at = _synthetic_finish_time(item)
            if phase in {"Succeeded", "Failed"} or (synthetic_finished_at is not None and synthetic_finished_at <= now):
                completed_keys.add(key)
        live_metrics = collect_arm_metrics(namespace, meta, finalized_pods=finalized_pods, time_scale=time_scale)
        publish_model_server_benchmark_snapshot(
            _build_benchmark_progress_payload(
                live_metrics,
                workload_preset=workload_preset,
                arm_name=arm_name,
                seed=seed,
                namespace=namespace,
                phase="running",
                elapsed_seconds=now - run_started_at,
                timeout_seconds=timeout_seconds,
                active=True,
            )
        )
        if len(completed_keys) >= expected_pods:
            return
        time.sleep(3)
    publish_model_server_benchmark_snapshot(
        _build_benchmark_progress_payload(
            collect_arm_metrics(namespace, meta, finalized_pods=finalized_pods, time_scale=time_scale),
            workload_preset=workload_preset,
            arm_name=arm_name,
            seed=seed,
            namespace=namespace,
            phase="timed_out",
            elapsed_seconds=max(0.0, timeout_seconds),
            timeout_seconds=timeout_seconds,
            active=False,
        )
    )
    completed_count = len(set(finalized_pods))
    print(
        f"[WARN] timed out waiting for namespace {namespace} to finish "
        f"({completed_count}/{expected_pods} pods completed in {timeout_seconds:.0f}s) — "
        f"collecting partial results",
        flush=True,
    )


def _condition_time(conditions: list[dict], kind: str, status: str = "True") -> float | None:
    for cond in conditions or []:
        if cond.get("type") == kind and str(cond.get("status")) == status:
            return _iso_to_ts(cond.get("lastTransitionTime"))
    return None


def _synthetic_finish_time(pod: dict) -> float | None:
    metadata = pod.get("metadata", {}) or {}
    annotations = metadata.get("annotations", {}) or {}
    runtime_raw = annotations.get("admirl.ai/scaled-runtime-seconds")
    if runtime_raw is None:
        return None
    try:
        runtime_seconds = float(runtime_raw)
    except (TypeError, ValueError):
        return None
    started_at = _iso_to_ts((pod.get("status", {}) or {}).get("startTime"))
    if started_at is None:
        return None
    return started_at + runtime_seconds


def _group_pod_rows(pods: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for pod in pods:
        metadata = pod.get("metadata", {}) or {}
        labels = metadata.get("labels", {}) or {}
        annotations = metadata.get("annotations", {}) or {}
        status = pod.get("status", {}) or {}
        workload_name = str(
            annotations.get("admirl.ai/workload-name")
            or labels.get("admirl.ai/workload-name")
            or labels.get("job-name")
            or labels.get("kueue.x-k8s.io/pod-group-name")
            or metadata.get("name", "")
        )
        group = workload_name
        entry = groups.setdefault(
            group,
            {
                "pods_by_key": {},
                "queue_class": labels.get("admirl.ai/workload-class", ""),
                "topology_preference": annotations.get("admirl.ai/topology-domain", ""),
                "expected_count": int(
                    annotations.get("admirl.ai/final-worker-count")
                    or annotations.get("kueue.x-k8s.io/pod-group-total-count")
                    or "1"
                ),
                "workload_name": workload_name,
            },
        )
        scheduled_at = _condition_time(status.get("conditions", []) or [], "PodScheduled")
        started_at = _iso_to_ts(status.get("startTime"))
        finished_at = None
        for container in status.get("containerStatuses", []) or []:
            term = (((container.get("state") or {}).get("terminated")) or {})
            finished_at = max(finished_at or 0.0, _iso_to_ts(term.get("finishedAt")) or 0.0) or finished_at
        synthetic_finished_at = _synthetic_finish_time(pod)
        key = _pod_key(pod)
        row = {
            "created_at": _iso_to_ts(metadata.get("creationTimestamp")),
            "scheduled_at": scheduled_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "synthetic_finished_at": synthetic_finished_at,
            "phase": status.get("phase", ""),
            "node_name": ((pod.get("spec") or {}).get("nodeName") or ""),
        }
        entry["pods_by_key"][key] = row
    return groups


def collect_arm_metrics(
    namespace: str,
    meta: dict,
    finalized_pods: dict[str, dict] | None = None,
    *,
    time_scale: float = 1.0,
) -> dict:
    pods = _kubectl_json(["get", "pods", "-n", namespace]).get("items", [])
    jobs = _kubectl_json_optional(["get", "jobs.batch", "-n", namespace]).get("items", [])
    workloads = _kubectl_json(["get", "workloads.kueue.x-k8s.io", "-n", namespace]).get("items", [])
    nodes = _kubectl_json(["get", "nodes"]).get("items", [])
    clusterqueues = _kubectl_json_optional(["get", "clusterqueues.kueue.x-k8s.io"]).get("items", [])
    localqueues = _kubectl_json_optional(["get", "localqueues.kueue.x-k8s.io", "-n", namespace]).get("items", [])
    resourceflavors = _kubectl_json_optional(["get", "resourceflavors.kueue.x-k8s.io"]).get("items", [])
    topologies = _kubectl_json_optional(["get", "topologies.kueue.x-k8s.io"]).get("items", [])
    node_domain = {
        item["metadata"]["name"]: ((item.get("metadata", {}) or {}).get("labels", {}) or {}).get("nvlink-domain", "")
        for item in nodes
    }
    node_flavor = {
        item["metadata"]["name"]: ((item.get("metadata", {}) or {}).get("labels", {}) or {}).get("admirl.ai/flavor", "")
        for item in nodes
    }
    workload_status = {}
    for workload in workloads:
        workload_status[workload["metadata"]["name"]] = workload
    job_status = {}
    for job in jobs:
        job_status[_workload_name_from_object(job)] = job

    groups = _group_pod_rows(pods)
    finalized_groups = _group_pod_rows(list((finalized_pods or {}).values()))
    for group_name, group in finalized_groups.items():
        entry = groups.setdefault(
            group_name,
            {
                "pods_by_key": {},
                "queue_class": group["queue_class"],
                "topology_preference": group["topology_preference"],
                "expected_count": group["expected_count"],
                "workload_name": group["workload_name"],
            },
        )
        for key, row in group["pods_by_key"].items():
            entry["pods_by_key"].setdefault(key, row)

    meta_rows = {row["workload_id"]: row for row in meta.get("workloads", [])}
    wait_values: list[float] = []
    completion_values: list[float] = []
    gang_wait_values: list[float] = []
    gang_completion_values: list[float] = []
    gang_total = 0
    gang_completed = 0
    topology_total = 0
    topology_hits = 0
    group_rows = []
    head_gang_arrival = None
    head_gang_wait = None
    head_gang_current_blocked = None
    head_gang_workload = ""
    head_gang_admit = None
    head_gang_candidate_flavors: list[str] = []
    small_job_bypass_while_gang_pending = 0
    small_job_head_flavor_admissions = 0
    small_job_head_flavor_gpu = 0.0
    now = time.time()

    for group_name, group in sorted(groups.items()):
        pods_in_group = list(group["pods_by_key"].values())
        created_at = min((item["created_at"] for item in pods_in_group if item["created_at"] is not None), default=None)
        started_at = min(
            (
                item["started_at"]
                for item in pods_in_group
                if item["started_at"] is not None
            ),
            default=None,
        )
        scheduled_at = min(
            (
                item["scheduled_at"]
                for item in pods_in_group
                if item["scheduled_at"] is not None
            ),
            default=None,
        )
        finished_at = max(
            (
                item["finished_at"]
                for item in pods_in_group
                if item["finished_at"] is not None
            ),
            default=None,
        )
        synthetic_finished_at = max(
            (
                item["synthetic_finished_at"]
                for item in pods_in_group
                if item["synthetic_finished_at"] is not None
            ),
            default=None,
        )
        effective_finished_at = finished_at or synthetic_finished_at
        workload_name = group["workload_name"]
        workload_obj = workload_status.get(group_name) or workload_status.get(workload_name)
        job_obj = job_status.get(workload_name)
        admitted_at = None
        if workload_obj is not None:
            conditions = (workload_obj.get("status", {}) or {}).get("conditions", []) or []
            admitted_at = _condition_time(conditions, "QuotaReserved") or _condition_time(conditions, "Admitted")
        job_created_at = _iso_to_ts(((job_obj or {}).get("metadata", {}) or {}).get("creationTimestamp"))
        if job_created_at is not None:
            created_at = job_created_at
        wait_start = admitted_at or scheduled_at or started_at
        wait_seconds = max(0.0, wait_start - created_at) if created_at is not None and wait_start is not None else None
        completion_seconds = (
            max(0.0, effective_finished_at - created_at)
            if created_at is not None and effective_finished_at is not None
            else None
        )
        meta_row = meta_rows.get(workload_name, {})
        expected_count = int(meta_row.get("selected_worker_count") or meta_row.get("worker_count") or group["expected_count"] or 1)
        synthetic_complete = len(pods_in_group) >= expected_count and all(
            item["phase"] == "Succeeded"
            or (
                item["synthetic_finished_at"] is not None
                and item["synthetic_finished_at"] <= now
            )
            for item in pods_in_group
        )
        complete = (_job_complete(job_obj) if job_obj is not None else False) or synthetic_complete

        if wait_seconds is not None:
            wait_values.append(wait_seconds)
        if completion_seconds is not None:
            completion_values.append(completion_seconds)

        if group["queue_class"] == "gang":
            gang_total += 1
            if wait_seconds is not None:
                gang_wait_values.append(wait_seconds)
            if completion_seconds is not None:
                gang_completion_values.append(completion_seconds)
            if complete:
                gang_completed += 1

        topo_pref = str(group["topology_preference"] or "").strip().upper()
        if topo_pref:
            topology_total += 1
            node_domains = [node_domain.get(item["node_name"], "") for item in pods_in_group if item["node_name"]]
            if node_domains and all(domain == topo_pref for domain in node_domains):
                topology_hits += 1
        admitted_flavors = sorted(
            {
                flavor
                for flavor in (node_flavor.get(item["node_name"], "") for item in pods_in_group if item["node_name"])
                if flavor
            }
        )
        used_spare_capacity = any(
            ("-spare-" in str(item.get("node_name") or "")) or ("-prov-" in str(item.get("node_name") or ""))
            for item in pods_in_group
            if item.get("node_name")
        )

        runtime_seconds = meta_row.get("runtime_seconds")
        arrival_time = meta_row.get("arrival_time")
        scaled_arrival_time = (
            float(arrival_time) / max(time_scale, 1.0)
            if arrival_time is not None
            else None
        )
        if group["queue_class"] == "gang" and arrival_time is not None:
            if head_gang_arrival is None or float(scaled_arrival_time) < float(head_gang_arrival):
                head_gang_arrival = float(scaled_arrival_time)
                head_gang_wait = wait_seconds
                head_gang_current_blocked = (
                    float(wait_seconds)
                    if wait_seconds is not None
                    else (max(0.0, now - created_at) if created_at is not None else None)
                )
                head_gang_workload = workload_name
                head_gang_candidate_flavors = list(meta_row.get("candidate_flavors", []) or [])
                head_gang_admit = (
                    float(scaled_arrival_time) + float(wait_seconds)
                    if wait_seconds is not None
                    else None
                )
        group_rows.append(
            {
                "group_name": group_name,
                "workload_name": workload_name,
                "queue_class": group["queue_class"],
                "topology_preference": group["topology_preference"],
                "elastic_enabled": bool(meta_row.get("elastic_enabled", False)),
                "worker_count": expected_count,
                "total_gpu": meta_row.get("selected_total_gpu", meta_row.get("total_gpu")),
                "arrival_time": arrival_time,
                "scaled_arrival_time": scaled_arrival_time,
                "runtime_seconds": runtime_seconds,
                "candidate_flavors": list(meta_row.get("candidate_flavors", []) or []),
                "admitted_flavors": admitted_flavors,
                "used_spare_capacity": used_spare_capacity,
                "created_at": created_at,
                "admitted_at": admitted_at,
                "scheduled_at": scheduled_at,
                "started_at": started_at,
                "wait_seconds": wait_seconds,
                "completion_seconds": completion_seconds,
                "completed": complete,
            }
        )

    small_job_bypass_while_gang_pending = _count_small_job_bypass_while_gang_pending(
        group_rows,
        head_gang_arrival=head_gang_arrival,
        head_gang_admit=head_gang_admit,
        head_gang_candidate_flavors=head_gang_candidate_flavors,
    )
    small_job_head_flavor_admissions, small_job_head_flavor_gpu = _small_job_head_flavor_admissions_while_gang_pending(
        group_rows,
        head_gang_arrival=head_gang_arrival,
        head_gang_admit=head_gang_admit,
        head_gang_candidate_flavors=head_gang_candidate_flavors,
    )

    gang_slowdowns = []
    for row in group_rows:
        if row["queue_class"] != "gang":
            continue
        runtime_seconds = row.get("runtime_seconds")
        completion_seconds = row.get("completion_seconds")
        if runtime_seconds and completion_seconds:
            gang_slowdowns.append(float(completion_seconds) / max(float(runtime_seconds), 1.0))

    metrics = {
        "namespace": namespace,
        "cluster_nodes_total": len(nodes),
        "namespace_pods_total": len(pods),
        "namespace_workloads_total": len(workloads),
        "clusterqueues_total": len(clusterqueues),
        "localqueues_total": len(localqueues),
        "resourceflavors_total": len(resourceflavors),
        "topologies_total": len(topologies),
        "jobs_total": len(meta.get("workloads", [])),
        "jobs_completed": sum(1 for row in group_rows if row["completed"]),
        "avg_workload_wait_seconds": (sum(wait_values) / len(wait_values)) if wait_values else 0.0,
        "p95_workload_wait_seconds": _percentile(wait_values, 95),
        "p99_workload_wait_seconds": _percentile(wait_values, 99),
        "avg_job_completion_seconds": (sum(completion_values) / len(completion_values)) if completion_values else 0.0,
        "p95_job_completion_seconds": _percentile(completion_values, 95),
        "p99_job_completion_seconds": _percentile(completion_values, 99),
        "avg_gang_wait_seconds": (sum(gang_wait_values) / len(gang_wait_values)) if gang_wait_values else 0.0,
        "p95_gang_wait_seconds": _percentile(gang_wait_values, 95),
        "p99_gang_wait_seconds": _percentile(gang_wait_values, 99),
        "avg_gang_completion_seconds": (sum(gang_completion_values) / len(gang_completion_values)) if gang_completion_values else 0.0,
        "p95_gang_completion_seconds": _percentile(gang_completion_values, 95),
        "p99_gang_completion_seconds": _percentile(gang_completion_values, 99),
        "gang_admission_ratio": (gang_completed / gang_total) if gang_total else 0.0,
        "head_gang_blocked_seconds": float(head_gang_wait or 0.0),
        "head_gang_current_blocked_seconds": float(head_gang_current_blocked or 0.0),
        "head_gang_workload": head_gang_workload,
        "small_job_bypass_count_while_gang_pending": small_job_bypass_while_gang_pending,
        "small_job_head_flavor_admissions_while_gang_pending": small_job_head_flavor_admissions,
        "small_job_head_flavor_gpu_while_gang_pending": small_job_head_flavor_gpu,
        "gang_slowdown_vs_isolated": (sum(gang_slowdowns) / len(gang_slowdowns)) if gang_slowdowns else 0.0,
        "topology_hit_rate": (topology_hits / topology_total) if topology_total else 0.0,
        "raw_groups": group_rows,
    }
    return _augment_arm_metrics(metrics)


def run_arm(
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
    time_scale: float,
    checkpoint_path: str | None,
    runtime_policy_override: str | None = None,
) -> dict:
    namespace = _sanitize_namespace(f"kueue-{arm.name}-{int(time.time())}")
    live_feasible_only = True
    oversample_factor = 8
    if canonical_kueue_preset(workload_preset) in {
        "kueue-lingjun-gang-starvation",
        "kueue-lingjun-gang-starvation-cohort",
        "kueue-lingjun-gang-topology-provisioning",
        "kueue-lingjun-gang-elastic-topology",
        "kueue-lingjun-gang-elastic-profile-cohort",
    }:
        live_feasible_only = True
        oversample_factor = 1
    arm_dir, meta = generate_arm_artifacts(
        output_root=output_root,
        arm=arm,
        workload_preset=workload_preset,
        cluster_layout=cluster_layout,
        seed=seed,
        num_jobs=num_jobs,
        arrival_span=arrival_span,
        trace_split=trace_split,
        trace_train_fraction=trace_train_fraction,
        runtime_scale=runtime_scale,
        include_provisioning=False,
        namespace=namespace,
        live_feasible_only=live_feasible_only,
        oversample_factor=oversample_factor,
    )

    runtime_policy = runtime_policy_override or arm.runtime_policy or (
        "learned_multi_objective" if arm.kueue_mode == "learned" else "blocked_guard"
    )
    elastic_policy_mode = arm.elastic_policy or "disabled"
    loaded_policy = None
    if elastic_policy_mode == "learned_multi_objective" and checkpoint_path:
        loaded_policy = load_runtime_policy(checkpoint_path)
    cleanup_kueue_test_state()
    configure_model_server(
        runtime_policy=runtime_policy,
        checkpoint_path=checkpoint_path if runtime_policy == "learned_multi_objective" else None,
    )
    reset_model_server_runtime_metrics()
    apply_kueue_controller(learned_enabled=(arm.kueue_mode == "learned"))

    provisioner_procs: list[tuple[subprocess.Popen[str], object]] = []
    try:
        _run(
            [
                "kubectl",
                "apply",
                "--server-side",
                "--force-conflicts",
                "--request-timeout=30s",
                "-f",
                str(arm_dir / "setup.yaml"),
            ],
            capture_output=False,
        )
        provisioner_procs = start_provisioners(
            arm_dir,
            canonical_kueue_preset(workload_preset),
            cluster_layout,
            workloads=list(meta.get("workloads", [])),
            time_scale=time_scale,
        )
        submit_workloads_live_with_policy(
            arm=arm,
            arm_dir=arm_dir,
            meta=meta,
            time_scale=time_scale,
            runtime_scale=runtime_scale,
            namespace=namespace,
            elastic_policy_mode=elastic_policy_mode,
            loaded_policy=loaded_policy,
        )

        scaled_arrivals = [float(item["arrival_time"]) / max(time_scale, 1.0) for item in meta["workloads"]]
        scaled_runtimes = [max(5.0, float(item["runtime_seconds"]) / max(runtime_scale, 1.0)) for item in meta["workloads"]]
        # Sum ALL arrivals and runtimes: workloads execute serially so the
        # total wall-clock is roughly max(arrivals) + sum(runtimes) when pods
        # must wait for predecessors to drain before being admitted.
        timeout_seconds = max(
            600.0,
            max(scaled_arrivals, default=0.0) + sum(scaled_runtimes) + 300.0,
        )
        finalized_pods: dict[str, dict] = {}
        expected_pods = sum(int(item.get("selected_worker_count", item["worker_count"])) for item in meta["workloads"])
        wait_for_arm_completion(
            namespace,
            expected_pods=expected_pods,
            timeout_seconds=timeout_seconds,
            finalized_pods=finalized_pods,
            meta=meta,
            time_scale=time_scale,
            workload_preset=workload_preset,
            arm_name=arm.name,
            seed=seed,
        )
        metrics = collect_arm_metrics(namespace, meta, finalized_pods=finalized_pods, time_scale=time_scale)
        metrics["arm"] = arm.name
        metrics["kueue_mode"] = arm.kueue_mode
        metrics["queueing_strategy"] = arm.queueing_strategy
        metrics["runtime_policy"] = runtime_policy
        metrics["elastic_policy"] = elastic_policy_mode
        metrics["timeout_seconds"] = timeout_seconds
        metrics["model_server_status"] = model_server_status()
        publish_model_server_benchmark_snapshot(
            _build_benchmark_progress_payload(
                metrics,
                workload_preset=workload_preset,
                arm_name=arm.name,
                seed=seed,
                namespace=namespace,
                phase="completed",
                elapsed_seconds=float(metrics.get("makespan_seconds", 0.0) or 0.0),
                timeout_seconds=timeout_seconds,
                active=False,
            )
        )
        return metrics
    finally:
        for proc, log_file in provisioner_procs:
            stop_process(proc, log_file)
        cleanup_arm(arm_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live Kueue admission benchmark matrix on the KWOK cluster")
    parser.add_argument("--workload-preset", default="kueue-lingjun-gang-starvation-cohort")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-jobs", type=int, default=12)
    parser.add_argument("--arrival-span", type=float, default=120.0)
    parser.add_argument("--trace-split", default="test")
    parser.add_argument("--trace-train-fraction", type=float, default=0.75)
    parser.add_argument("--runtime-scale", type=float, default=120.0)
    parser.add_argument("--time-scale", type=float, default=30.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--arm", action="append", dest="arms", default=[])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--runtime-policy-override", default="", help="Override the model-server runtime policy for all selected arms")
    parser.add_argument(
        "--summarize-existing-root",
        type=Path,
        default=None,
        help="Rewrite summary and paper metrics from saved live-results.json files without rerunning the cluster",
    )
    args = parser.parse_args()

    if args.summarize_existing_root is not None:
        summary_path = _write_result_pack(args.summarize_existing_root, _load_existing_result_pack(args.summarize_existing_root))
        print(json.dumps({"summary_path": str(summary_path), "mode": "summarize-existing"}, indent=2))
        return

    if not is_kueue_preset(args.workload_preset):
        raise SystemExit(f"unsupported Kueue workload preset: {args.workload_preset}")

    selected_arms = [arm for arm in ARMS if not args.arms or arm.name in set(args.arms)]
    if not selected_arms:
        raise SystemExit("no benchmark arms selected")
    effective_runtime_policy = args.runtime_policy_override.strip().lower() or None
    requires_checkpoint = False
    for arm in selected_arms:
        policy_name = effective_runtime_policy or arm.runtime_policy or (
            "learned_multi_objective" if arm.kueue_mode == "learned" else "blocked_guard"
        )
        if policy_name == "learned_multi_objective":
            requires_checkpoint = True
            break
    if requires_checkpoint and not args.checkpoint:
        raise SystemExit("--checkpoint is required when any learned arm is selected")

    ensure_model_server()
    source_dir = kueue_source_dir()
    build_kueue_image()

    canonical = canonical_kueue_preset(args.workload_preset)
    cluster_layout = layout_for_preset(canonical, None)
    kwok_layout = PRESET_TO_KWOK_LAYOUT[canonical]
    apply_kwok_training_nodes(kwok_layout)

    args.output_root.mkdir(parents=True, exist_ok=True)
    result_pack = {
        "workload_preset": canonical,
        "seed": args.seed,
        "num_jobs": args.num_jobs,
        "arrival_span": args.arrival_span,
        "trace_split": args.trace_split,
        "trace_train_fraction": args.trace_train_fraction,
        "runtime_scale": args.runtime_scale,
        "time_scale": args.time_scale,
        "kueue_source": _kueue_source_metadata(source_dir),
        "arms": {},
    }
    for arm in selected_arms:
        metrics = run_arm(
            output_root=args.output_root,
            arm=arm,
            workload_preset=canonical,
            cluster_layout=cluster_layout,
            seed=args.seed,
            num_jobs=args.num_jobs,
            arrival_span=args.arrival_span,
            trace_split=args.trace_split,
            trace_train_fraction=args.trace_train_fraction,
            runtime_scale=args.runtime_scale,
            time_scale=args.time_scale,
            checkpoint_path=args.checkpoint or None,
            runtime_policy_override=effective_runtime_policy,
        )
        result_pack["arms"][arm.name] = metrics
    summary_path = _write_result_pack(args.output_root, result_pack)
    print(json.dumps({"summary_path": str(summary_path), "arms": list(result_pack["arms"])}, indent=2))


if __name__ == "__main__":
    main()
