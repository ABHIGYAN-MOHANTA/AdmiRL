from __future__ import annotations

import json
import math
from typing import Iterable

from model_server.kueue_rl.kueue_admission import (
    KUEUE_CLUSTER_LAYOUTS,
    KueueWorkload,
    default_cluster_layout_for_kueue_preset,
    workload_runtime_multiplier,
)


GPU_RESOURCE_NAME = "admirl.ai/gpu"


def _flavor_name(group: dict) -> str:
    return f"rf-{group['gpus']}gpu-{group['domain'].lower()}"


def _scaled_runtime_annotation_key(flavor_name: str) -> str:
    return f"admirl.ai/scaled-runtime-{flavor_name}"


def topology_doc() -> dict:
    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "Topology",
        "metadata": {"name": "default"},
        "spec": {
            "levels": [
                {"nodeLabel": "cloud.provider.com/topology-block"},
                {"nodeLabel": "kubernetes.io/hostname"},
            ]
        },
    }


def resource_flavor_docs(layout: str) -> list[dict]:
    docs = []
    seen = set()
    for group in KUEUE_CLUSTER_LAYOUTS[layout]:
        flavor = _flavor_name(group)
        if flavor in seen:
            continue
        seen.add(flavor)
        docs.append(
            {
                "apiVersion": "kueue.x-k8s.io/v1beta2",
                "kind": "ResourceFlavor",
                "metadata": {"name": flavor},
                "spec": {
                    "nodeLabels": {
                        "admirl.ai/flavor": flavor,
                        "cloud.provider.com/node-group": flavor,
                    },
                    "topologyName": "default",
                },
            }
        )
    return docs


def cluster_queue_doc(
    layout: str,
    queueing_strategy: str = "BestEffortFIFO",
    cohort_name: str = "research-cohort",
    include_provisioning: bool = True,
    cluster_queue_name: str = "training-cluster-queue",
) -> dict:
    flavors = []
    for group in KUEUE_CLUSTER_LAYOUTS[layout]:
        flavor = _flavor_name(group)
        nominal_gpu = group["count"] * group["gpus"]
        nominal_cpu = (group["count"] * group["cpu_milli"]) // 1000
        nominal_mem_gi = group["count"] * (group["mem_bytes"] // (1024**3))
        flavors.append(
            {
                "name": flavor,
                "resources": [
                    {"name": "cpu", "nominalQuota": str(nominal_cpu)},
                    {"name": "memory", "nominalQuota": f"{nominal_mem_gi}Gi"},
                    {"name": GPU_RESOURCE_NAME, "nominalQuota": str(nominal_gpu)},
                ],
            }
        )
    spec = {
        "cohortName": cohort_name,
        "namespaceSelector": {},
        "queueingStrategy": queueing_strategy,
        "admissionScope": {"admissionMode": "UsageBasedAdmissionFairSharing"},
        "resourceGroups": [
            {
                "coveredResources": ["cpu", "memory", GPU_RESOURCE_NAME],
                "flavors": flavors,
            }
        ],
    }
    if include_provisioning:
        spec["admissionChecksStrategy"] = {
            "admissionChecks": [
                {
                    "name": "training-provisioning",
                }
            ]
        }

    return {
        "apiVersion": "kueue.x-k8s.io/v1beta2",
        "kind": "ClusterQueue",
        "metadata": {"name": cluster_queue_name},
        "spec": spec,
    }


def _cluster_queue_name_map(
    workloads: Iterable[KueueWorkload],
    default_cluster_queue_name: str,
) -> dict[str, str]:
    logical_names = sorted({item.cluster_queue for item in workloads if item.cluster_queue})
    if not logical_names:
        return {"training-cluster-queue": default_cluster_queue_name}
    if logical_names == ["training-cluster-queue"]:
        return {"training-cluster-queue": default_cluster_queue_name}
    mapping: dict[str, str] = {}
    for logical_name in logical_names:
        suffix = logical_name.removeprefix("training-cluster-queue").strip("-") or "default"
        mapping[logical_name] = f"{default_cluster_queue_name}-{suffix}"[:63]
    return mapping


def local_queue_docs(
    workloads: Iterable[KueueWorkload],
    namespace: str = "default",
    cluster_queue_name: str = "training-cluster-queue",
) -> list[dict]:
    cluster_queue_map = _cluster_queue_name_map(workloads, cluster_queue_name)
    weights = {}
    for item in workloads:
        weights.setdefault(
            item.queue_name,
            {
                "weight": 1,
                "cluster_queue": cluster_queue_map.get(item.cluster_queue or "training-cluster-queue", cluster_queue_name),
            },
        )
    docs = []
    for queue_name, entry in sorted(weights.items()):
        docs.append(
            {
                "apiVersion": "kueue.x-k8s.io/v1beta2",
                "kind": "LocalQueue",
                "metadata": {
                    "namespace": namespace,
                    "name": queue_name,
                },
                "spec": {
                    "clusterQueue": entry["cluster_queue"],
                    "fairSharing": {"weight": str(entry["weight"])},
                },
            }
        )
    return docs


def _split_nominal_quota(total: int, buckets: int, index: int) -> int:
    if buckets <= 0:
        return max(0, total)
    base = max(0, total // buckets)
    remainder = max(0, total % buckets)
    return base + (1 if index < remainder else 0)


def admission_docs() -> list[dict]:
    return [
        {
            "apiVersion": "kueue.x-k8s.io/v1beta2",
            "kind": "AdmissionCheck",
            "metadata": {"name": "training-provisioning"},
            "spec": {
                "controllerName": "kueue.x-k8s.io/provisioning-request",
                "parameters": {
                    "apiGroup": "kueue.x-k8s.io",
                    "kind": "ProvisioningRequestConfig",
                    "name": "training-provisioning-config",
                },
            },
        },
        {
            "apiVersion": "kueue.x-k8s.io/v1beta2",
            "kind": "ProvisioningRequestConfig",
            "metadata": {"name": "training-provisioning-config"},
            "spec": {
                "provisioningClassName": "check-capacity.autoscaling.x-k8s.io",
                "managedResources": [GPU_RESOURCE_NAME],
                "retryStrategy": {
                    "backoffLimitCount": 1,
                    "backoffBaseSeconds": 0,
                    "backoffMaxSeconds": 600,
                },
            },
        },
    ]


def namespace_doc(namespace: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": namespace,
            "labels": {
                "kueue-managed": "true",
            },
        },
    }


def _scaled_runtime_seconds(runtime_seconds: float, runtime_scale: float) -> int:
    scale = max(float(runtime_scale), 1.0)
    return max(5, int(math.ceil(float(runtime_seconds) / scale)))


def _scaled_runtime_seconds_for_flavor(workload: KueueWorkload, flavor_name: str, runtime_scale: float) -> int:
    multiplier = workload_runtime_multiplier(workload.workload_id, flavor_name)
    return _scaled_runtime_seconds(float(workload.runtime_seconds) * float(multiplier), runtime_scale)


def live_worker_resource_requests(workload: KueueWorkload) -> tuple[int, int]:
    gpu = max(1, int(workload.per_worker_gpu))
    if gpu >= 8:
        return 32, 128
    if gpu >= 6:
        return 30, 120
    if gpu >= 4:
        return 20, 80
    if gpu >= 2:
        return 12, 48
    return 8, 32


def _candidate_flavor_affinity(workload: KueueWorkload) -> dict | None:
    allowed_flavors = [flavor for flavor in workload.candidate_flavors if flavor]
    if not allowed_flavors:
        return None
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "admirl.ai/flavor",
                                "operator": "In",
                                "values": allowed_flavors,
                            }
                        ]
                    }
                ]
            }
        }
    }


def _k8s_name(token: str, *, limit: int = 50) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in str(token or "").lower())
    cleaned = cleaned.strip("-")
    return (cleaned or "workload")[:limit]


def job_docs(
    workload: KueueWorkload,
    *,
    initial_worker_count: int | None = None,
    elastic_enabled: bool | None = None,
    namespace: str = "default",
    scheduler_name: str = "default-scheduler",
    runtime_scale: float = 60.0,
) -> list[dict]:
    worker_count = max(1, int(initial_worker_count if initial_worker_count is not None else workload.worker_count))
    elastic_job = bool(workload.elastic_enabled if elastic_enabled is None else elastic_enabled)
    runtime_by_flavor = {
        flavor: _scaled_runtime_seconds_for_flavor(workload, flavor, runtime_scale)
        for flavor in workload.candidate_flavors
        if flavor
    }
    scaled_runtime_seconds = max(
        runtime_by_flavor.values(),
        default=_scaled_runtime_seconds(workload.runtime_seconds, runtime_scale),
    )
    request_cpu, request_mem_gi = live_worker_resource_requests(workload)
    affinity = _candidate_flavor_affinity(workload)
    job_name = _k8s_name(workload.workload_id, limit=50)
    runtime_env = [
        {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "BASE_RUNTIME_SECONDS", "value": str(scaled_runtime_seconds)},
    ]
    for flavor_name, flavor_runtime in sorted(runtime_by_flavor.items()):
        env_name = "RUNTIME_" + "".join(ch if ch.isalnum() else "_" for ch in flavor_name.upper())
        runtime_env.append({"name": env_name, "value": str(flavor_runtime)})
    runtime_script = "runtime=\"$BASE_RUNTIME_SECONDS\"; "
    if runtime_by_flavor:
        runtime_script += "case \"$NODE_NAME\" in "
        for flavor_name in sorted(runtime_by_flavor):
            env_name = "RUNTIME_" + "".join(ch if ch.isalnum() else "_" for ch in flavor_name.upper())
            runtime_script += f"*{flavor_name}*) runtime=\"${env_name}\" ;; "
        runtime_script += "esac; "
    runtime_script += "sleep \"$runtime\""

    job_annotations = {
        "admirl.ai/workload-name": workload.workload_id,
        "admirl.ai/fairshare-group": workload.fairshare_group,
        "admirl.ai/topology-domain": workload.topology_preference,
        "admirl.ai/runtime-seconds": f"{float(workload.runtime_seconds):.3f}",
        "admirl.ai/scaled-runtime-seconds": str(scaled_runtime_seconds),
        "admirl.ai/final-worker-count": str(max(workload.worker_count, workload.max_worker_count)),
        "admirl.ai/initial-worker-count": str(worker_count),
        "admirl.ai/min-worker-count": str(workload.min_worker_count),
        "admirl.ai/preferred-worker-count": str(workload.preferred_worker_count),
        "admirl.ai/max-worker-count": str(workload.max_worker_count),
        "admirl.ai/elastic-enabled": "true" if workload.elastic_enabled else "false",
        **{
            _scaled_runtime_annotation_key(flavor_name): str(flavor_runtime)
            for flavor_name, flavor_runtime in runtime_by_flavor.items()
        },
    }
    if elastic_job:
        job_annotations["kueue.x-k8s.io/elastic-job"] = "true"

    base_labels = {
        "kueue.x-k8s.io/queue-name": workload.queue_name,
        "admirl.ai/workload-class": workload.queue_class,
        "admirl.ai/workload-name": workload.workload_id,
        "admirl.ai/elastic-managed": "true" if workload.elastic_enabled else "false",
    }
    if workload.topology_preference:
        base_labels["admirl.ai/topology-preference"] = workload.topology_preference
        base_labels["gpu-topology-hint"] = workload.topology_preference

    return [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "labels": {
                    **base_labels,
                    "kueue.x-k8s.io/queue-name": workload.queue_name,
                },
                "annotations": job_annotations,
            },
            "spec": {
                "parallelism": worker_count,
                "completions": worker_count,
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": base_labels,
                        "annotations": job_annotations,
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "schedulerName": scheduler_name,
                        **({"affinity": affinity} if affinity else {}),
                        "containers": [
                            {
                                "name": "worker",
                                "image": "busybox",
                                "command": ["sh", "-c", runtime_script],
                                "env": runtime_env,
                                "resources": {
                                    "requests": {
                                        "cpu": str(request_cpu),
                                        "memory": f"{request_mem_gi}Gi",
                                        GPU_RESOURCE_NAME: str(workload.per_worker_gpu),
                                    },
                                    "limits": {
                                        GPU_RESOURCE_NAME: str(workload.per_worker_gpu),
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        }
    ]


def pod_group_docs(
    workload: KueueWorkload,
    namespace: str = "default",
    scheduler_name: str = "default-scheduler",
    runtime_scale: float = 60.0,
) -> list[dict]:
    docs = []
    pod_group = workload.workload_id.replace("_", "-")[:50]
    runtime_by_flavor = {
        flavor: _scaled_runtime_seconds_for_flavor(workload, flavor, runtime_scale)
        for flavor in workload.candidate_flavors
        if flavor
    }
    scaled_runtime_seconds = max(runtime_by_flavor.values(), default=_scaled_runtime_seconds(workload.runtime_seconds, runtime_scale))
    request_cpu, request_mem_gi = live_worker_resource_requests(workload)
    affinity = _candidate_flavor_affinity(workload)
    runtime_env = [
        {
            "name": "NODE_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
        {
            "name": "BASE_RUNTIME_SECONDS",
            "value": str(scaled_runtime_seconds),
        },
    ]
    for flavor_name, flavor_runtime in sorted(runtime_by_flavor.items()):
        env_name = "RUNTIME_" + "".join(ch if ch.isalnum() else "_" for ch in flavor_name.upper())
        runtime_env.append({"name": env_name, "value": str(flavor_runtime)})
    runtime_script = "runtime=\"$BASE_RUNTIME_SECONDS\"; "
    if runtime_by_flavor:
        runtime_script += "case \"$NODE_NAME\" in "
        for flavor_name in sorted(runtime_by_flavor):
            env_name = "RUNTIME_" + "".join(ch if ch.isalnum() else "_" for ch in flavor_name.upper())
            runtime_script += f"*{flavor_name}*) runtime=\"${env_name}\" ;; "
        runtime_script += "esac; "
    runtime_script += "sleep \"$runtime\""
    base_labels = {
        "kueue.x-k8s.io/queue-name": workload.queue_name,
        "kueue.x-k8s.io/pod-group-name": pod_group,
        "admirl.ai/workload-class": workload.queue_class,
    }
    if workload.topology_preference:
        base_labels["admirl.ai/topology-preference"] = workload.topology_preference
        base_labels["gpu-topology-hint"] = workload.topology_preference
    for index in range(workload.worker_count):
        docs.append(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": f"{pod_group}-{index}",
                    "namespace": namespace,
                    "labels": {
                        **base_labels,
                        "kueue.x-k8s.io/pod-group-pod-index": str(index),
                    },
                    "annotations": {
                        "kueue.x-k8s.io/pod-group-total-count": str(workload.worker_count),
                        "admirl.ai/workload-name": workload.workload_id,
                        "admirl.ai/fairshare-group": workload.fairshare_group,
                        "admirl.ai/topology-domain": workload.topology_preference,
                        "admirl.ai/provisioning-pending": "false",
                        "admirl.ai/runtime-seconds": f"{float(workload.runtime_seconds):.3f}",
                        "admirl.ai/scaled-runtime-seconds": str(scaled_runtime_seconds),
                        **{
                            _scaled_runtime_annotation_key(flavor_name): str(flavor_runtime)
                            for flavor_name, flavor_runtime in runtime_by_flavor.items()
                        },
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "schedulerName": scheduler_name,
                    **({"affinity": affinity} if affinity else {}),
                    "containers": [
                        {
                            "name": "worker",
                            "image": "busybox",
                            "command": ["sh", "-c", runtime_script],
                            "env": runtime_env,
                            "resources": {
                                "requests": {
                                    "cpu": str(request_cpu),
                                    "memory": f"{request_mem_gi}Gi",
                                    GPU_RESOURCE_NAME: str(workload.per_worker_gpu),
                                },
                                "limits": {
                                    GPU_RESOURCE_NAME: str(workload.per_worker_gpu),
                                },
                            },
                        }
                    ],
                },
            }
        )
    return docs


def benchmark_setup_docs(
    workloads: Iterable[KueueWorkload],
    *,
    layout: str,
    queueing_strategy: str = "BestEffortFIFO",
    namespace: str = "default",
    include_provisioning: bool = True,
    cluster_queue_name: str = "training-cluster-queue",
    cohort_name: str = "research-cohort",
) -> list[dict]:
    docs: list[dict] = [namespace_doc(namespace), topology_doc()]
    docs.extend(resource_flavor_docs(layout))
    cluster_queue_map = _cluster_queue_name_map(workloads, cluster_queue_name)
    if layout == "kueue-gang-starvation" and set(cluster_queue_map) == {"training-cluster-queue-gang", "training-cluster-queue-small"}:
        quota_docs = [
            (cluster_queue_map["training-cluster-queue-gang"], 4, 20, "80Gi"),
            (cluster_queue_map["training-cluster-queue-small"], 4, 20, "80Gi"),
        ]
        for cq_name, gpu_quota, cpu_quota, mem_quota in quota_docs:
            spec = {
                "cohortName": cohort_name,
                "namespaceSelector": {},
                "queueingStrategy": queueing_strategy,
                "admissionScope": {"admissionMode": "UsageBasedAdmissionFairSharing"},
                "resourceGroups": [
                    {
                        "coveredResources": ["cpu", "memory", GPU_RESOURCE_NAME],
                        "flavors": [
                            {
                                "name": "rf-4gpu-a",
                                "resources": [
                                    {"name": "cpu", "nominalQuota": str(cpu_quota)},
                                    {"name": "memory", "nominalQuota": mem_quota},
                                    {"name": GPU_RESOURCE_NAME, "nominalQuota": str(gpu_quota)},
                                ],
                            }
                        ],
                    }
                ],
            }
            if include_provisioning:
                spec["admissionChecksStrategy"] = {"admissionChecks": [{"name": "training-provisioning"}]}
            docs.append(
                {
                    "apiVersion": "kueue.x-k8s.io/v1beta2",
                    "kind": "ClusterQueue",
                    "metadata": {"name": cq_name},
                    "spec": spec,
                }
            )
    elif len(cluster_queue_map) > 1:
        logical_items = sorted(cluster_queue_map.items())
        for cq_index, (_, cq_name) in enumerate(logical_items):
            flavors = []
            for group in KUEUE_CLUSTER_LAYOUTS[layout]:
                flavor = _flavor_name(group)
                nominal_gpu = group["count"] * group["gpus"]
                nominal_cpu = (group["count"] * group["cpu_milli"]) // 1000
                nominal_mem_gi = group["count"] * (group["mem_bytes"] // (1024**3))
                flavors.append(
                    {
                        "name": flavor,
                        "resources": [
                            {"name": "cpu", "nominalQuota": str(max(1, _split_nominal_quota(nominal_cpu, len(logical_items), cq_index)))},
                            {"name": "memory", "nominalQuota": f"{max(1, _split_nominal_quota(nominal_mem_gi, len(logical_items), cq_index))}Gi"},
                            {"name": GPU_RESOURCE_NAME, "nominalQuota": str(max(1, _split_nominal_quota(nominal_gpu, len(logical_items), cq_index)))},
                        ],
                    }
                )
            spec = {
                "cohortName": cohort_name,
                "namespaceSelector": {},
                "queueingStrategy": queueing_strategy,
                "admissionScope": {"admissionMode": "UsageBasedAdmissionFairSharing"},
                "resourceGroups": [
                    {
                        "coveredResources": ["cpu", "memory", GPU_RESOURCE_NAME],
                        "flavors": flavors,
                    }
                ],
            }
            if include_provisioning:
                spec["admissionChecksStrategy"] = {"admissionChecks": [{"name": "training-provisioning"}]}
            docs.append(
                {
                    "apiVersion": "kueue.x-k8s.io/v1beta2",
                    "kind": "ClusterQueue",
                    "metadata": {"name": cq_name},
                    "spec": spec,
                }
            )
    else:
        docs.append(
            cluster_queue_doc(
                layout,
                queueing_strategy=queueing_strategy,
                cohort_name=cohort_name,
                include_provisioning=include_provisioning,
                cluster_queue_name=cluster_queue_name,
            )
        )
    docs.extend(local_queue_docs(workloads, namespace=namespace, cluster_queue_name=cluster_queue_name))
    if include_provisioning:
        docs.extend(admission_docs())
    return docs


def render_yaml(docs: Iterable[dict]) -> str:
    return "\n---\n".join(json.dumps(doc, indent=2) for doc in docs if doc) + "\n"


def layout_for_preset(workload_preset: str, cluster_layout: str | None) -> str:
    return cluster_layout or default_cluster_layout_for_kueue_preset(workload_preset)
