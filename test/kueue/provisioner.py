from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_server.kueue_rl.kueue_admission import KUEUE_CLUSTER_LAYOUTS


def spare_node_manifest(flavor_name: str, index: int, gpu_count: int, domain: str, cpu: str, memory: str) -> dict:
    name = f"{flavor_name}-spare-{index:02d}"
    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {
            "name": name,
            "annotations": {
                "kwok.x-k8s.io/node": "fake",
                "node.alpha.kubernetes.io/ttl": "0",
            },
            "labels": {
                "kubernetes.io/arch": "amd64",
                "kubernetes.io/os": "linux",
                "kubernetes.io/hostname": name,
                "type": "kwok",
                "admirl.ai/spare": "true",
                "gpu": str(gpu_count),
                "nvlink-domain": domain,
                "admirl.ai/flavor": flavor_name,
                "admirl.ai/gpu-family": f"{gpu_count}gpu",
                "cloud.provider.com/node-group": flavor_name,
                "cloud.provider.com/topology-block": domain,
            },
        },
        "status": {
            "allocatable": {
                "cpu": cpu,
                "memory": memory,
                "pods": "110",
                "admirl.ai/gpu": str(gpu_count),
            },
            "capacity": {
                "cpu": cpu,
                "memory": memory,
                "pods": "110",
                "admirl.ai/gpu": str(gpu_count),
            },
            "conditions": [{"type": "Ready", "status": "True", "reason": "KubeletReady", "message": "kwok spare node ready"}],
        },
    }


def build_spare_nodes(layout: str, flavor_name: str, count: int) -> list[dict]:
    for group in KUEUE_CLUSTER_LAYOUTS[layout]:
        candidate = f"rf-{group['gpus']}gpu-{group['domain'].lower()}"
        if candidate == flavor_name:
            return [
                spare_node_manifest(
                    flavor_name=flavor_name,
                    index=index,
                    gpu_count=group["gpus"],
                    domain=group["domain"],
                    cpu=str(group["cpu_milli"] // 1000),
                    memory=f"{group['mem_bytes'] // (1024**3)}Gi",
                )
                for index in range(count)
            ]
    raise ValueError(f"unknown flavor {flavor_name!r} for layout {layout!r}")


def apply_spare_nodes(layout: str, flavor_name: str, count: int, dry_run: bool = False) -> list[dict]:
    docs = build_spare_nodes(layout, flavor_name, count)
    if dry_run:
        return docs
    payload = "\n---\n".join(json.dumps(doc, indent=2) for doc in docs)
    subprocess.run(["kubectl", "apply", "-f", "-"], input=payload, text=True, check=True)
    return docs


def main():
    parser = argparse.ArgumentParser(description="Provision extra KWOK fake nodes for a Kueue flavor after a delay")
    parser.add_argument("--layout", required=True, choices=sorted(KUEUE_CLUSTER_LAYOUTS))
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.delay_seconds > 0:
        time.sleep(args.delay_seconds)
    docs = apply_spare_nodes(args.layout, args.flavor, args.count, dry_run=args.dry_run)
    if args.output is not None:
        args.output.write_text("\n---\n".join(json.dumps(doc, indent=2) for doc in docs) + "\n", encoding="utf-8")
    else:
        print("\n---\n".join(json.dumps(doc, indent=2) for doc in docs))


if __name__ == "__main__":
    main()
