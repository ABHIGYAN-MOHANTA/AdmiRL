import argparse
import json


NODE_LAYOUTS = {
    "training-gang-starvation": [
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "A"},
        {"count": 2, "gpus": 2, "cpu": "16", "memory": "64Gi", "domain": "B"},
        {"count": 1, "gpus": 6, "cpu": "32", "memory": "128Gi", "domain": "C"},
    ],
    "training-gang-topology-provisioning": [
        {"count": 1, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "C"},
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "A"},
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "B"},
    ],
    "training-gang-elastic-topology": [
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "C"},
        {"count": 1, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "A"},
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "B"},
    ],
    "training-gang-elastic-profile-cohort": [
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "C"},
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "A"},
        {"count": 2, "gpus": 4, "cpu": "24", "memory": "96Gi", "domain": "B"},
    ],
}


def node_manifest(name, gpus, cpu, memory, domain):
    flavor = f"rf-{gpus}gpu-{domain.lower()}"
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
                "gpu": str(gpus),
                "nvlink-domain": domain,
                "admirl.ai/flavor": flavor,
                "admirl.ai/gpu-family": f"{gpus}gpu",
                "cloud.provider.com/node-group": flavor,
                "cloud.provider.com/topology-block": domain,
            },
        },
        "status": {
            "allocatable": {
                "cpu": cpu,
                "memory": memory,
                "pods": "110",
                "admirl.ai/gpu": str(gpus),
            },
            "capacity": {
                "cpu": cpu,
                "memory": memory,
                "pods": "110",
                "admirl.ai/gpu": str(gpus),
            },
            "conditions": [
                {"type": "Ready", "status": "True", "reason": "KubeletReady", "message": "kwok node ready"},
            ],
            "nodeInfo": {
                "architecture": "amd64",
                "operatingSystem": "linux",
                "kubeletVersion": "kwok",
                "kubeProxyVersion": "kwok",
            },
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Generate KWOK fake node YAML for distributed-training scheduling experiments")
    parser.add_argument("--layout", choices=sorted(NODE_LAYOUTS), default="training-gang-starvation")
    parser.add_argument("--prefix", default="kwok-gpu")
    args = parser.parse_args()

    docs = []
    index = 0
    for group in NODE_LAYOUTS[args.layout]:
        for _ in range(group["count"]):
            flavor = f"rf-{group['gpus']}gpu-{group['domain'].lower()}"
            docs.append(
                node_manifest(
                    name=f"{args.prefix}-{flavor}-{index:03d}",
                    gpus=group["gpus"],
                    cpu=group["cpu"],
                    memory=group["memory"],
                    domain=group["domain"],
                )
            )
            index += 1

    print("\n---\n".join(json.dumps(doc, indent=2) for doc in docs))


if __name__ == "__main__":
    main()
