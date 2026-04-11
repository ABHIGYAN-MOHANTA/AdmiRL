from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeState:
    name: str
    domain: str
    cpu_total: int
    mem_total: int
    gpu_total: int
    free_cpu: int = field(init=False)
    free_mem: int = field(init=False)
    free_gpu: int = field(init=False)

    def __post_init__(self):
        self.free_cpu = self.cpu_total
        self.free_mem = self.mem_total
        self.free_gpu = self.gpu_total

    def _resolve_resources(self, resource=None, *, cpu_milli: int | None = None, mem_bytes: int | None = None, gpu: int | None = None) -> tuple[int, int, int]:
        if resource is not None:
            cpu_milli = int(getattr(resource, "cpu_milli"))
            mem_bytes = int(getattr(resource, "mem_bytes"))
            gpu = int(getattr(resource, "gpu"))
        if cpu_milli is None or mem_bytes is None or gpu is None:
            raise ValueError("resource or cpu_milli/mem_bytes/gpu must be provided")
        return int(cpu_milli), int(mem_bytes), int(gpu)

    def fits(self, resource=None, *, cpu_milli: int | None = None, mem_bytes: int | None = None, gpu: int | None = None) -> bool:
        cpu_milli, mem_bytes, gpu = self._resolve_resources(resource, cpu_milli=cpu_milli, mem_bytes=mem_bytes, gpu=gpu)
        return (
            self.free_cpu >= cpu_milli
            and self.free_mem >= mem_bytes
            and self.free_gpu >= gpu
        )

    def allocate(self, resource=None, *, cpu_milli: int | None = None, mem_bytes: int | None = None, gpu: int | None = None) -> None:
        cpu_milli, mem_bytes, gpu = self._resolve_resources(resource, cpu_milli=cpu_milli, mem_bytes=mem_bytes, gpu=gpu)
        self.free_cpu -= cpu_milli
        self.free_mem -= mem_bytes
        self.free_gpu -= gpu

    def release(self, resource=None, *, cpu_milli: int | None = None, mem_bytes: int | None = None, gpu: int | None = None) -> None:
        cpu_milli, mem_bytes, gpu = self._resolve_resources(resource, cpu_milli=cpu_milli, mem_bytes=mem_bytes, gpu=gpu)
        self.free_cpu = min(self.cpu_total, self.free_cpu + cpu_milli)
        self.free_mem = min(self.mem_total, self.free_mem + mem_bytes)
        self.free_gpu = min(self.gpu_total, self.free_gpu + gpu)

    def snapshot(self) -> "NodeState":
        copy = NodeState(
            name=self.name,
            domain=self.domain,
            cpu_total=self.cpu_total,
            mem_total=self.mem_total,
            gpu_total=self.gpu_total,
        )
        copy.free_cpu = self.free_cpu
        copy.free_mem = self.free_mem
        copy.free_gpu = self.free_gpu
        return copy


def topology_scalar(hint: str) -> float:
    mapping = {"": 0.0, "A": 0.25, "B": 0.5, "C": 0.75, "D": 1.0}
    return mapping.get(hint, 0.0)


def cluster_gpu_fragmentation(nodes: list[NodeState]) -> float:
    total_free = sum(max(0, node.free_gpu) for node in nodes)
    if total_free <= 0:
        return 0.0
    max_free = max(max(0, node.free_gpu) for node in nodes)
    return 1.0 - (max_free / total_free)


def cluster_gpu_utilization(nodes: list[NodeState]) -> float:
    total_gpu = sum(node.gpu_total for node in nodes)
    total_free = sum(node.free_gpu for node in nodes)
    if total_gpu <= 0:
        return 0.0
    return 1.0 - (total_free / total_gpu)


def largest_free_gpu_block(nodes: list[NodeState]) -> int:
    return max((max(0, node.free_gpu) for node in nodes), default=0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    q = max(0.0, min(100.0, float(q)))
    index = (len(ordered) - 1) * (q / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
