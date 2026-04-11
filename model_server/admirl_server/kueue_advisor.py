from __future__ import annotations

from .kueue_runtime import (
    _candidates_from_request,
    _sorted_runtime_candidates,
    build_kueue_admission_response,
)


def _guard_sort_key(item) -> tuple[float, float, float, float, str]:
    return (
        -float(item.priority),
        -float(item.wait_seconds),
        -float(item.worker_count),
        -float(item.total_gpu),
        item.action_id,
    )


def _build_admission_guard_plan(candidates: list) -> dict[str, object]:
    grouped: dict[str, list] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.workload_id, []).append(candidate)

    blocked_heads: list[tuple[str, object]] = []
    for workload_id, items in grouped.items():
        immediate_any = any(item.immediate_fit for item in items)
        provisionable_any = any(item.provisionable for item in items)
        gang_any = any(item.worker_count >= 2 and item.total_gpu > item.per_worker_gpu for item in items)
        if immediate_any or not provisionable_any or not gang_any:
            continue
        blocked_heads.append((workload_id, sorted(items, key=_guard_sort_key)[0]))

    if not blocked_heads:
        return {
            "protected_workload": "",
            "protected_flavors": [],
            "protected_priority": 0,
            "protected_worker_count": 0,
            "protected_total_gpu": 0,
        }

    blocked_heads.sort(key=lambda item: _guard_sort_key(item[1]) + (item[0],))
    protected_workload_id, protected_candidate = blocked_heads[0]
    protected_items = grouped.get(protected_workload_id, [])
    protected_flavors = sorted({item.flavor_name for item in protected_items})
    return {
        "protected_workload": protected_workload_id,
        "protected_flavors": protected_flavors,
        "protected_priority": int(protected_candidate.priority),
        "protected_worker_count": int(protected_candidate.worker_count),
        "protected_total_gpu": int(protected_candidate.total_gpu),
    }


def build_kueue_admission_advice(request_state: dict, policy=None, policy_mode: str = "blocked_guard") -> dict:
    response = build_kueue_admission_response(
        request_state=request_state,
        policy=policy,
        policy_mode=policy_mode,
    )
    candidates = _sorted_runtime_candidates(_candidates_from_request(request_state))
    response.update(_build_admission_guard_plan(candidates))
    response["advisor_source"] = response.get("source", "kueue")
    return response
