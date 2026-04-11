from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from model_server.kueue_rl.cluster import NodeState
    from model_server.kueue_rl.config import MAX_NODE_GPU, MAX_QUEUE_JOBS, MAX_TIME_HORIZON, MAX_WAIT_TIME, STATE_DIM
    from model_server.kueue_rl.kueue_admission import (
        CandidateAction,
        KueueWorkload,
        KueueFairShareState,
        blocked_head_context,
        canonical_kueue_preset,
        build_candidate_actions,
        is_kueue_preset,
        kueue_nodes_for_layout,
        kueue_state_vector,
        workload_profile_flavor_match_scalar,
        workload_profile_flavor_mismatch_scalar,
        workload_profile_prefer_a_scalar,
        workload_profile_prefer_c_scalar,
    )
    from model_server.kueue_rl.model import ActorCritic
except ModuleNotFoundError:
    from kueue_rl.cluster import NodeState
    from kueue_rl.config import MAX_NODE_GPU, MAX_QUEUE_JOBS, MAX_TIME_HORIZON, MAX_WAIT_TIME, STATE_DIM
    from kueue_rl.kueue_admission import (
        CandidateAction,
        KueueWorkload,
        KueueFairShareState,
        blocked_head_context,
        canonical_kueue_preset,
        build_candidate_actions,
        is_kueue_preset,
        kueue_nodes_for_layout,
        kueue_state_vector,
        workload_profile_flavor_match_scalar,
        workload_profile_flavor_mismatch_scalar,
        workload_profile_prefer_a_scalar,
        workload_profile_prefer_c_scalar,
    )
    from kueue_rl.model import ActorCritic

_RUNTIME_HEAVY_PRESETS = {
    "kueue-lingjun-gang-topology-provisioning",
    "kueue-lingjun-gang-elastic-topology",
    "kueue-lingjun-gang-elastic-profile-cohort",
}

_STARVATION_GUARD_PRESETS = {
    "kueue-lingjun-gang-starvation",
    "kueue-lingjun-gang-starvation-cohort",
}


@dataclass
class LoadedRuntimePolicy:
    checkpoint_path: str
    workload_preset: str
    cluster_layout: str
    num_jobs: int
    arrival_span: float
    model: ActorCritic


def load_runtime_policy(checkpoint_path: str) -> LoadedRuntimePolicy:
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    state_dim = int(payload.get("state_dim", STATE_DIM))
    n_actions = int(payload.get("n_actions", MAX_QUEUE_JOBS))
    if state_dim != STATE_DIM or n_actions != MAX_QUEUE_JOBS:
        raise ValueError(
            f"checkpoint {path} is incompatible with current Kueue admission config: "
            f"state_dim={state_dim} actions={n_actions}, expected {STATE_DIM}/{MAX_QUEUE_JOBS}"
        )

    workload_preset = canonical_kueue_preset(str(payload.get("workload_preset", "")))
    if not is_kueue_preset(workload_preset):
        raise ValueError(f"checkpoint {path} is not a Kueue admission checkpoint")

    model = ActorCritic(state_dim=STATE_DIM, n_actions=MAX_QUEUE_JOBS)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    return LoadedRuntimePolicy(
        checkpoint_path=str(path),
        workload_preset=workload_preset,
        cluster_layout=str(payload.get("cluster_layout", "")),
        num_jobs=int(payload.get("num_jobs", MAX_QUEUE_JOBS)),
        arrival_span=float(payload.get("arrival_span", 180.0)),
        model=model,
    )


def is_kueue_request(request_state: dict) -> bool:
    return str(request_state.get("request_mode", "") or "").strip() == "kueue-admission"


def _node_from_payload(payload: dict) -> NodeState:
    node = NodeState(
        name=str(payload.get("name", "")),
        domain=str(payload.get("domain", payload.get("nvlink_domain", ""))),
        cpu_total=max(0, int(payload.get("cpu_total", payload.get("alloc_cpu_milli", 0) or 0))),
        mem_total=max(0, int(payload.get("mem_total", payload.get("alloc_mem_bytes", 0) or 0))),
        gpu_total=max(0, int(payload.get("gpu_total", 0) or 0)),
    )
    node.free_cpu = max(0, int(payload.get("free_cpu", payload.get("free_cpu_milli", 0) or 0)))
    node.free_mem = max(0, int(payload.get("free_mem", payload.get("free_mem_bytes", 0) or 0)))
    node.free_gpu = max(0, int(payload.get("free_gpu", payload.get("gpu_free", 0) or 0)))
    return node


def _nodes_from_request(request_state: dict) -> list[NodeState]:
    raw_nodes = request_state.get("nodes")
    if isinstance(raw_nodes, list) and raw_nodes:
        return [_node_from_payload(item) for item in raw_nodes]
    cluster = request_state.get("cluster", {}) or {}
    if isinstance(cluster.get("nodes"), list) and cluster["nodes"]:
        return [_node_from_payload(item) for item in cluster["nodes"]]
    layout = str(request_state.get("cluster_layout", "") or "")
    if layout:
        return kueue_nodes_for_layout(layout)
    return []


def _candidate_from_payload(payload: dict) -> CandidateAction:
    return CandidateAction(
        action_id=str(payload.get("action_id", "")),
        workload_id=str(payload.get("workload_id", "")),
        profile_name=str(payload.get("workload_profile", payload.get("profile_name", payload.get("workload_id", "")))),
        flavor_name=str(payload.get("flavor_name", "")),
        queue_name=str(payload.get("queue_name", "")),
        cluster_queue=str(payload.get("cluster_queue", "")),
        fairshare_group=str(payload.get("fairshare_group", "")),
        priority=int(payload.get("priority", 0) or 0),
        wait_seconds=float(payload.get("wait_seconds", 0.0) or 0.0),
        runtime_seconds=float(payload.get("runtime_seconds", 0.0) or 0.0),
        worker_count=int(payload.get("worker_count", 1) or 1),
        total_gpu=int(payload.get("total_gpu", 0) or 0),
        per_worker_gpu=int(payload.get("per_worker_gpu", 0) or 0),
        topology_aware=bool(payload.get("topology_aware", False)),
        topology_preference=str(payload.get("topology_preference", "")),
        flavor_domain=str(payload.get("flavor_domain", "")),
        immediate_fit=bool(payload.get("immediate_fit", False)),
        provisionable=bool(payload.get("provisionable", False)),
        available_gpu=int(payload.get("available_gpu", 0) or 0),
        total_gpu_capacity=int(payload.get("total_gpu_capacity", 0) or 0),
        fairshare_debt=float(payload.get("fairshare_debt", 0.0) or 0.0),
        requeue_count=int(payload.get("requeue_count", 0) or 0),
        queue_class=str(payload.get("queue_class", "small")),
        flavor_gpu_size=int(payload.get("flavor_gpu_size", 0) or 0),
        oversize_gpu=int(payload.get("oversize_gpu", 0) or 0),
        competing_older_pressure=float(payload.get("competing_older_pressure", 0.0) or 0.0),
        elastic_enabled=bool(payload.get("elastic_enabled", False)),
        min_worker_count=int(payload.get("min_worker_count", payload.get("worker_count", 1)) or 1),
        preferred_worker_count=int(payload.get("preferred_worker_count", payload.get("worker_count", 1)) or 1),
        max_worker_count=int(payload.get("max_worker_count", payload.get("worker_count", 1)) or 1),
        scale_tag=str(payload.get("scale_tag", "fixed")),
        scale_fraction=float(payload.get("scale_fraction", 1.0) or 1.0),
    )


def _candidates_from_request(request_state: dict) -> list[CandidateAction]:
    raw = request_state.get("candidates") or []
    return [_candidate_from_payload(item) for item in raw]


def _sorted_runtime_candidates(candidates: list[CandidateAction]) -> list[CandidateAction]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.wait_seconds,
            -item.priority,
            item.workload_id,
            item.flavor_name,
            item.worker_count,
            item.scale_tag,
            item.action_id,
        ),
    )


def _base_candidate_priority(candidate: CandidateAction) -> float:
    score = float(candidate.priority) * 20.0
    score += min(candidate.wait_seconds, MAX_WAIT_TIME) / 10.0
    if candidate.immediate_fit:
        score += 2.0
    elif candidate.provisionable:
        score += 1.0
    score -= max(0.0, candidate.fairshare_debt) * 8.0
    return score


def _blocked_guard_base_priority(candidate: CandidateAction) -> float:
    score = _base_candidate_priority(candidate)
    if candidate.elastic_enabled:
        if candidate.scale_tag == "max":
            score += 4.0
        elif candidate.scale_tag == "preferred":
            score += 2.0
        else:
            score -= 1.0
        if candidate.competing_older_pressure > 0.0 and candidate.scale_tag == "min":
            score += 7.0
        if candidate.competing_older_pressure > 0.0 and candidate.scale_tag == "max":
            score -= 7.0
    if candidate.queue_class == "gang":
        score += 10.0 + float(candidate.worker_count) * 3.0 + float(candidate.total_gpu) * 0.75
    else:
        score += float(candidate.total_gpu) * 0.25
    if candidate.flavor_gpu_size == candidate.per_worker_gpu:
        score += 2.0
    return score


def _blocked_guard_scores(candidates: list[CandidateAction]) -> tuple[dict[str, float], str]:
    context = blocked_head_context(candidates)
    blocked_workload_id = str(context.get("blocked_workload_id", "") or "")
    blocked_flavors = set(context.get("blocked_flavors", set()) or set())
    workload_groups = dict(context.get("workload_groups", {}) or {})

    pair_scores = {candidate.action_id: _blocked_guard_base_priority(candidate) for candidate in candidates}
    if not blocked_workload_id or not blocked_flavors:
        return pair_scores, "blocked-guard-fallback-kueue"

    for candidate in candidates:
        score = float(pair_scores.get(candidate.action_id, 0.0))
        if candidate.workload_id == blocked_workload_id:
            score += 220.0
            if candidate.provisionable:
                score += 40.0
            if candidate.immediate_fit:
                score += 120.0
            pair_scores[candidate.action_id] = score
            continue

        workload_items = workload_groups.get(candidate.workload_id, [])
        workload_flavors = {item.flavor_name for item in workload_items}
        overlaps_blocked = bool(blocked_flavors) and not workload_flavors.isdisjoint(blocked_flavors)

        if candidate.immediate_fit and overlaps_blocked and candidate.queue_class != "gang":
            pair_scores[candidate.action_id] = score - 600.0
            continue
        if candidate.immediate_fit and not overlaps_blocked:
            pair_scores[candidate.action_id] = score + 15.0
            continue
        if overlaps_blocked and candidate.queue_class != "gang":
            pair_scores[candidate.action_id] = score - 40.0
    return pair_scores, "blocked-guard-kueue"


def _workloads_from_candidates(candidates: list[CandidateAction]) -> list[KueueWorkload]:
    workloads: dict[str, KueueWorkload] = {}
    flavors_by_workload: dict[str, list[str]] = defaultdict(list)
    for item in candidates:
        if item.workload_id not in workloads:
            workloads[item.workload_id] = KueueWorkload(
                workload_id=item.workload_id,
                queue_name=item.queue_name,
                cluster_queue=item.cluster_queue,
                fairshare_group=item.fairshare_group,
                queue_class="gang" if item.max_worker_count >= 2 else item.queue_class,
                priority=item.priority,
                worker_count=item.max_worker_count,
                per_worker_gpu=item.per_worker_gpu,
                per_worker_cpu_milli=0,
                per_worker_mem_bytes=0,
                total_gpu=item.max_worker_count * item.per_worker_gpu,
                runtime_seconds=item.runtime_seconds,
                arrival_time=0.0,
                topology_aware=item.topology_aware,
                topology_preference=item.topology_preference,
                restart_count=item.requeue_count,
                candidate_flavors=(),
                elastic_enabled=item.elastic_enabled,
                min_worker_count=item.min_worker_count,
                preferred_worker_count=item.preferred_worker_count,
                max_worker_count=item.max_worker_count,
                profile_name=item.profile_name,
        )
        flavors_by_workload[item.workload_id].append(item.flavor_name)
    result = []
    for workload_id, workload in workloads.items():
        result.append(
            KueueWorkload(
                **{
                    **workload.__dict__,
                    "candidate_flavors": tuple(dict.fromkeys(flavors_by_workload[workload_id])),
                }
            )
        )
    return result


def _mask_from_candidates(candidates: list[CandidateAction]) -> list[float]:
    mask = [0.0] * MAX_QUEUE_JOBS
    for index, item in enumerate(candidates[:MAX_QUEUE_JOBS]):
        if item.immediate_fit or item.provisionable:
            mask[index] = 1.0
    return mask


def _apply_guardrails(
    *,
    candidates: list[CandidateAction],
    pair_scores: dict[str, float],
) -> dict[str, float]:
    context = blocked_head_context(candidates)
    blocked_workload_id = str(context.get("blocked_workload_id", "") or "")
    blocked_flavors = set(context.get("blocked_flavors", set()) or set())
    workload_groups = dict(context.get("workload_groups", {}) or {})
    if not blocked_workload_id or not blocked_flavors:
        return pair_scores

    adjusted = dict(pair_scores)
    for candidate in candidates:
        score = float(adjusted.get(candidate.action_id, 0.0))
        if candidate.workload_id == blocked_workload_id:
            # Strongly prefer advancing the blocked head, especially when it is
            # close to fitting or already provisionable.
            score += 250.0
            if candidate.immediate_fit:
                score += 500.0
            if candidate.provisionable:
                score += 120.0
            if candidate.worker_count >= 2:
                score += 80.0
            adjusted[candidate.action_id] = score
            continue

        workload_flavors = {item.flavor_name for item in workload_groups.get(candidate.workload_id, [])}
        overlaps_blocked = not workload_flavors.isdisjoint(blocked_flavors)
        if candidate.immediate_fit and overlaps_blocked and candidate.queue_class != "gang":
            # This is the key starvation guardrail: don't let same-flavor small
            # jobs keep consuming the scarce pool while an older gang is stuck.
            adjusted[candidate.action_id] = score - 1000.0
            continue
        if candidate.immediate_fit and not overlaps_blocked:
            # Safe bypass on disjoint flavors is still desirable.
            adjusted[candidate.action_id] = score + 80.0
    return adjusted


def _candidate_prob_rows(candidates: list[CandidateAction], probs: list[float]) -> list[tuple[CandidateAction, float]]:
    rows: list[tuple[CandidateAction, float]] = []
    for index, candidate in enumerate(candidates[:MAX_QUEUE_JOBS]):
        if not (candidate.immediate_fit or candidate.provisionable):
            continue
        learned_score = float(probs[index]) if index < len(probs) else 0.0
        rows.append((candidate, learned_score))
    return rows


def _safe_immediate_override(
    *,
    candidates: list[CandidateAction],
    probs: list[float],
) -> dict[str, str | float] | None:
    rows = _candidate_prob_rows(candidates, probs)
    if len(rows) < 2:
        return None

    baseline_top = max(
        (candidate for candidate, _ in rows),
        key=lambda candidate: (_blocked_guard_base_priority(candidate), candidate.action_id),
    )
    learned_top, learned_prob = max(rows, key=lambda item: (item[1], item[0].action_id))
    if learned_top.action_id == baseline_top.action_id:
        return None
    if not learned_top.immediate_fit or baseline_top.immediate_fit:
        return None
    if baseline_top.queue_class == "gang" or learned_top.queue_class == "gang":
        return None
    if not baseline_top.provisionable:
        return None

    context = blocked_head_context(candidates)
    workload_groups = dict(context.get("workload_groups", {}) or {})
    learned_flavors = {item.flavor_name for item in workload_groups.get(learned_top.workload_id, [])}
    baseline_flavors = {item.flavor_name for item in workload_groups.get(baseline_top.workload_id, [])}
    if learned_flavors and baseline_flavors and not learned_flavors.isdisjoint(baseline_flavors):
        return None

    baseline_prob = next(
        (score for candidate, score in rows if candidate.action_id == baseline_top.action_id),
        0.0,
    )
    if learned_prob <= baseline_prob:
        return None

    return {
        "promote_action_id": learned_top.action_id,
        "demote_action_id": baseline_top.action_id,
        "promote_probability": learned_prob,
        "demote_probability": baseline_prob,
    }


def _apply_safe_immediate_override(
    *,
    candidates: list[CandidateAction],
    probs: list[float],
    pair_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, str | float] | None]:
    override = _safe_immediate_override(candidates=candidates, probs=probs)
    if override is None or not pair_scores:
        return pair_scores, None

    adjusted = dict(pair_scores)
    promote_action_id = str(override["promote_action_id"])
    demote_action_id = str(override["demote_action_id"])
    top_score = max(float(score) for score in adjusted.values())
    adjusted[promote_action_id] = max(float(adjusted.get(promote_action_id, 0.0)), top_score + 1.0)
    adjusted[demote_action_id] = float(adjusted.get(demote_action_id, 0.0)) - 1.0
    return adjusted, override


def _elastic_safe_override(
    *,
    candidates: list[CandidateAction],
    probs: list[float],
) -> dict[str, object] | None:
    context = blocked_head_context(candidates)
    blocked_flavors = set(context.get("blocked_flavors", set()) or set())
    if not blocked_flavors:
        return None

    rows = _candidate_prob_rows(candidates, probs)
    by_workload: dict[str, list[tuple[CandidateAction, float]]] = defaultdict(list)
    for candidate, learned_prob in rows:
        if not candidate.elastic_enabled:
            continue
        by_workload[candidate.workload_id].append((candidate, learned_prob))

    best_override: dict[str, object] | None = None
    best_margin = float("-inf")
    for workload_id, workload_rows in by_workload.items():
        risky_rows = [
            (candidate, learned_prob)
            for candidate, learned_prob in workload_rows
            if candidate.flavor_name in blocked_flavors
            and (candidate.immediate_fit or candidate.provisionable)
            and candidate.competing_older_pressure > 0.0
        ]
        if not risky_rows:
            continue
        safe_rows = [
            (candidate, learned_prob)
            for candidate, learned_prob in workload_rows
            if candidate.immediate_fit
            and candidate.scale_fraction < 1.0
            and candidate.flavor_name not in blocked_flavors
        ]
        if not safe_rows or not risky_rows:
            continue

        promote_candidate, promote_prob = max(safe_rows, key=lambda item: (item[1], -item[0].scale_fraction, item[0].action_id))
        risky_top_prob = max(float(item[1]) for item in risky_rows)
        margin = float(promote_prob) - risky_top_prob
        if margin > best_margin:
            best_margin = margin
            best_override = {
                "workload_id": workload_id,
                "promote_action_id": promote_candidate.action_id,
                "demote_action_ids": [candidate.action_id for candidate, _ in risky_rows if candidate.action_id != promote_candidate.action_id],
                "promote_probability": float(promote_prob),
                "risky_top_probability": float(risky_top_prob),
            }
    return best_override


def _apply_elastic_safe_override(
    *,
    candidates: list[CandidateAction],
    probs: list[float],
    pair_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, object] | None]:
    override = _elastic_safe_override(candidates=candidates, probs=probs)
    if override is None or not pair_scores:
        return pair_scores, None

    adjusted = dict(pair_scores)
    workload_id = str(override["workload_id"])
    workload_scores = [
        float(adjusted.get(candidate.action_id, 0.0))
        for candidate in candidates
        if candidate.workload_id == workload_id and candidate.action_id in adjusted
    ]
    if not workload_scores:
        return pair_scores, None

    promote_action_id = str(override["promote_action_id"])
    top_group_score = max(workload_scores)
    adjusted[promote_action_id] = max(float(adjusted.get(promote_action_id, 0.0)), top_group_score + 50.0)
    for action_id in list(override.get("demote_action_ids", []) or []):
        adjusted[str(action_id)] = float(adjusted.get(str(action_id), 0.0)) - 50.0
    return adjusted, override


def _runtime_time_from_request(request_state: dict, candidates: list[CandidateAction]) -> float:
    value = float(request_state.get("time", 0.0) or 0.0)
    if value > MAX_TIME_HORIZON:
        return max((float(item.wait_seconds) for item in candidates), default=0.0)
    return value


def _has_blocked_bypass_risk(candidates: list[CandidateAction], context: dict[str, object]) -> bool:
    blocked_workload_id = str(context.get("blocked_workload_id", "") or "")
    blocked_flavors = set(context.get("blocked_flavors", set()) or set())
    workload_groups = dict(context.get("workload_groups", {}) or {})
    blocked_group = workload_groups.get(blocked_workload_id, [])
    if not blocked_workload_id or not blocked_flavors or not blocked_group:
        return False

    blocked_total_gpu = max(int(item.total_gpu) for item in blocked_group)
    blocked_workers = max(int(item.worker_count) for item in blocked_group)
    candidate_ids = {item.action_id for item in candidates}

    for workload_id, items in workload_groups.items():
        if workload_id == blocked_workload_id:
            continue
        immediate_items = [item for item in items if item.action_id in candidate_ids and item.immediate_fit]
        if not immediate_items:
            continue
        workload_flavors = {item.flavor_name for item in immediate_items}
        if workload_flavors.isdisjoint(blocked_flavors):
            continue
        for item in immediate_items:
            if item.queue_class != "gang":
                return True
            if int(item.total_gpu) < blocked_total_gpu or int(item.worker_count) < blocked_workers:
                return True
    return False


def _multi_objective_baseline_scores(
    *,
    candidates: list[CandidateAction],
    risk_active: bool,
) -> tuple[dict[str, float], str]:
    if risk_active:
        scores, _ = _blocked_guard_scores(candidates)
        return scores, "blocked-guard"

    scores: dict[str, float] = {}
    for candidate in candidates:
        score = _blocked_guard_base_priority(candidate)
        if candidate.elastic_enabled:
            if candidate.scale_tag == "min" and candidate.competing_older_pressure > 0.0:
                score += 6.0
            elif candidate.scale_tag == "max" and candidate.competing_older_pressure > 0.0:
                score -= 6.0
        if not candidate.queue_class == "gang" and candidate.immediate_fit:
            score += 6.0
        if candidate.immediate_fit and candidate.flavor_gpu_size == candidate.per_worker_gpu:
            score += 3.0
        if candidate.oversize_gpu > 0:
            score -= min(float(candidate.oversize_gpu), float(MAX_NODE_GPU)) * 0.6
        if candidate.provisionable and not candidate.immediate_fit:
            score -= 1.5
        scores[candidate.action_id] = score
    return scores, "blocked-guard-base"


def _multi_objective_learned_scores(
    *,
    candidates: list[CandidateAction],
    probs: list[float],
    risk_active: bool,
    policy_preset: str,
) -> tuple[dict[str, float], str]:
    base_scores, base_source = _multi_objective_baseline_scores(candidates=candidates, risk_active=risk_active)
    if risk_active and policy_preset in _STARVATION_GUARD_PRESETS:
        guarded = _apply_guardrails(candidates=candidates, pair_scores=dict(base_scores))
        return guarded, "learned-multi-objective-starvation-guard-kueue"

    valid_rows = _candidate_prob_rows(candidates, probs)
    if not valid_rows:
        return base_scores, f"multi-objective-{base_source}-fallback-kueue"

    best_pressure_by_workload: dict[str, float] = {}
    for candidate in candidates:
        if not (candidate.immediate_fit or candidate.provisionable):
            continue
        current = best_pressure_by_workload.get(candidate.workload_id)
        pressure = float(candidate.competing_older_pressure)
        if current is None or pressure < current:
            best_pressure_by_workload[candidate.workload_id] = pressure

    mean_prob = sum(score for _, score in valid_rows) / float(len(valid_rows))
    adjusted = dict(base_scores)
    for candidate, learned_prob in valid_rows:
        residual = (float(learned_prob) - mean_prob)
        residual_scale = 30.0 if risk_active else 18.0
        if (not risk_active) and policy_preset in _RUNTIME_HEAVY_PRESETS:
            residual_scale = 84.0
        bonus = residual * residual_scale
        if candidate.immediate_fit and candidate.queue_class != "gang":
            bonus += residual * 6.0
        if candidate.immediate_fit and candidate.flavor_gpu_size == candidate.per_worker_gpu:
            bonus += max(0.0, residual) * 4.0
        adjusted[candidate.action_id] = float(adjusted.get(candidate.action_id, 0.0)) + bonus

        if candidate.elastic_enabled:
            best_pressure = best_pressure_by_workload.get(candidate.workload_id, float(candidate.competing_older_pressure))
            pressure_delta = max(0.0, float(candidate.competing_older_pressure) - best_pressure)
            if pressure_delta > 0.0:
                adjusted[candidate.action_id] -= 12.0 * pressure_delta
                if candidate.scale_fraction >= 1.0:
                    adjusted[candidate.action_id] -= 6.0 * pressure_delta
            prefers_specific_flavor = (
                workload_profile_prefer_a_scalar(candidate.workload_id, candidate.profile_name) > 0.0
                or workload_profile_prefer_c_scalar(candidate.workload_id, candidate.profile_name) > 0.0
            )
            if prefers_specific_flavor:
                if workload_profile_flavor_match_scalar(candidate.workload_id, candidate.flavor_name, candidate.profile_name) > 0.0:
                    adjusted[candidate.action_id] += 18.0
                    if candidate.scale_fraction >= 1.0:
                        adjusted[candidate.action_id] += 8.0
                    elif workload_profile_prefer_a_scalar(candidate.workload_id, candidate.profile_name) > 0.0:
                        adjusted[candidate.action_id] -= 2.0
                if workload_profile_flavor_mismatch_scalar(candidate.workload_id, candidate.flavor_name, candidate.profile_name) > 0.0:
                    adjusted[candidate.action_id] -= 24.0
                    if candidate.scale_fraction < 1.0:
                        adjusted[candidate.action_id] -= 10.0

    if risk_active:
        adjusted = _apply_guardrails(candidates=candidates, pair_scores=adjusted)
        adjusted, override = _apply_elastic_safe_override(
            candidates=candidates,
            probs=probs,
            pair_scores=adjusted,
        )
        return adjusted, "learned-multi-objective-elastic-safe-kueue" if override is not None else "learned-multi-objective-guarded-kueue"

    adjusted, override = _apply_safe_immediate_override(
        candidates=candidates,
        probs=probs,
        pair_scores=adjusted,
    )
    return adjusted, "learned-multi-objective-safe-kueue" if override is not None else "learned-multi-objective-kueue"


def build_kueue_admission_response(request_state: dict, policy=None, policy_mode: str = "blocked_guard") -> dict:
    candidates = _sorted_runtime_candidates(_candidates_from_request(request_state))
    if not candidates:
        return {
            "ranked_workloads": [],
            "workload_scores": {},
            "flavor_rankings": {},
            "pair_scores": {},
            "source": "empty-kueue",
        }

    nodes = _nodes_from_request(request_state)
    fair_share_violations = int(request_state.get("fair_share_violation_count", 0) or 0)
    blocked_seconds = float(request_state.get("blocked_seconds", 0.0) or 0.0)
    idle_quota = float(request_state.get("idle_quota_while_blocked", 0.0) or 0.0)
    workloads = _workloads_from_candidates(candidates)
    state = kueue_state_vector(
        candidates=candidates,
        waiting=workloads,
        running=[],
        future=[],
        nodes=nodes,
        time_now=_runtime_time_from_request(request_state, candidates),
        blocked_seconds=blocked_seconds,
        idle_quota_while_blocked=idle_quota,
        fair_share_violations=fair_share_violations,
    )

    normalized_policy_mode = str(policy_mode or "").strip().lower() or "blocked_guard"
    pair_scores: dict[str, float] = {}
    source = "blocked-guard-kueue"
    use_learned_policy = (
        policy is not None
        and is_kueue_preset(getattr(policy, "workload_preset", ""))
        and normalized_policy_mode == "learned_multi_objective"
    )
    if use_learned_policy:
        policy_preset = canonical_kueue_preset(str(getattr(policy, "workload_preset", "")))
        context = blocked_head_context(candidates)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(_mask_from_candidates(candidates), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = policy.model(state_tensor, mask_tensor)
            probs = dist.probs.squeeze(0).tolist()
        risk_active = _has_blocked_bypass_risk(candidates, context)
        pair_scores, source = _multi_objective_learned_scores(
            candidates=candidates,
            probs=probs,
            risk_active=risk_active,
            policy_preset=policy_preset,
        )
    elif normalized_policy_mode == "blocked_guard":
        pair_scores, source = _blocked_guard_scores(candidates)
    else:
        raise ValueError(
            f"unsupported Kueue runtime policy {normalized_policy_mode!r}; "
            "expected 'blocked_guard' or 'learned_multi_objective'"
        )

    workload_scores: dict[str, float] = {}
    flavor_rankings: dict[str, list[str]] = defaultdict(list)
    workload_to_pairs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for candidate in candidates:
        score = float(pair_scores.get(candidate.action_id, 0.0))
        workload_scores[candidate.workload_id] = max(workload_scores.get(candidate.workload_id, float("-inf")), score)
        workload_to_pairs[candidate.workload_id].append((candidate.flavor_name, score))

    for workload_id, pairs in workload_to_pairs.items():
        ordered = sorted(pairs, key=lambda item: (-item[1], item[0]))
        seen: set[str] = set()
        deduped: list[str] = []
        for name, _ in ordered:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        flavor_rankings[workload_id] = deduped

    ranked_workloads = [
        workload_id
        for workload_id, _ in sorted(workload_scores.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "ranked_workloads": ranked_workloads,
        "workload_scores": workload_scores,
        "flavor_rankings": dict(flavor_rankings),
        "pair_scores": pair_scores,
        "state_dim": len(state),
        "source": source,
    }
