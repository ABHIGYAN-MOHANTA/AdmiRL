from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.kueue.run_live_kueue_matrix import (
    IMPORTANT_GANG_METRICS,
    LOWER_IS_BETTER,
    _build_arm_comparison,
    ensure_model_server,
)
from model_server.admirl_server.kueue_runtime import load_runtime_policy


DEFAULT_LOCAL_KUEUE_DIR = Path(
    os.environ.get("ADMIRL_LOCAL_KUEUE_SOURCE_DIR", str(Path.home() / "Desktop" / "kueue"))
).expanduser().resolve()
DEFAULT_UPSTREAM_LOCAL_DIR = Path(
    os.environ.get("ADMIRL_UPSTREAM_KUEUE_SOURCE_DIR", str(Path.home() / "Desktop" / "Upstream" / "kueue"))
).expanduser().resolve()
DEFAULT_UPSTREAM_KUEUE_URL = "https://github.com/kubernetes-sigs/kueue.git"
DEFAULT_UPSTREAM_KUEUE_REF = "v0.15.0"
SEED_SET = [7, 11, 13, 17, 23]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    label: str
    workload_preset: str
    arm: str
    source_mode: str
    source_dir: str | None = None
    git_url: str | None = None
    git_ref: str | None = None
    checkpoint_path: str | None = None
    runtime_policy: str | None = None


def _requires_checkpoint(spec: ArmSpec) -> bool:
    return spec.arm.startswith("learned")


def _validate_checkpoint_for_spec(spec: ArmSpec) -> None:
    if not _requires_checkpoint(spec):
        return
    checkpoint_path = str(spec.checkpoint_path or "").strip()
    if not checkpoint_path:
        raise SystemExit(
            f"missing checkpoint for {spec.label}; expected a learned checkpoint for preset {spec.workload_preset!r}"
        )
    if checkpoint_path.startswith("/absolute/path/to/"):
        raise SystemExit(
            f"checkpoint for {spec.label} is still the literal placeholder path {checkpoint_path!r}; "
            "replace it with a real checkpoint path on disk"
        )
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(
            f"checkpoint for {spec.label} does not exist: {path}"
        )
    try:
        loaded = load_runtime_policy(str(path))
    except Exception as exc:
        raise SystemExit(
            f"checkpoint for {spec.label} could not be loaded: {exc}"
        ) from exc
    if loaded.workload_preset != spec.workload_preset:
        raise SystemExit(
            f"checkpoint for {spec.label} is for preset {loaded.workload_preset!r}, "
            f"but this arm expects {spec.workload_preset!r}: {path}"
        )


def _default_checkpoint_for_preset(repo_root: Path, workload_preset: str) -> str | None:
    preset = str(workload_preset or "").strip().lower()
    candidates: list[Path] = []
    if "elastic" in preset:
        candidates.extend(
            [
                repo_root / "test" / "results" / "checkpoints-rerun-20260419" / "admirl-elastic.pt",
                repo_root / "test" / "results" / "checkpoints" / "admirl-elastic.pt",
                repo_root / "test" / "results" / "kueue-checkpoints-sweep" / "eet-tuned-seed7.pt",
            ]
        )
    if "starvation" in preset or "cohort" in preset:
        candidates.extend(
            [
                repo_root / "test" / "results" / "checkpoints-rerun-20260419" / "admirl-cohort.pt",
                repo_root / "test" / "results" / "checkpoints" / "admirl-cohort.pt",
                repo_root / "test" / "results" / "kueue-checkpoints-sweep" / "cohort-tuned-seed7.pt",
            ]
        )
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    return None


def _arm_specs(repo_root: Path) -> list[ArmSpec]:
    upstream_local_available = DEFAULT_UPSTREAM_LOCAL_DIR.exists()
    return [
        ArmSpec(
            name="cohort-best-effort",
            label="cohort + BestEffortFIFO",
            workload_preset="kueue-lingjun-gang-starvation-cohort",
            arm="stock-best-effort-default",
            source_mode="local" if upstream_local_available else "git",
            source_dir=str(DEFAULT_UPSTREAM_LOCAL_DIR) if upstream_local_available else None,
            git_url=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_URL,
            git_ref=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_REF,
        ),
        ArmSpec(
            name="cohort-strict",
            label="cohort + StrictFIFO",
            workload_preset="kueue-lingjun-gang-starvation-cohort",
            arm="strict-default-sensitivity",
            source_mode="local" if upstream_local_available else "git",
            source_dir=str(DEFAULT_UPSTREAM_LOCAL_DIR) if upstream_local_available else None,
            git_url=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_URL,
            git_ref=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_REF,
        ),
        ArmSpec(
            name="cohort-learned",
            label="cohort + learned",
            workload_preset="kueue-lingjun-gang-starvation-cohort",
            arm="learned-best-effort-default",
            source_mode="local",
            source_dir=str(DEFAULT_LOCAL_KUEUE_DIR),
            checkpoint_path=_default_checkpoint_for_preset(repo_root, "kueue-lingjun-gang-starvation-cohort"),
        ),
        ArmSpec(
            name="elastic-best-effort",
            label="elastic + BestEffortFIFO",
            workload_preset="kueue-lingjun-gang-elastic-topology",
            arm="stock-best-effort-default",
            source_mode="local" if upstream_local_available else "git",
            source_dir=str(DEFAULT_UPSTREAM_LOCAL_DIR) if upstream_local_available else None,
            git_url=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_URL,
            git_ref=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_REF,
        ),
        ArmSpec(
            name="elastic-strict",
            label="elastic + StrictFIFO",
            workload_preset="kueue-lingjun-gang-elastic-topology",
            arm="strict-default-sensitivity",
            source_mode="local" if upstream_local_available else "git",
            source_dir=str(DEFAULT_UPSTREAM_LOCAL_DIR) if upstream_local_available else None,
            git_url=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_URL,
            git_ref=None if upstream_local_available else DEFAULT_UPSTREAM_KUEUE_REF,
        ),
        ArmSpec(
            name="elastic-learned",
            label="elastic + learned",
            workload_preset="kueue-lingjun-gang-elastic-topology",
            arm="learned-elastic-default",
            source_mode="local",
            source_dir=str(DEFAULT_LOCAL_KUEUE_DIR),
            checkpoint_path=_default_checkpoint_for_preset(repo_root, "kueue-lingjun-gang-elastic-topology"),
        ),
    ]


def _apply_checkpoint_overrides(specs: dict[str, ArmSpec], *, cohort_checkpoint: str, elastic_checkpoint: str) -> dict[str, ArmSpec]:
    updated = dict(specs)
    if cohort_checkpoint:
        updated["cohort-learned"] = ArmSpec(
            **{
                **updated["cohort-learned"].__dict__,
                "checkpoint_path": str(Path(cohort_checkpoint).expanduser().resolve()),
            }
        )
    if elastic_checkpoint:
        updated["elastic-learned"] = ArmSpec(
            **{
                **updated["elastic-learned"].__dict__,
                "checkpoint_path": str(Path(elastic_checkpoint).expanduser().resolve()),
            }
        )
    return updated


def _profile_env(spec: ArmSpec) -> dict[str, str]:
    env = dict(os.environ)
    env["ADMIRL_KUEUE_SOURCE_MODE"] = spec.source_mode
    if spec.source_dir:
        env["ADMIRL_KUEUE_SOURCE_DIR"] = spec.source_dir
    else:
        env.pop("ADMIRL_KUEUE_SOURCE_DIR", None)
    if spec.git_url:
        env["ADMIRL_KUEUE_GIT_URL"] = spec.git_url
    else:
        env.pop("ADMIRL_KUEUE_GIT_URL", None)
    if spec.git_ref:
        env["ADMIRL_KUEUE_GIT_REF"] = spec.git_ref
    else:
        env.pop("ADMIRL_KUEUE_GIT_REF", None)
    return env


def _matrix_command(
    *,
    spec: ArmSpec,
    seed: int,
    output_root: Path,
    num_jobs: int,
    arrival_span: float,
    trace_split: str,
    trace_train_fraction: float,
    runtime_scale: float,
    time_scale: float,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "test/kueue/run_live_kueue_matrix.py",
        "--workload-preset",
        spec.workload_preset,
        "--arm",
        spec.arm,
        "--seed",
        str(seed),
        "--num-jobs",
        str(num_jobs),
        "--arrival-span",
        str(arrival_span),
        "--trace-split",
        trace_split,
        "--trace-train-fraction",
        str(trace_train_fraction),
        "--runtime-scale",
        str(runtime_scale),
        "--time-scale",
        str(time_scale),
        "--output-root",
        str(output_root),
    ]
    if spec.checkpoint_path:
        cmd.extend(["--checkpoint", spec.checkpoint_path])
    if spec.runtime_policy:
        cmd.extend(["--runtime-policy-override", spec.runtime_policy])
    return cmd


def _run_arm_seed(
    *,
    spec: ArmSpec,
    seed: int,
    output_root: Path,
    num_jobs: int,
    arrival_span: float,
    trace_split: str,
    trace_train_fraction: float,
    runtime_scale: float,
    time_scale: float,
    timeout_seconds: float,
    force: bool,
) -> dict:
    seed_root = output_root / spec.name / f"seed-{seed}"
    summary_path = seed_root / "live-matrix-summary.json"
    if force or not summary_path.exists():
        cmd = _matrix_command(
            spec=spec,
            seed=seed,
            output_root=seed_root,
            num_jobs=num_jobs,
            arrival_span=arrival_span,
            trace_split=trace_split,
            trace_train_fraction=trace_train_fraction,
            runtime_scale=runtime_scale,
            time_scale=time_scale,
        )
        print(f"[run] {spec.name} seed={seed}")
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=_profile_env(spec),
            check=True,
            timeout=timeout_seconds,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_summary_path"] = str(summary_path)
    return summary


def _average(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _aggregate_arm_runs(spec: ArmSpec, runs: list[dict]) -> dict:
    metric_summary: dict[str, dict] = {}
    for metric in IMPORTANT_GANG_METRICS:
        values: list[float] = []
        for run in runs:
            arm_metrics = run["arms"].get(spec.arm, {})
            if metric in arm_metrics:
                values.append(float(arm_metrics[metric]))
        if not values:
            continue
        metric_summary[metric] = {
            "mean": _average(values),
            "min": min(values),
            "max": max(values),
            "runs": len(values),
            "values": values,
        }
    return {
        "label": spec.label,
        "arm": spec.arm,
        "preset": spec.workload_preset,
        "source_mode": spec.source_mode,
        "source_dir": spec.source_dir,
        "git_url": spec.git_url,
        "git_ref": spec.git_ref,
        "checkpoint_path": spec.checkpoint_path,
        "runtime_policy": spec.runtime_policy,
        "metrics": metric_summary,
        "summaries": [run["_summary_path"] for run in runs],
    }


def _aggregate_pairwise(
    *,
    baseline_name: str,
    baseline_spec: ArmSpec,
    baseline_runs: list[dict],
    candidate_name: str,
    candidate_spec: ArmSpec,
    candidate_runs: list[dict],
) -> dict:
    comparisons: dict[str, list[dict]] = {}
    for baseline_summary, candidate_summary in zip(baseline_runs, candidate_runs):
        comp = _build_arm_comparison(
            baseline_spec.arm,
            candidate_spec.arm,
            baseline_summary["arms"][baseline_spec.arm],
            candidate_summary["arms"][candidate_spec.arm],
        )["metrics"]
        for metric, info in comp.items():
            comparisons.setdefault(metric, []).append(info)
    aggregate = {}
    for metric, rows in comparisons.items():
        improvement_values = [float(row["improvement_fraction"]) for row in rows if row.get("improvement_fraction") is not None]
        aggregate[metric] = {
            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "baseline_mean": _average([float(row["baseline"]) for row in rows]),
            "candidate_mean": _average([float(row["candidate"]) for row in rows]),
            "improvement_fraction_mean": _average(improvement_values),
            "improvement_fraction_min": min(improvement_values) if improvement_values else 0.0,
            "improvement_fraction_max": max(improvement_values) if improvement_values else 0.0,
            "candidate_wins": sum(1 for row in rows if row.get("winner") == "candidate"),
            "runs": len(rows),
        }
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metrics": aggregate,
    }


def _build_markdown_report(summary: dict) -> str:
    lines = [
        "# Tuned 5-Seed Kueue Suite",
        "",
        "## Arm Means",
        "",
    ]
    for arm_name, arm_info in summary["arms"].items():
        lines.append(f"### {arm_info['label']}")
        lines.append("")
        for metric in [
            "head_gang_blocked_seconds",
            "avg_gang_wait_seconds",
            "avg_gang_completion_seconds",
            "avg_elastic_wait_seconds",
            "avg_elastic_completion_seconds",
            "p95_gang_wait_seconds",
            "makespan_seconds",
            "throughput_jobs_per_minute",
            "small_job_bypass_count_while_gang_pending",
            "avg_small_wait_seconds",
        ]:
            info = arm_info["metrics"].get(metric)
            if not info:
                continue
            lines.append(f"- `{metric}`: mean `{info['mean']:.3f}`")
        lines.append("")
    lines.append("## Pairwise Learned Comparisons")
    lines.append("")
    for key, comp in summary["comparisons"].items():
        lines.append(f"### {key}")
        lines.append("")
        for metric in [
            "head_gang_blocked_seconds",
            "avg_gang_wait_seconds",
            "avg_gang_completion_seconds",
            "avg_elastic_wait_seconds",
            "avg_elastic_completion_seconds",
            "p95_gang_wait_seconds",
            "makespan_seconds",
            "throughput_jobs_per_minute",
            "small_job_bypass_count_while_gang_pending",
            "avg_small_wait_seconds",
        ]:
            info = comp["metrics"].get(metric)
            if not info:
                continue
            lines.append(
                f"- `{metric}`: `{info['baseline_mean']:.3f} -> {info['candidate_mean']:.3f}`, "
                f"`{info['improvement_fraction_mean'] * 100.0:.1f}%` better, `{info['candidate_wins']}/{info['runs']}` wins"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tuned 5-seed Kueue suite with direct matrix calls")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-jobs-cohort", type=int, default=8)
    parser.add_argument("--num-jobs-elastic", type=int, default=8)
    parser.add_argument("--arrival-span-cohort", type=float, default=120.0)
    parser.add_argument("--arrival-span-elastic", type=float, default=120.0)
    parser.add_argument("--trace-split", default="test")
    parser.add_argument("--trace-train-fraction", type=float, default=0.75)
    parser.add_argument("--runtime-scale", type=float, default=120.0)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--arm-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--cohort-checkpoint", default="", help="Optional explicit checkpoint for cohort learned runs")
    parser.add_argument("--elastic-checkpoint", default="", help="Optional explicit checkpoint for elastic learned runs")
    args = parser.parse_args()

    specs = {spec.name: spec for spec in _arm_specs(REPO_ROOT)}
    specs = _apply_checkpoint_overrides(
        specs,
        cohort_checkpoint=args.cohort_checkpoint,
        elastic_checkpoint=args.elastic_checkpoint,
    )
    missing = [
        spec
        for spec in specs.values()
        if _requires_checkpoint(spec) and not spec.checkpoint_path
    ]
    if missing:
        details = "\n".join(
            f"- {spec.label}: expected a local checkpoint for preset {spec.workload_preset!r}"
            for spec in missing
        )
        raise SystemExit(
            "missing learned checkpoints for the tuned suite:\n"
            f"{details}\n"
            "Train and save the missing checkpoints first, for example under "
            "`test/results/checkpoints/admirl-cohort.pt` and "
            "`test/results/checkpoints/admirl-elastic.pt`."
        )
    for spec in specs.values():
        _validate_checkpoint_for_spec(spec)

    ensure_model_server()
    args.output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    runs: dict[str, list[dict]] = {name: [] for name in specs}

    for seed in seeds:
        for spec in specs.values():
            is_cohort = "cohort" in spec.workload_preset
            runs[spec.name].append(
                _run_arm_seed(
                    spec=spec,
                    seed=seed,
                    output_root=args.output_root,
                    num_jobs=args.num_jobs_cohort if is_cohort else args.num_jobs_elastic,
                    arrival_span=args.arrival_span_cohort if is_cohort else args.arrival_span_elastic,
                    trace_split=args.trace_split,
                    trace_train_fraction=args.trace_train_fraction,
                    runtime_scale=args.runtime_scale,
                    time_scale=args.time_scale,
                    timeout_seconds=args.arm_timeout_seconds,
                    force=args.force,
                )
            )

    arms_summary = {
        name: _aggregate_arm_runs(specs[name], arm_runs)
        for name, arm_runs in runs.items()
    }
    comparisons = {
        "cohort-best-effort-vs-learned": _aggregate_pairwise(
            baseline_name="cohort + BestEffortFIFO",
            baseline_spec=specs["cohort-best-effort"],
            baseline_runs=runs["cohort-best-effort"],
            candidate_name="cohort + learned",
            candidate_spec=specs["cohort-learned"],
            candidate_runs=runs["cohort-learned"],
        ),
        "cohort-strict-vs-learned": _aggregate_pairwise(
            baseline_name="cohort + StrictFIFO",
            baseline_spec=specs["cohort-strict"],
            baseline_runs=runs["cohort-strict"],
            candidate_name="cohort + learned",
            candidate_spec=specs["cohort-learned"],
            candidate_runs=runs["cohort-learned"],
        ),
        "elastic-best-effort-vs-learned": _aggregate_pairwise(
            baseline_name="elastic + BestEffortFIFO",
            baseline_spec=specs["elastic-best-effort"],
            baseline_runs=runs["elastic-best-effort"],
            candidate_name="elastic + learned",
            candidate_spec=specs["elastic-learned"],
            candidate_runs=runs["elastic-learned"],
        ),
        "elastic-strict-vs-learned": _aggregate_pairwise(
            baseline_name="elastic + StrictFIFO",
            baseline_spec=specs["elastic-strict"],
            baseline_runs=runs["elastic-strict"],
            candidate_name="elastic + learned",
            candidate_spec=specs["elastic-learned"],
            candidate_runs=runs["elastic-learned"],
        ),
    }

    summary = {
        "config": {
            "seeds": seeds,
            "num_jobs_cohort": args.num_jobs_cohort,
            "num_jobs_elastic": args.num_jobs_elastic,
            "arrival_span_cohort": args.arrival_span_cohort,
            "arrival_span_elastic": args.arrival_span_elastic,
            "trace_split": args.trace_split,
            "trace_train_fraction": args.trace_train_fraction,
            "runtime_scale": args.runtime_scale,
            "time_scale": args.time_scale,
            "arm_timeout_seconds": args.arm_timeout_seconds,
        },
        "arms": arms_summary,
        "comparisons": comparisons,
    }
    summary_path = args.output_root / "suite-summary.json"
    report_path = args.output_root / "suite-report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(_build_markdown_report(summary), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
