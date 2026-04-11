from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
    MODEL_SERVER_URL,
)
from test.kueue.run_live_kueue_matrix import stop_process


DEFAULT_UPSTREAM_KUEUE_URL = "https://github.com/kubernetes-sigs/kueue.git"
DEFAULT_UPSTREAM_KUEUE_REF = "v0.15.0"
DEFAULT_LOCAL_KUEUE_DIR = Path.home() / "Desktop" / "kueue"


def _maybe_start_model_server() -> tuple[subprocess.Popen[str] | None, object | None]:
    try:
        ensure_model_server()
        return None, None
    except RuntimeError:
        pass

    log_path = REPO_ROOT / "test" / "results" / "kueue-live-source-compare-model-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    env = dict(os.environ)
    env["ADMIRL_MAX_NODES"] = env.get("ADMIRL_MAX_NODES", "64")
    env["ADMIRL_MODEL_SERVER_PORT"] = MODEL_SERVER_URL.rsplit(":", 1)[-1]
    proc = subprocess.Popen(
        [sys.executable, "model_server/app.py"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if proc.poll() is not None:
            stop_process(proc, log_file)
            raise SystemExit("model server exited before becoming ready")
        try:
            ensure_model_server()
            return proc, log_file
        except RuntimeError:
            time.sleep(1.0)
    stop_process(proc, log_file)
    raise SystemExit("timed out waiting for model server to become ready")


@dataclass(frozen=True)
class SourceProfile:
    label: str
    arm: str
    source_mode: str
    source_dir: str | None
    git_url: str | None
    git_ref: str | None
    checkpoint_path: str | None
    runtime_policy: str | None = None


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("no seeds selected")
    return seeds


def _average(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _profile_env(profile: SourceProfile) -> dict[str, str]:
    env = dict(os.environ)
    env["ADMIRL_KUEUE_SOURCE_MODE"] = profile.source_mode
    if profile.source_dir:
        env["ADMIRL_KUEUE_SOURCE_DIR"] = profile.source_dir
    else:
        env.pop("ADMIRL_KUEUE_SOURCE_DIR", None)
    if profile.git_url:
        env["ADMIRL_KUEUE_GIT_URL"] = profile.git_url
    else:
        env.pop("ADMIRL_KUEUE_GIT_URL", None)
    if profile.git_ref:
        env["ADMIRL_KUEUE_GIT_REF"] = profile.git_ref
    else:
        env.pop("ADMIRL_KUEUE_GIT_REF", None)
    return env


def _matrix_command(
    *,
    profile: SourceProfile,
    workload_preset: str,
    seed: int,
    num_jobs: int,
    arrival_span: float,
    trace_split: str,
    trace_train_fraction: float,
    runtime_scale: float,
    time_scale: float,
    output_root: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "test/kueue/run_live_kueue_matrix.py",
        "--workload-preset",
        workload_preset,
        "--arm",
        profile.arm,
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
    if profile.checkpoint_path:
        cmd.extend(["--checkpoint", profile.checkpoint_path])
    if profile.runtime_policy:
        cmd.extend(["--runtime-policy-override", profile.runtime_policy])
    return cmd


def _run_matrix_profile(
    *,
    profile: SourceProfile,
    workload_preset: str,
    seed: int,
    num_jobs: int,
    arrival_span: float,
    trace_split: str,
    trace_train_fraction: float,
    runtime_scale: float,
    time_scale: float,
    output_root: Path,
    force: bool,
) -> dict:
    summary_path = output_root / "live-matrix-summary.json"
    if force or not summary_path.exists():
        subprocess.run(
            _matrix_command(
                profile=profile,
                workload_preset=workload_preset,
                seed=seed,
                num_jobs=num_jobs,
                arrival_span=arrival_span,
                trace_split=trace_split,
                trace_train_fraction=trace_train_fraction,
                runtime_scale=runtime_scale,
                time_scale=time_scale,
                output_root=output_root,
            ),
            cwd=str(REPO_ROOT),
            env=_profile_env(profile),
            check=True,
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _run_comparison_record(
    *,
    workload_preset: str,
    seed: int,
    baseline_profile: SourceProfile,
    candidate_profile: SourceProfile,
    baseline_summary: dict,
    candidate_summary: dict,
) -> dict:
    baseline_arm = baseline_summary["arms"][baseline_profile.arm]
    candidate_arm = candidate_summary["arms"][candidate_profile.arm]
    return {
        "workload_preset": workload_preset,
        "seed": seed,
        "baseline_profile": baseline_profile.label,
        "candidate_profile": candidate_profile.label,
        "baseline_summary_path": str(baseline_summary["_summary_path"]),
        "candidate_summary_path": str(candidate_summary["_summary_path"]),
        "baseline_kueue_source": baseline_summary.get("kueue_source", {}),
        "candidate_kueue_source": candidate_summary.get("kueue_source", {}),
        "baseline_runtime_policy": baseline_summary["arms"][baseline_profile.arm].get("runtime_policy", ""),
        "candidate_runtime_policy": candidate_summary["arms"][candidate_profile.arm].get("runtime_policy", ""),
        "metrics": _build_arm_comparison(
            baseline_profile.arm,
            candidate_profile.arm,
            baseline_arm,
            candidate_arm,
        )["metrics"],
    }


def aggregate_comparison_runs(runs: list[dict]) -> dict:
    metric_summary = {}
    for metric in IMPORTANT_GANG_METRICS:
        baseline_values = []
        candidate_values = []
        improvements = []
        wins = 0
        for run in runs:
            info = run["metrics"].get(metric)
            if not info:
                continue
            baseline_values.append(float(info.get("baseline", 0.0)))
            candidate_values.append(float(info.get("candidate", 0.0)))
            if info.get("improvement_fraction") is not None:
                improvements.append(float(info["improvement_fraction"]))
            if info.get("winner") == "candidate":
                wins += 1
        if not baseline_values or not candidate_values:
            continue
        metric_summary[metric] = {
            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "baseline_mean": _average(baseline_values),
            "candidate_mean": _average(candidate_values),
            "improvement_fraction_mean": _average(improvements),
            "improvement_fraction_min": min(improvements) if improvements else 0.0,
            "improvement_fraction_max": max(improvements) if improvements else 0.0,
            "candidate_wins": wins,
            "runs": len(baseline_values),
        }
    return {
        "run_count": len(runs),
        "metrics": metric_summary,
        "runs": runs,
    }


def build_markdown_report(*, config: dict, aggregate: dict) -> str:
    lines = [
        "# Kueue Live Comparison Suite",
        "",
        "## Configuration",
        "",
        f"- Preset: {config['preset']}",
        f"- Seeds: {', '.join(str(seed) for seed in config['seeds'])}",
        f"- Baseline arm: {config['baseline_profile']['arm']}",
        f"- Candidate arm: {config['candidate_profile']['arm']}",
        f"- Baseline runtime policy: {config['baseline_profile'].get('runtime_policy') or 'arm default'}",
        f"- Candidate runtime policy: {config['candidate_profile'].get('runtime_policy') or 'arm default'}",
        f"- Baseline source: {json.dumps(config['baseline_profile']['source'], sort_keys=True)}",
        f"- Candidate source: {json.dumps(config['candidate_profile']['source'], sort_keys=True)}",
        "",
        "## Aggregate",
        "",
        "| Metric | Baseline Mean | Candidate Mean | Mean Improvement | Wins |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, data in aggregate["metrics"].items():
        lines.append(
            f"| `{metric}` | {data['baseline_mean']:.3f} | {data['candidate_mean']:.3f} | "
            f"{data['improvement_fraction_mean'] * 100.0:.1f}% | {data['candidate_wins']}/{data['runs']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare upstream Kueue baseline against our admission-augmented Kueue across live seeds")
    parser.add_argument("--workload-preset", default="kueue-lingjun-gang-starvation-cohort")
    parser.add_argument("--seeds", default="7,11,13,17,23")
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--arrival-span", type=float, default=120.0)
    parser.add_argument("--trace-split", default="test")
    parser.add_argument("--trace-train-fraction", type=float, default=0.75)
    parser.add_argument("--runtime-scale", type=float, default=120.0)
    parser.add_argument("--time-scale", type=float, default=10.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--baseline-label", default="upstream-default")
    parser.add_argument("--baseline-arm", default="stock-best-effort-default")
    parser.add_argument("--baseline-source-mode", default="git")
    parser.add_argument("--baseline-source-dir", default="")
    parser.add_argument("--baseline-git-url", default=DEFAULT_UPSTREAM_KUEUE_URL)
    parser.add_argument("--baseline-git-ref", default=DEFAULT_UPSTREAM_KUEUE_REF)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--baseline-runtime-policy", default="")

    parser.add_argument("--candidate-label", default="ours-admission-rl")
    parser.add_argument("--candidate-arm", default="learned-best-effort-default")
    parser.add_argument("--candidate-source-mode", default="local")
    parser.add_argument("--candidate-source-dir", default=str(DEFAULT_LOCAL_KUEUE_DIR))
    parser.add_argument("--candidate-git-url", default="")
    parser.add_argument("--candidate-git-ref", default="")
    parser.add_argument("--candidate-checkpoint", default="")
    parser.add_argument("--candidate-runtime-policy", default="")
    args = parser.parse_args()

    baseline_profile = SourceProfile(
        label=args.baseline_label,
        arm=args.baseline_arm,
        source_mode=args.baseline_source_mode,
        source_dir=args.baseline_source_dir or None,
        git_url=args.baseline_git_url or None,
        git_ref=args.baseline_git_ref or None,
        checkpoint_path=args.baseline_checkpoint or None,
        runtime_policy=args.baseline_runtime_policy or None,
    )
    candidate_profile = SourceProfile(
        label=args.candidate_label,
        arm=args.candidate_arm,
        source_mode=args.candidate_source_mode,
        source_dir=args.candidate_source_dir or None,
        git_url=args.candidate_git_url or None,
        git_ref=args.candidate_git_ref or None,
        checkpoint_path=args.candidate_checkpoint or None,
        runtime_policy=args.candidate_runtime_policy or None,
    )

    seeds = _parse_seeds(args.seeds)
    model_server_proc, model_server_log = _maybe_start_model_server()
    try:
        runs = []
        for seed in seeds:
            baseline_root = args.output_root / baseline_profile.label / f"seed-{seed}"
            candidate_root = args.output_root / candidate_profile.label / f"seed-{seed}"
            baseline_summary = _run_matrix_profile(
                profile=baseline_profile,
                workload_preset=args.workload_preset,
                seed=seed,
                num_jobs=args.num_jobs,
                arrival_span=args.arrival_span,
                trace_split=args.trace_split,
                trace_train_fraction=args.trace_train_fraction,
                runtime_scale=args.runtime_scale,
                time_scale=args.time_scale,
                output_root=baseline_root,
                force=args.force,
            )
            candidate_summary = _run_matrix_profile(
                profile=candidate_profile,
                workload_preset=args.workload_preset,
                seed=seed,
                num_jobs=args.num_jobs,
                arrival_span=args.arrival_span,
                trace_split=args.trace_split,
                trace_train_fraction=args.trace_train_fraction,
                runtime_scale=args.runtime_scale,
                time_scale=args.time_scale,
                output_root=candidate_root,
                force=args.force,
            )
            baseline_summary["_summary_path"] = baseline_root / "live-matrix-summary.json"
            candidate_summary["_summary_path"] = candidate_root / "live-matrix-summary.json"
            runs.append(
                _run_comparison_record(
                    workload_preset=args.workload_preset,
                    seed=seed,
                    baseline_profile=baseline_profile,
                    candidate_profile=candidate_profile,
                    baseline_summary=baseline_summary,
                    candidate_summary=candidate_summary,
                )
            )
    finally:
        stop_process(model_server_proc, model_server_log)

    aggregate = aggregate_comparison_runs(runs)
    config = {
        "preset": args.workload_preset,
        "seeds": seeds,
        "num_jobs": args.num_jobs,
        "arrival_span": args.arrival_span,
        "trace_split": args.trace_split,
        "trace_train_fraction": args.trace_train_fraction,
        "runtime_scale": args.runtime_scale,
        "time_scale": args.time_scale,
        "baseline_profile": {
            "label": baseline_profile.label,
            "arm": baseline_profile.arm,
            "source": {
                "mode": baseline_profile.source_mode,
                "dir": baseline_profile.source_dir,
                "git_url": baseline_profile.git_url,
                "git_ref": baseline_profile.git_ref,
            },
            "runtime_policy": baseline_profile.runtime_policy,
        },
        "candidate_profile": {
            "label": candidate_profile.label,
            "arm": candidate_profile.arm,
            "source": {
                "mode": candidate_profile.source_mode,
                "dir": candidate_profile.source_dir,
                "git_url": candidate_profile.git_url,
                "git_ref": candidate_profile.git_ref,
            },
            "runtime_policy": candidate_profile.runtime_policy,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "source-compare-summary.json"
    summary_path.write_text(json.dumps({"config": config, "aggregate": aggregate}, indent=2), encoding="utf-8")
    report_path = args.output_root / "source-compare-report.md"
    report_path.write_text(build_markdown_report(config=config, aggregate=aggregate), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
