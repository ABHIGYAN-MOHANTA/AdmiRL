from flask import jsonify, request

from .kueue_runtime import load_runtime_policy
from .state import VALID_RUNTIME_POLICIES, effective_runtime_policy_name, runtime_policy_uses_checkpoint, state


def _status_payload_locked() -> dict:
    last_response = state.last_decision.get("response") or {}
    last_source = (
        last_response.get("advisor_source")
        or last_response.get("source")
        or ""
    )
    effective_policy = effective_runtime_policy_name(
        state.runtime_policy,
        state.learned_checkpoint is not None,
    )
    return {
        "runtime_policy": state.runtime_policy,
        "effective_policy": effective_policy,
        "learned_checkpoint_path": state.learned_checkpoint_path,
        "last_decision_timestamp": state.last_decision.get("timestamp"),
        "last_decision_source": last_source,
        "runtime_metrics": state.runtime_metrics_snapshot_locked(),
    }


def register_policy_routes(app):
    @app.route("/api/policy/status", methods=["GET"])
    def policy_status():
        with state.lock:
            return jsonify(_status_payload_locked())

    @app.route("/api/policy/runtime-policy", methods=["GET", "POST"])
    def runtime_policy():
        if request.method == "GET":
            with state.lock:
                return jsonify({
                    "runtime_policy": state.runtime_policy,
                    "valid_policies": sorted(VALID_RUNTIME_POLICIES),
                    "learned_checkpoint_path": state.learned_checkpoint_path,
                })

        data = request.get_json(force=True) or {}
        policy_name = str(data.get("policy", "")).strip().lower()
        if policy_name not in VALID_RUNTIME_POLICIES:
            return jsonify({
                "error": f"invalid runtime policy {policy_name!r}",
                "valid_policies": sorted(VALID_RUNTIME_POLICIES),
            }), 400
        if runtime_policy_uses_checkpoint(policy_name):
            with state.lock:
                if state.learned_checkpoint is None:
                    return jsonify({
                        "error": "no learned checkpoint loaded",
                    }), 400
                state.runtime_policy = policy_name
        else:
            with state.lock:
                state.runtime_policy = policy_name

        return jsonify({
            "status": "ok",
            "runtime_policy": state.runtime_policy,
        })

    @app.route("/api/policy/load-checkpoint", methods=["POST"])
    def load_checkpoint():
        data = request.get_json(force=True) or {}
        checkpoint_path = str(data.get("path", "")).strip()
        if not checkpoint_path:
            return jsonify({"error": "missing checkpoint path"}), 400

        try:
            loaded = load_runtime_policy(checkpoint_path)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        with state.lock:
            state.learned_checkpoint = loaded
            state.learned_checkpoint_path = loaded.checkpoint_path
            state.runtime_policy = "learned_multi_objective"

        return jsonify({
            "status": "ok",
            "runtime_policy": "learned_multi_objective",
            "checkpoint_path": loaded.checkpoint_path,
            "workload_preset": loaded.workload_preset,
            "cluster_layout": loaded.cluster_layout,
            "num_jobs": loaded.num_jobs,
        })

    @app.route("/api/policy/reset", methods=["POST"])
    def reset_policy():
        with state.lock:
            state.runtime_policy = "blocked_guard"
            state.learned_checkpoint = None
            state.learned_checkpoint_path = ""
        return jsonify({
            "status": "ok",
            "runtime_policy": state.runtime_policy,
        })

    @app.route("/api/runtime-metrics/reset", methods=["POST"])
    def reset_runtime_metrics():
        with state.lock:
            state.reset_runtime_metrics_locked()
        return jsonify({"status": "ok"})

    @app.route("/health")
    def health():
        with state.lock:
            payload = _status_payload_locked()
        payload["status"] = "ok"
        return jsonify(payload)
