from flask import jsonify, render_template

from .state import effective_runtime_policy_name, state


def register_dashboard_routes(app):
    @app.route("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/api/last-decision", methods=["GET"])
    def get_last_decision():
        with state.lock:
            return jsonify(state.last_decision)

    @app.route("/api/runtime-status", methods=["GET"])
    def runtime_status():
        with state.lock:
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
            return jsonify({
                "runtime_policy": state.runtime_policy,
                "effective_policy": effective_policy,
                "learned_checkpoint_path": state.learned_checkpoint_path,
                "last_decision_timestamp": state.last_decision.get("timestamp"),
                "last_decision_source": last_source,
                "runtime_metrics": state.runtime_metrics_snapshot_locked(),
            })
