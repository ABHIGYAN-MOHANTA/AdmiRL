from flask import jsonify, request

from .state import state


def register_benchmark_routes(app):
    @app.route("/api/benchmark/status", methods=["GET"])
    def benchmark_status():
        window = int(request.args.get("window", "60") or 60)
        with state.lock:
            return jsonify(state.benchmark_status_snapshot_locked(window=window))

    @app.route("/api/benchmark/progress", methods=["POST"])
    def benchmark_progress():
        payload = request.get_json(force=True) or {}
        with state.lock:
            state.record_benchmark_snapshot_locked(payload)
            snapshot = state.benchmark_status_snapshot_locked()
        return jsonify({
            "status": "ok",
            "benchmark": snapshot,
        })

    @app.route("/api/benchmark/reset", methods=["POST"])
    def benchmark_reset():
        with state.lock:
            state.reset_benchmark_metrics_locked()
        return jsonify({"status": "ok"})
