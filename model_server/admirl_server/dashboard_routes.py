import os

from flask import Response, jsonify, redirect

from .state import state


DEFAULT_GRAFANA_URL = "http://127.0.0.1:3000/d/admirl-runtime/admirl-runtime-and-benchmark-operations?kiosk"


def grafana_dashboard_url() -> str:
    return os.getenv("ADMIRL_GRAFANA_URL", DEFAULT_GRAFANA_URL).strip() or DEFAULT_GRAFANA_URL


def register_dashboard_routes(app):
    @app.route("/")
    @app.route("/dashboard")
    def dashboard_redirect():
        return redirect(grafana_dashboard_url(), code=302)

    @app.route("/api/last-decision", methods=["GET"])
    def get_last_decision():
        with state.lock:
            return jsonify(state.last_decision)

    @app.route("/metrics", methods=["GET"])
    def prometheus_metrics():
        with state.lock:
            payload = state.prometheus_metrics_locked()
        return Response(payload, mimetype="text/plain; version=0.0.4; charset=utf-8")
