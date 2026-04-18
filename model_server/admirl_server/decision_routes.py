import time
from datetime import datetime

from flask import jsonify, request

from .kueue_advisor import build_kueue_admission_advice
from .state import runtime_policy_uses_checkpoint, state


def _store_last_decision(request_state: dict, response: dict) -> str:
    timestamp = datetime.now().isoformat()
    with state.lock:
        state.last_decision["timestamp"] = timestamp
        state.last_decision["request"] = request_state
        state.last_decision["response"] = response
    return timestamp


def _build_admission_advice(request_state: dict) -> dict:
    policy = (
        state.learned_checkpoint
        if runtime_policy_uses_checkpoint(state.runtime_policy) and state.learned_checkpoint is not None
        else None
    )
    return build_kueue_admission_advice(
        request_state=request_state,
        policy=policy,
        policy_mode=state.runtime_policy,
    )


def _handle_admission_advice_request():
    start = time.perf_counter()
    request_state = request.get_json(force=True)
    response = _build_admission_advice(request_state)
    timestamp = _store_last_decision(request_state, response)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    with state.lock:
        state.record_request_latency_locked("kueue_admission_advice", elapsed_ms)
        state.record_decision_event_locked(
            route_name="kueue_admission_advice",
            timestamp=timestamp,
            request_state=request_state,
            response=response,
            elapsed_ms=elapsed_ms,
        )
    return jsonify(response)


def register_decision_routes(app):
    @app.route("/api/kueue/admission-advice", methods=["POST"])
    def kueue_admission_advice():
        return _handle_admission_advice_request()
