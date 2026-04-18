from flask import Flask

from .benchmark_routes import register_benchmark_routes
from .dashboard_routes import register_dashboard_routes
from .decision_routes import register_decision_routes
from .policy_routes import register_policy_routes


def create_app():
    app = Flask(__name__)
    register_dashboard_routes(app)
    register_benchmark_routes(app)
    register_decision_routes(app)
    register_policy_routes(app)
    return app
