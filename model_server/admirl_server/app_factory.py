from pathlib import Path

from flask import Flask

from .dashboard_routes import register_dashboard_routes
from .decision_routes import register_decision_routes
from .policy_routes import register_policy_routes


def create_app():
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    register_dashboard_routes(app)
    register_decision_routes(app)
    register_policy_routes(app)
    return app
