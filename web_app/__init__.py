"""
web_app/__init__.py
-------------------
Flask application factory.
"""

import os
from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

    from web_app.routes import register_routes
    register_routes(app)

    @app.context_processor
    def inject_globals():
        from web_app.routes import _load_model_results
        data = _load_model_results()
        return {
            "trained":     data["available"],
            "best_model":  data["best_model"],
            "model_results": data["models"],   # available in all templates for topbar/sidebar
        }

    return app
