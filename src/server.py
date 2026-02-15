from __future__ import annotations

import os

from flask import Flask
from flask_cors import CORS
from routes import register_routes

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://m4rulli.github.io",
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:3000",
            ]
        },
        r"/health": {
            "origins": [
                "https://m4rulli.github.io",
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:3000",
            ]
        },
    },
)
register_routes(app)


@app.route("/")
def root():
    return {
        "service": "matrix-tools-api",
        "status": "ok",
        "message": "API server is running. Use /health and /api/* endpoints.",
    }, 200


@app.route("/health")
def health():
    return {"status": "ok", "mode": "api-only"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
