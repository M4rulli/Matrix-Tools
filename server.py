from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, redirect, request, send_from_directory
from routes import register_routes
from werkzeug.routing import BaseConverter

SUPPORTED_LANGUAGES = {"it", "en"}

# Legacy Italian page slugs still accepted and redirected to canonical English slugs.
LEGACY_SLUGS = {
    "linearizzazione.html": "linearization.html",
    "decomposizione-spettrale.html": "spectral-decomposition.html",
    "equazioni-differenziali.html": "differential-equations.html",
    "equazioni-differenze.html": "difference-equations.html",
    "sistemi-dinamici.html": "dynamical-systems.html",
    "potenze-modulari.html": "modular-powers.html",
    "diagramma-hasse.html": "hasse-diagram.html",
    "polinomi-booleani.html": "boolean-polynomials.html",
    "teorema-cinese-resto.html": "chinese-remainder-theorem.html",
    "identita-bezout.html": "bezout-identity.html",
    "determinante-laplace.html": "laplace-determinant.html",
    "autovalori-autovettori.html": "eigenvalues-eigenvectors.html",
    "sistemi-lineari.html": "linear-systems.html",
    "simplesso.html": "simplex.html",
    "condizioni-complementari.html": "complementary-conditions.html",
    "studio-funzione.html": "function-study.html",
    "integrali.html": "integrals.html",
}

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_PAGES_DIR = FRONTEND_DIR / "pages"


def build_static_pages_set() -> set[str]:
    if not FRONTEND_PAGES_DIR.exists():
        return set()
    return {
        path.name
        for path in FRONTEND_PAGES_DIR.iterdir()
        if path.is_file() and path.suffix == ".html"
    }


STATIC_PAGES = build_static_pages_set()

app = Flask(__name__)
register_routes(app)


class LangConverter(BaseConverter):
    regex = "it|en"


app.url_map.converters["lang"] = LangConverter


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def detect_preferred_language() -> str:
    accept_language = request.headers.get("Accept-Language", "")
    for token in accept_language.split(","):
        lang_token = token.split(";")[0].strip().lower()
        primary = lang_token.split("-")[0]
        if primary in SUPPORTED_LANGUAGES:
            return primary
    return "en"


def canonical_slug(slug: str) -> str:
    return LEGACY_SLUGS.get(slug, slug)


@app.route("/")
def root():
    return redirect(f"/{detect_preferred_language()}/", code=302)


@app.route("/it")
def localized_root_no_slash_it():
    return redirect("/it/", code=308)


@app.route("/en")
def localized_root_no_slash_en():
    return redirect("/en/", code=308)


@app.route("/<lang:lang>/")
def localized_index(lang: str):
    if lang not in SUPPORTED_LANGUAGES:
        abort(404)
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<lang:lang>/<path:slug>")
def localized_page(lang: str, slug: str):
    if lang not in SUPPORTED_LANGUAGES:
        abort(404)

    canonical = canonical_slug(slug)
    if canonical != slug:
        return redirect(f"/{lang}/{canonical}", code=308)

    if canonical in STATIC_PAGES:
        return send_from_directory(FRONTEND_PAGES_DIR, canonical)

    abort(404)


@app.route("/<path:slug>")
def non_prefixed_pages(slug: str):
    canonical = canonical_slug(slug)
    if canonical in STATIC_PAGES:
        return redirect(f"/{detect_preferred_language()}/{canonical}", code=308)
    abort(404)


@app.route("/frontend/<path:filepath>")
def frontend_assets(filepath: str):
    return send_from_directory(FRONTEND_DIR, filepath)


@app.route("/health")
def health():
    return {"status": "ok", "mode": "api+static"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
