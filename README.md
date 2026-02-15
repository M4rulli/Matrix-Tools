# Matrix Tools

[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![SymPy](https://img.shields.io/badge/SymPy-1.13.3-3B5526?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![Frontend](https://img.shields.io/badge/Frontend-Static%20(HTML%2FCSS%2FJS)-0A7EA4)](#architecture)

Matrix Tools is a bilingual (Italian/English) mathematical web application for advanced computation workflows.
It provides a static frontend with interactive math input and a Python API backend that solves symbolic and algorithmic problems step-by-step.

## What the Web App Does

Matrix Tools helps users:

- Solve advanced math/control/optimization problems through guided modules.
- Enter expressions with MathLive and render high-quality formulas via MathJax.
- View structured, step-by-step outputs (including LaTeX-friendly content).
- Save exercises in a notebook workflow and reuse computed results.

### Available Domains

- Control Theory
- Algebra and Logic
- Linear Algebra
- Operations Research
- Mathematical Analysis

## Architecture

```text
Matrix-Tools/
├── src/
│   ├── server.py                 # Flask entrypoint (API only)
│   ├── requirements.txt
│   └── routes/                   # API blueprints per domain/module
├── docs/                         # Static frontend (GitHub Pages-ready)
│   ├── index.html
│   ├── pages/                    # Prebuilt module pages
│   ├── components/               # Shared UI components (navbar/sidebar/footer/modal)
│   ├── css/
│   ├── js/
│   └── assets/
├── LICENSE
└── README.md
```

## Technology Stack

- Backend: Flask, Werkzeug, SymPy
- Frontend: JavaScript, HTML, CSS, Bootstrap, Font Awesome
- Math UX: MathLive, MathJax
- Deployment model: static frontend + API backend

## Quick Start

### 1. Install dependencies

```bash
pip install -r src/requirements.txt
```

### 2. Run the server

```bash
python3 src/server.py
```

Default local URL: `http://127.0.0.1:10000`

## License

This project is licensed under the **GNU GPL-3.0-only** license.
See [LICENSE](LICENSE) for details.
