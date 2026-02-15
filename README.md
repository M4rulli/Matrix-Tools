# Matrix Tools

Refactor repository layout:

```text
Matrix-Tools/
├── src/                    # Backend/API source (Flask routes, templates, scripts)
│   ├── server.py
│   ├── requirements.txt
│   ├── routes/
│   ├── scripts/
│   └── templates/
├── docs/                   # Static frontend (GitHub Pages ready)
│   ├── index.html
│   ├── css/
│   ├── js/
│   ├── assets/
│   ├── components/
│   └── pages/
└── README.md
```

## Run backend (API + static serving)

```bash
python3 src/server.py
```

Default local URL: `http://127.0.0.1:10000`

## Install backend dependencies

```bash
pip install -r src/requirements.txt
```

## Regenerate static pages

```bash
python3 src/scripts/export_static_pages.py
```
