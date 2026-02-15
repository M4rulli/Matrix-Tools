# Matrix Tools

Refactor repository layout:

```text
Matrix-Tools/
├── src/                    # Backend/API source (Flask routes)
│   ├── server.py
│   ├── requirements.txt
│   └── routes/
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
