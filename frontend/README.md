# Static Frontend (No Jinja)

This folder contains a client-side component approach for static hosting.

## What is included

- `components/navbar.html`: shared top navbar component (single source of truth)
- `js/component-loader.js`: mini client-side template loader
- `pages/*.html`: static exported tool pages (no Jinja tags)

## Usage

Each page uses placeholders and loads components with JavaScript:

```html
<div id="navbar"></div>
<script src="/frontend/js/component-loader.js"></script>
```

Localized static routes are served without duplicating page files:

- `/it/`
- `/en/`
- `/it/<slug>.html`
- `/en/<slug>.html`

## Backend role

Backend should be used for API endpoints (`/api/*`) and static file delivery only.
No Jinja template rendering is required for frontend pages.
