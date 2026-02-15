#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw
import json

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT.parent / "docs" / "assets" / "icons"
ICONS.mkdir(parents=True, exist_ok=True)

BASE_SIZE = 1024

# Palette chosen for high contrast in tiny sizes.
BG = (16, 118, 110, 255)
BG2 = (14, 116, 144, 255)
FG = (248, 250, 252, 255)
ACC = (34, 211, 238, 255)

img = Image.new("RGBA", (BASE_SIZE, BASE_SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

pad = int(BASE_SIZE * 0.08)
# Rounded tile background
d.rounded_rectangle((pad, pad, BASE_SIZE - pad, BASE_SIZE - pad), radius=int(BASE_SIZE * 0.2), fill=BG)
# subtle diagonal highlight
d.polygon([(pad, int(BASE_SIZE*0.55)), (BASE_SIZE-pad, pad), (BASE_SIZE-pad, int(BASE_SIZE*0.35)), (int(BASE_SIZE*0.32), BASE_SIZE-pad)], fill=BG2)

# Matrix glyph (3x3 grid)
grid_size = int(BASE_SIZE * 0.48)
grid_x = int((BASE_SIZE - grid_size) / 2)
grid_y = int((BASE_SIZE - grid_size) / 2)
cell = grid_size // 3
stroke = max(8, BASE_SIZE // 80)

for i in range(4):
    x = grid_x + i * cell
    d.line((x, grid_y, x, grid_y + grid_size), fill=FG, width=stroke)
for i in range(4):
    y = grid_y + i * cell
    d.line((grid_x, y, grid_x + grid_size, y), fill=FG, width=stroke)

# Accent determinant-like slash
d.line((grid_x + cell // 4, grid_y + grid_size - cell // 3, grid_x + grid_size - cell // 4, grid_y + cell // 3), fill=ACC, width=stroke)

# Save master PNG
master_png = ICONS / "icon-1024.png"
img.save(master_png)

sizes = [16, 32, 48, 64, 96, 180, 192, 512]
for s in sizes:
    resized = img.resize((s, s), Image.Resampling.LANCZOS)
    resized.save(ICONS / f"favicon-{s}x{s}.png")

# Apple touch icon
img.resize((180, 180), Image.Resampling.LANCZOS).save(ICONS / "apple-touch-icon.png")

# ICO
img.resize((64, 64), Image.Resampling.LANCZOS).save(
    ICONS / "favicon.ico",
    format="ICO",
    sizes=[(16, 16), (32, 32), (48, 48), (64, 64)],
)

# SVG fallback
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="rgb({BG[0]},{BG[1]},{BG[2]})"/>
      <stop offset="100%" stop-color="rgb({BG2[0]},{BG2[1]},{BG2[2]})"/>
    </linearGradient>
  </defs>
  <rect x="80" y="80" width="864" height="864" rx="180" fill="url(#g)"/>
  <g stroke="rgb({FG[0]},{FG[1]},{FG[2]})" stroke-width="14" fill="none" stroke-linecap="round">
    <path d="M266 266h492M266 430h492M266 594h492M266 758h492"/>
    <path d="M266 266v492M430 266v492M594 266v492M758 266v492"/>
  </g>
  <path d="M300 724L724 300" stroke="rgb({ACC[0]},{ACC[1]},{ACC[2]})" stroke-width="16" stroke-linecap="round"/>
</svg>'''
(ICONS / "favicon.svg").write_text(svg, encoding="utf-8")

manifest = {
    "name": "Matrix Tools",
    "short_name": "MatrixTools",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0f766e",
    "theme_color": "#0f766e",
    "icons": [
        {"src": "/assets/icons/favicon-192x192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/assets/icons/favicon-512x512.png", "sizes": "512x512", "type": "image/png"}
    ]
}
(ICONS / "site.webmanifest").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Favicons generated in", ICONS)
