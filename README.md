# 🌾 CropGuard AI

Computer-vision crop leaf nutrient-deficiency diagnostics, built with Flask + OpenCV.
Upload a leaf photo, get an HSV spectral breakdown, a severity/confidence score, and a
downloadable PDF treatment plan.

## Features

- **Public leaf scanner** — no login required. Select a crop, upload a photo, get an
  instant diagnosis (Healthy / Nitrogen / Phosphorus / Potassium, with confidence %,
  severity, and affected leaf area).
- **Real OpenCV analysis** — HSV color-space segmentation measures green (chlorophyll),
  yellow (chlorosis), brown (necrosis), and purple (anthocyanin) pixel ratios on the
  leaf region of the photo.
- **PDF diagnostic certificates** — generated with ReportLab, includes a unique
  Report ID (`CG-XXXXXX`), the leaf photo, spectral data, and a fertilizer action plan.
- **Admin dashboard** — scan totals, deficiency/severity charts (Chart.js), a searchable
  filterable prediction log, and CSV export.
- **Fertilizer rule configurator** — admins can add/edit/delete crop + deficiency →
  treatment rules used to generate recommendations.
- **Secure admin auth** — password hashing via Werkzeug, session-based login.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| Computer vision | OpenCV (`cv2`), NumPy |
| PDF generation | ReportLab |
| Database | SQLite3 |
| Frontend | HTML5, Bootstrap 5, Chart.js, Font Awesome |
| Auth | Werkzeug password hashing, Flask sessions |

## Project structure

```
cropguard_ai/
├── app.py                 # Flask backend: routes, OpenCV analysis, DB, PDF generation
├── requirements.txt
├── Procfile                # For Render/Railway/Heroku-style deployment
├── run_app.bat              # Windows one-click launcher
├── .env.example             # Copy to .env and fill in your own secrets
├── crop_deficiency.db       # SQLite DB (auto-created on first run)
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── uploads/              # Uploaded leaf photos
│   └── reports/               # Generated PDF reports
└── templates/                # Jinja2 templates (public site + admin panel)
```

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Or on Windows, double-click `run_app.bat`.

- Public portal: http://127.0.0.1:5000/
- Leaf scanner: http://127.0.0.1:5000/detect
- Admin login: http://127.0.0.1:5000/admin/login

**Default admin credentials:** `host` / `CropGuard@2026`
Change these immediately — either from **Admin → Security** after logging in, or by
setting `CROPGUARD_ADMIN_USER` / `CROPGUARD_ADMIN_PASS` env vars *before the database
is first created* (see `.env.example`).

## Making it publicly accessible (no localhost restriction)

Running `python app.py` only serves your own machine at `127.0.0.1`. To let anyone
reach the site, you need a public host. Options, easiest first:

### 1. Free/low-cost cloud hosting (recommended)
- **Render.com** — connect your GitHub repo, it detects Python + the `Procfile`
  automatically, and gives you a public `https://yourapp.onrender.com` URL.
- **Railway.app** — similar one-click deploy flow, free/low-cost tier.
- **PythonAnywhere** — good free tier for small Flask apps.

Steps for Render:
1. Push this project to a GitHub repo.
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app` (already in the `Procfile`, Render auto-detects it).
5. Add environment variables from `.env.example` in Render's dashboard (don't commit
   real secrets to git).
6. Deploy — you'll get a public URL immediately.

### 2. Quick temporary public link (for demos/testing only)
```bash
python app.py
# in another terminal:
ngrok http 5000
```
This gives a temporary public URL that tunnels to your local machine. Fine for a demo,
not for permanent hosting (it dies when you close the terminal).

### 3. Production notes
- The app already uses **Gunicorn** in production (`gunicorn app:app`), not Flask's
  dev server — required for handling real traffic.
- **SQLite is fine for light traffic.** If you expect many concurrent users, migrate to
  PostgreSQL (Render/Railway both offer free Postgres add-ons) — SQLite can lock under
  heavy concurrent writes.
- **Persistent storage:** some free hosts wipe the filesystem on redeploy, which would
  delete `static/uploads/` and the SQLite DB. For anything beyond a demo, either use a
  host with a persistent disk (Render offers this) or move uploads to S3/Cloudinary and
  the DB to a managed Postgres instance.
- **Secrets:** never leave `CROPGUARD_SECRET_KEY` or the admin password hardcoded in
  committed code — use environment variables, as set up in `.env.example`.

## How the diagnosis works

The scanner covers **12 nutrient deficiencies + Healthy** (3 primary macronutrients,
3 secondary macronutrients, 6 micronutrients), using a rule-based computer-vision
pipeline -- **not a trained neural network**. It combines:

1. **HSV color bands** within a leaf mask: green (chlorophyll), yellow (chlorosis),
   brown (necrosis/scorch), purple (anthocyanin), white (severe chlorosis).
2. **Vein vs. interveinal contrast** -- edge detection isolates probable vein pixels
   and compares their color against the tissue between them, to detect "yellow
   between green veins" patterns (Iron, Magnesium, Manganese).
3. **Margin scorch analysis** -- a distance transform separates the leaf's outer
   ring from its interior to detect edge-concentrated browning (Potassium, Molybdenum).
4. **Dark speckle detection** -- small contour blobs within chlorotic areas, a proxy
   for the necrotic spotting seen in Manganese deficiency.
5. **Leaf shape distortion** -- contour solidity flags twisting/cupping/hooking
   (Calcium, Copper, Molybdenum, Boron, Zinc).
6. **Leaf position (the field you select before scanning)** -- old/lower vs.
   young/upper leaf. This is the single biggest accuracy lever, since several
   deficiencies are visually near-identical and only separated by nutrient
   mobility (e.g. interveinal chlorosis on an old leaf → Magnesium; the same
   pattern on a young leaf → Iron).

All the signals are combined into a weighted score per deficiency, then converted to
a confidence % via softmax.

### Confidence tiers (shown on every result)
- **High** — Healthy, Nitrogen, Phosphorus, Potassium, Magnesium, Iron: strong, distinct
  color signals. Most reliable.
- **Medium** — Sulfur, Manganese: color + a secondary cue (speckling), decent reliability.
- **Low** — Calcium, Zinc, Boron, Copper, Molybdenum: rely mainly on leaf-shape distortion,
  which is inherently harder to separate from a single photo without a trained model.
  Treat these as a starting hypothesis, not a confirmed diagnosis.

### Honest limitations
- This is heuristic pattern-matching on pixel colors and contours, tuned and tested
  against synthetic reference images -- it has **not** been validated against a large
  labeled dataset of real diseased leaves, and real photos are messier (lighting,
  disease look-alikes, mixed deficiencies, motion blur).
- Crop species is **not** auto-detected -- you select it manually.
- For anything beyond a demo/decision-support tool, the path to real accuracy is a
  labeled photo dataset (ideally thousands of images per deficiency per crop) and a
  trained CNN -- a materially larger project than this heuristic engine.

The `fertilizer_rules` table (editable from the admin panel) is looked up by
crop + deficiency to build the treatment plan attached to the report and PDF.

## Security notes

- Change the default admin password before deploying anywhere public.
- `CROPGUARD_SECRET_KEY` should be a long random string in production, set via env var.
- Uploaded files are validated by extension and size-limited (8MB) — for a public
  deployment, consider adding rate limiting on `/detect` to prevent abuse.
