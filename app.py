"""
CropGuard AI - Flask backend
Computer-vision based crop leaf nutrient deficiency diagnostics.

Detection engine covers 12 nutrient deficiencies (3 primary macronutrients,
3 secondary macronutrients, 6 micronutrients) plus Healthy, using HSV color
analysis, vein/interveinal contrast, margin-scorch detection, dark-speckle
detection, and leaf-shape distortion -- combined with a user-supplied leaf
position (older/lower vs newer/younger) since nutrient mobility is often the
only reliable way to separate visually similar deficiencies (e.g. Iron vs
Magnesium both look like "yellow between green veins").

IMPORTANT: this is a heuristic, rule-based computer-vision pipeline, not a
trained neural network. Color- and pattern-clear deficiencies (N, P, K, Mg,
Fe, S) are detected with reasonable confidence. Shape/texture-based ones
(Ca, Cu, Mo, B, Zn) are inherently harder to separate from a single leaf
photo and are flagged as lower-confidence in the UI.
"""

import os
import sqlite3
import random
import string
import io
import csv
import json
import base64
import requests
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, g, jsonify, Response
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "static", "reports")
DB_PATH = os.path.join(BASE_DIR, "crop_deficiency.db")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("CROPGUARD_SECRET_KEY", "dev-key-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB uploads

DEFAULT_ADMIN_USER = os.environ.get("CROPGUARD_ADMIN_USER", "host")
DEFAULT_ADMIN_PASS = os.environ.get("CROPGUARD_ADMIN_PASS", "CropGuard@2026")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

CROPS = ["Rice", "Tomato", "Maize", "Wheat", "Cotton", "Chilli", "Groundnut"]

# Confidence tier shown in the UI -- honest about which classes are
# genuinely color-distinguishable vs. inferred from weaker shape cues.
CONFIDENCE_TIER = {
    "Healthy": "high", "Nitrogen": "high", "Phosphorus": "high", "Potassium": "high",
    "Magnesium": "high", "Iron": "high", "Sulfur": "medium", "Manganese": "medium",
    "Calcium": "low", "Zinc": "low", "Boron": "low", "Copper": "low", "Molybdenum": "low",
}

DEFICIENCY_META = {
    "Healthy": {
        "emoji": "🟢", "category": "—",
        "symptom": "No abnormal discoloration or distortion detected. Balanced chlorophyll distribution across the leaf.",
    },
    "Nitrogen": {
        "emoji": "🟡", "category": "Primary macronutrient",
        "symptom": "General yellowing (chlorosis) starting from older, lower leaves; growth appears stunted.",
    },
    "Phosphorus": {
        "emoji": "🟣", "category": "Primary macronutrient",
        "symptom": "Abnormally dark green leaves developing purple, bronze, or reddish tints, often on the undersides.",
    },
    "Potassium": {
        "emoji": "🟤", "category": "Primary macronutrient",
        "symptom": "Margins and tips of older leaves turn yellow, then brown and dry out -- a scorched look.",
    },
    "Magnesium": {
        "emoji": "🟡", "category": "Secondary macronutrient",
        "symptom": "Interveinal chlorosis (yellowing between veins) on older leaves, while the veins stay sharply green.",
    },
    "Calcium": {
        "emoji": "🟠", "category": "Secondary macronutrient",
        "symptom": "New young leaves emerge distorted, hooked, or twisted; leaf tips often show dieback.",
    },
    "Sulfur": {
        "emoji": "🟡", "category": "Secondary macronutrient",
        "symptom": "Pale green to yellow across the whole leaf, similar to nitrogen deficiency, but starting on younger upper leaves.",
    },
    "Iron": {
        "emoji": "⚪", "category": "Micronutrient",
        "symptom": "Distinct interveinal chlorosis on the youngest new leaves; can turn almost white with green veins in severe cases.",
    },
    "Manganese": {
        "emoji": "⚪", "category": "Micronutrient",
        "symptom": "Interveinal yellowing on new leaves similar to iron, but with small speckled brown dead spots in the yellow areas.",
    },
    "Zinc": {
        "emoji": "⚪", "category": "Micronutrient",
        "symptom": "\"Little leaf\" syndrome -- small, crowded leaves, sometimes with chlorotic striping. Rosetting (shortened stem "
                    "spacing) can't be confirmed from a single leaf photo.",
    },
    "Boron": {
        "emoji": "⚪", "category": "Micronutrient",
        "symptom": "Thick, brittle leaves with dying growth tips; stems can crack and fruit/root hearts may rot.",
    },
    "Copper": {
        "emoji": "🟢", "category": "Micronutrient",
        "symptom": "New leaves stay dark green but become misshapen, twisted, or wilted without losing color.",
    },
    "Molybdenum": {
        "emoji": "🟡", "category": "Micronutrient",
        "symptom": "Older leaves twist or cup upward, often looking scorched at the edges (\"whiptail\").",
    },
}

ALL_DEFICIENCY_TYPES = list(DEFICIENCY_META.keys())

# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables if they do not exist and seed defaults."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin',
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            crop_type TEXT NOT NULL,
            crop_confidence REAL NOT NULL DEFAULT 30,
            leaf_position TEXT NOT NULL DEFAULT 'old',
            image_filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            deficiency_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            confidence_tier TEXT NOT NULL DEFAULT 'medium',
            severity_level TEXT NOT NULL,
            affected_area_pct REAL NOT NULL,
            green_pct REAL NOT NULL,
            yellow_pct REAL NOT NULL,
            brown_pct REAL NOT NULL,
            purple_pct REAL NOT NULL,
            white_pct REAL NOT NULL DEFAULT 0,
            visual_symptoms TEXT,
            immediate_action TEXT,
            recommended_fertilizer TEXT,
            application_method TEXT,
            dosage TEXT,
            recovery_time TEXT,
            risk_level TEXT,
            overall_health TEXT,
            diagnosis_method TEXT NOT NULL DEFAULT 'heuristic'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fertilizer_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_type TEXT NOT NULL,
            deficiency_type TEXT NOT NULL,
            immediate_action TEXT NOT NULL,
            recommended_fertilizer TEXT NOT NULL,
            application_method TEXT NOT NULL,
            dosage TEXT NOT NULL,
            recovery_time TEXT NOT NULL,
            UNIQUE(crop_type, deficiency_type)
        )
    """)
    conn.commit()

    # Seed default admin user
    cur = conn.execute("SELECT COUNT(*) AS c FROM users WHERE username = ?", (DEFAULT_ADMIN_USER,))
    if cur.fetchone()["c"] == 0:
        conn.execute(
            "INSERT INTO users (username, email, password, role, created_at) VALUES (?,?,?,?,?)",
            (DEFAULT_ADMIN_USER, "admin@cropguard.local",
             generate_password_hash(DEFAULT_ADMIN_PASS), "admin",
             datetime.now().isoformat(timespec="seconds"))
        )
        conn.commit()

    # Seed default fertilizer rules -- one generic ("All" crops) rule per
    # deficiency. Admins can add crop-specific overrides from the panel.
    cur = conn.execute("SELECT COUNT(*) AS c FROM fertilizer_rules")
    if cur.fetchone()["c"] == 0:
        defaults = [
            ("All", "Nitrogen", "Apply nitrogen-rich fertilizer within 3-5 days; avoid over-irrigation.",
             "Urea (46-0-0) or Ammonium Sulfate", "Broadcast + light irrigation", "40-50 kg/acre", "10-14 days"),
            ("All", "Phosphorus", "Apply phosphorus fertilizer at root zone; ensure soil pH 6-7 for uptake.",
             "Single Super Phosphate (SSP) / DAP", "Soil incorporation near root zone", "30-40 kg/acre", "14-21 days"),
            ("All", "Potassium", "Apply potassium fertilizer; reduce nitrogen temporarily to rebalance uptake.",
             "Muriate of Potash (MOP)", "Broadcast + irrigation", "25-35 kg/acre", "10-15 days"),
            ("All", "Magnesium", "Apply magnesium sulfate; correct soil pH if strongly acidic.",
             "Magnesium Sulfate (Epsom Salt)", "Soil application or foliar spray", "10-15 kg/acre", "10-14 days"),
            ("All", "Calcium", "Apply calcium foliar spray during fruit/leaf development; keep soil moisture consistent.",
             "Calcium Chloride or Calcium Nitrate foliar spray", "Foliar spray, weekly during growth stage", "2-4 g/litre water", "10-20 days"),
            ("All", "Sulfur", "Apply sulfate-based fertilizer; gypsum works well on alkaline soils.",
             "Gypsum or Ammonium Sulfate", "Soil application", "15-20 kg/acre", "10-15 days"),
            ("All", "Iron", "Apply chelated iron as foliar spray for fastest correction, especially on alkaline soil.",
             "Iron Chelate (Fe-EDTA) foliar spray", "Foliar spray, 2 applications 7 days apart", "2-3 g/litre water", "7-10 days"),
            ("All", "Manganese", "Apply manganese sulfate as foliar spray; avoid over-liming soil.",
             "Manganese Sulfate (MnSO4) foliar spray", "Foliar spray, 2 applications 10 days apart", "2-3 g/litre water", "10-14 days"),
            ("All", "Zinc", "Apply zinc sulfate to soil or as foliar spray during early growth stage.",
             "Zinc Sulfate (ZnSO4)", "Soil application or foliar spray", "5-10 kg/acre", "14-21 days"),
            ("All", "Boron", "Apply borax in small, precise doses -- boron toxicity occurs quickly if over-applied.",
             "Borax (Sodium Borate)", "Soil application or dilute foliar spray", "1-2 kg/acre (soil) or 1 g/litre (foliar)", "14-21 days"),
            ("All", "Copper", "Apply copper sulfate or copper oxychloride; common on sandy/peaty soils.",
             "Copper Sulfate (CuSO4) or Copper Oxychloride", "Foliar spray", "1-2 g/litre water", "14-21 days"),
            ("All", "Molybdenum", "Apply sodium molybdate as foliar spray; correct soil pH if strongly acidic.",
             "Sodium Molybdate", "Foliar spray", "0.5-1 g/litre water", "10-14 days"),
            ("All", "Healthy", "No treatment required. Continue standard fertilization and monitoring schedule.",
             "Maintain balanced NPK + micronutrient schedule", "Routine schedule", "As per crop calendar", "N/A"),
        ]
        conn.executemany(
            """INSERT INTO fertilizer_rules
               (crop_type, deficiency_type, immediate_action, recommended_fertilizer,
                application_method, dosage, recovery_time)
               VALUES (?,?,?,?,?,?,?)""",
            defaults
        )
        conn.commit()

    conn.close()


# --------------------------------------------------------------------------
# Auth helper
# --------------------------------------------------------------------------
def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Please log in to access the admin panel.", "warning")
            return redirect(url_for("admin_login"))
        return view(*args, **kwargs)
    return wrapped


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def new_report_id():
    return "CG-" + "".join(random.choices(string.digits, k=6))


# --------------------------------------------------------------------------
# Core computer-vision analysis
# --------------------------------------------------------------------------
def analyze_leaf_image(image_path, leaf_position="old"):
    """
    Analyze a leaf image using HSV color segmentation plus structural cues
    (vein/interveinal contrast, margin scorch, dark speckling, shape
    distortion) to score across 12 nutrient deficiencies + Healthy.

    leaf_position: "old" (older/lower leaf) or "young" (newer/upper leaf).
    This drives mobility-based disambiguation, e.g. interveinal chlorosis
    on an OLD leaf points to Magnesium; the same pattern on a YOUNG leaf
    points to Iron.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read uploaded image")

    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    # --- Leaf mask: real foreground segmentation, not just color exclusion ---
    # A simple "exclude near-white/near-black" mask fails badly on real
    # phone photos with blurred/bokeh backgrounds (very common in macro
    # leaf shots) -- it was letting the ENTIRE frame count as "leaf",
    # diluting every color measurement with background pixels. GrabCut
    # actually segments the photographed subject from its surroundings,
    # assuming the leaf is roughly centered (a reasonable ask of the user).
    leaf_mask = None
    try:
        # GrabCut's internal k-means initialization uses OpenCV's RNG, which
        # is non-deterministic by default -- the SAME photo could otherwise
        # get a different diagnosis on a re-scan, which is unacceptable.
        cv2.setRNGSeed(42)
        gc_work = 320  # downscaled working resolution -- full 512 GrabCut is too slow for a live request
        small = cv2.resize(img, (gc_work, gc_work))
        gc_mask = np.zeros((gc_work, gc_work), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (int(gc_work * 0.06), int(gc_work * 0.06), int(gc_work * 0.88), int(gc_work * 0.88))
        cv2.grabCut(small, gc_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        small_result = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        candidate = cv2.resize(small_result, (512, 512), interpolation=cv2.INTER_NEAREST)
        coverage = cv2.countNonZero(candidate) / (512 * 512)
        if 0.06 <= coverage <= 0.95:  # sane result -- neither empty nor "everything"
            leaf_mask = candidate
    except Exception:
        pass

    if leaf_mask is None:
        # Fallback: simple background-color exclusion (still better than
        # nothing if GrabCut fails or produces a degenerate mask).
        bg_white = cv2.inRange(hsv, (0, 0, 235), (180, 20, 255))
        bg_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 18))
        leaf_mask = cv2.bitwise_not(cv2.bitwise_or(bg_white, bg_black))
    leaf_pixels = max(int(cv2.countNonZero(leaf_mask)), 1)

    # --- Color-band masks within the leaf ---
    green_mask = cv2.bitwise_and(cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)), leaf_mask)
    yellow_mask = cv2.bitwise_and(cv2.inRange(hsv, (18, 40, 60), (34, 255, 255)), leaf_mask)
    brown_mask = cv2.bitwise_and(cv2.inRange(hsv, (5, 40, 20), (20, 255, 180)), leaf_mask)
    # Anthocyanin purpling in real photos often reads as dark reddish-maroon
    # in HSV, not pure violet -- it spans both the violet/magenta end
    # (high hue) AND wraps around into low-hue reddish tones when dark and
    # saturated (distinguishing it from brighter, drier brown scorch).
    purple_violet = cv2.inRange(hsv, (130, 20, 20), (179, 255, 230))
    purple_maroon = cv2.inRange(hsv, (0, 60, 25), (12, 255, 175))
    purple_mask = cv2.bitwise_and(cv2.bitwise_or(purple_violet, purple_maroon), leaf_mask)
    white_mask_raw = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 170), (180, 35, 255)), leaf_mask)
    # Camera flash/glare on a glossy leaf creates small bright specular spots
    # that look identical to "white chlorosis" in raw HSV terms. A real
    # chlorotic/whitened area is diffuse and covers a meaningful patch of
    # tissue; glare is small and isolated. Morphological opening removes
    # the small glare blobs while keeping genuine larger whitened regions.
    white_mask = cv2.morphologyEx(white_mask_raw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    green_pct = round(cv2.countNonZero(green_mask) / leaf_pixels * 100, 2)
    yellow_pct = round(cv2.countNonZero(yellow_mask) / leaf_pixels * 100, 2)
    brown_pct = round(cv2.countNonZero(brown_mask) / leaf_pixels * 100, 2)
    purple_pct = round(cv2.countNonZero(purple_mask) / leaf_pixels * 100, 2)
    white_pct = round(cv2.countNonZero(white_mask) / leaf_pixels * 100, 2)
    affected_area_pct = round(max(0.0, min(100.0, 100 - green_pct)), 2)

    # --- Vein vs interveinal contrast ---
    # Veins create local brightness/color edges; use them as a proxy for
    # "where the veins are" and compare their greenness against the
    # surrounding (interveinal) tissue. Erode away the leaf's own outer
    # silhouette first -- the leaf/background boundary is itself a strong
    # Canny edge and would otherwise be misread as "vein".
    core_mask = cv2.erode(leaf_mask, np.ones((9, 9), np.uint8), iterations=1)
    if cv2.countNonZero(core_mask) < 200:
        core_mask = leaf_mask  # very small leaf region in frame -- don't erode away everything
    # High thresholds + no dilation: real leaf photos have lots of fine
    # texture (secondary veins, surface texture, lighting gradients) that
    # a loose edge detector picks up almost everywhere, diluting the
    # "vein greenness" signal to near-meaningless. Only genuinely strong,
    # well-defined primary vein edges should count.
    edges = cv2.Canny(gray, 150, 250)
    vein_mask = cv2.bitwise_and(edges, core_mask)
    interveinal_mask = cv2.bitwise_and(core_mask, cv2.bitwise_not(vein_mask))

    vein_px = max(int(cv2.countNonZero(vein_mask)), 1)
    interveinal_px = max(int(cv2.countNonZero(interveinal_mask)), 1)
    vein_green_pct = cv2.countNonZero(cv2.bitwise_and(green_mask, vein_mask)) / vein_px * 100
    # Advanced interveinal chlorosis often progresses past yellow into
    # reddish-brown/purple as anthocyanins accumulate -- checking for
    # yellow alone undercounts real, advanced-stage cases. Count any
    # non-green discoloration between the veins.
    interveinal_nongreen_mask = cv2.bitwise_or(cv2.bitwise_or(yellow_mask, purple_mask), brown_mask)
    interveinal_nongreen_pct = cv2.countNonZero(cv2.bitwise_and(interveinal_nongreen_mask, interveinal_mask)) / interveinal_px * 100
    # Real-world photos have natural vein/lighting texture that Canny picks
    # up even on ordinary leaves. Gate on both components: below the
    # threshold, treat it as noise (score = 0); at or above it, use the
    # full raw signal so genuinely strong patterns aren't watered down.
    if vein_green_pct >= 35.0 and interveinal_nongreen_pct >= 20.0:
        interveinal_chlorosis_score = round(min(100.0, (vein_green_pct * interveinal_nongreen_pct) / 100), 2)
    else:
        interveinal_chlorosis_score = 0.0

    # --- Chlorotic patch structure: fine uniform marbling vs. large blotches ---
    # Magnesium deficiency classically shows many small, evenly-distributed
    # patches following the fine vein network. Zinc deficiency shows fewer,
    # larger, more irregular blotches. Measure how much one dominant patch
    # accounts for the total chlorotic area -- high means "blotchy", low
    # means "fine marbling".
    blotch_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blotch_areas = sorted([cv2.contourArea(c) for c in blotch_contours if cv2.contourArea(c) > 5], reverse=True)
    blotch_total = sum(blotch_areas)
    blotch_dominance_pct = (blotch_areas[0] / blotch_total * 100) if blotch_total > 0 and blotch_areas else 0.0

    # --- Margin (scorch) analysis ---
    # Pad with a background border first: if the leaf fills the entire
    # photo frame edge-to-edge, distanceTransform has no zero pixels to
    # measure from and silently returns garbage. Padding guarantees a
    # true "outside" for the transform to reference.
    padded = cv2.copyMakeBorder(leaf_mask, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
    dist_padded = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    dist = dist_padded[6:-6, 6:-6]
    # Scale the margin band to the leaf's own size: a leaf photographed
    # small-in-frame needs a thinner band than one filling the whole photo,
    # otherwise a fixed pixel width is either too thin to catch real edge
    # scorch or too thick and bleeds into the interior.
    leaf_radius_px = (leaf_pixels / np.pi) ** 0.5
    margin_width = max(10, min(40, leaf_radius_px * 0.14))
    margin_mask = ((dist > 0) & (dist < margin_width)).astype(np.uint8) * 255
    margin_mask = cv2.bitwise_and(margin_mask, leaf_mask)
    interior_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(margin_mask))
    margin_px = max(int(cv2.countNonZero(margin_mask)), 1)
    interior_px = max(int(cv2.countNonZero(interior_mask)), 1)
    # Real scorch/drying progresses through a color range -- golden-tan
    # through to dark brown -- not one fixed hue. Checking brown alone
    # undercounts real cases where the margin has only dried to tan/gold
    # rather than fully browned yet.
    scorch_mask = cv2.bitwise_or(brown_mask, yellow_mask)
    margin_scorch_pct = cv2.countNonZero(cv2.bitwise_and(scorch_mask, margin_mask)) / margin_px * 100
    interior_scorch_pct = cv2.countNonZero(cv2.bitwise_and(scorch_mask, interior_mask)) / interior_px * 100
    scorch_score = round(max(0.0, margin_scorch_pct - interior_scorch_pct), 2)

    # --- Dark speckle detection (proxy for Manganese necrotic spots) ---
    # Real photos have natural shadows, dust, and minor blemishes that create
    # many tiny dark specks regardless of the leaf's actual condition -- a
    # flat synthetic test image doesn't have this noise, which is why this
    # bug didn't show up until real photos were tested. Defenses: (1) blur
    # first to merge away single-pixel noise, (2) real Manganese spotting
    # only occurs on leaves that are already meaningfully chlorotic, so
    # gate the whole signal off unless overall yellowing is present --
    # generic shadows/dust on an otherwise green leaf don't qualify.
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    dark_mask = cv2.bitwise_and(cv2.inRange(blurred, (0, 40, 0), (180, 255, 95)), leaf_mask)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    speckle_count = sum(1 for c in contours if 4 <= cv2.contourArea(c) <= 45)
    if yellow_pct < 10.0:
        speckle_score = 0.0
    else:
        speckle_score = round(min(100.0, max(0, speckle_count - 3) / leaf_pixels * 20000), 2)

    # --- Leaf shape distortion (proxy for twisting/cupping/hooking) ---
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    distortion_score = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            distortion_score = round(max(0.0, (1 - solidity)) * 100, 2)

    young = leaf_position == "young"
    old = not young

    # Non-green tissue factor: shape-distortion deficiencies that involve
    # dieback/discoloration (Calcium) should only score highly when there's
    # actually some non-green tissue present, otherwise a distorted-but-
    # still-fully-green leaf gets misread as Calcium instead of Copper.
    non_green_factor = 0.3 + 0.7 * min(1.0, (yellow_pct + brown_pct + white_pct) / 6.0)
    copper_purity = max(0.2, 1.0 - (yellow_pct + brown_pct) / 12.0)

    # --- Weighted scoring across all 13 classes ---
    scores = {
        "Healthy": green_pct * 1.0,
        "Nitrogen": max(0.0, yellow_pct - interveinal_chlorosis_score * 1.5 - scorch_score * 1.8) * (1.0 if old else 0.4),
        "Phosphorus": purple_pct * 1.3,
        "Potassium": (scorch_score * 3.0 + brown_pct * 0.4) * (1.0 if old else 0.5),
        "Magnesium": interveinal_chlorosis_score * (1.0 if old else 0.3) * max(0.3, 1.0 - blotch_dominance_pct / 140.0) + speckle_score * (0.5 if old else 0.05),
        "Calcium": distortion_score * 2.8 * (1.0 if young else 0.5) * non_green_factor,
        "Sulfur": max(0.0, yellow_pct - interveinal_chlorosis_score * 1.5) * (1.0 if young else 0.4),
        "Iron": (interveinal_chlorosis_score * (1.15 if young else 0.3)) + white_pct * 0.9,
        "Manganese": (interveinal_chlorosis_score * (0.9 if young else 0.3)) + speckle_score * 2.0 * (1.0 if young else 0.3),
        "Zinc": distortion_score * 0.9 + interveinal_chlorosis_score * (0.3 + blotch_dominance_pct / 100.0) * (1.0 if old else 0.5),
        "Boron": (distortion_score * 1.1 + brown_pct * 0.4) * (1.0 if young else 0.6),
        "Copper": distortion_score * 1.5 * copper_purity * (1.0 if young else 0.5),
        "Molybdenum": (distortion_score * 0.9 + scorch_score * 0.5) * (1.0 if old else 0.5),
    }

    # Healthy wins only if green dominates and nothing else scores meaningfully
    non_healthy_scores = {k: v for k, v in scores.items() if k != "Healthy"}
    top_non_healthy = max(non_healthy_scores, key=non_healthy_scores.get)
    top_non_healthy_score = non_healthy_scores[top_non_healthy]

    if green_pct >= 65 and top_non_healthy_score < 10:
        deficiency_type = "Healthy"
    else:
        deficiency_type = top_non_healthy

    # Softmax-style confidence over all scores (temperature-scaled)
    values = np.array(list(scores.values()), dtype=float)
    labels = list(scores.keys())
    exp = np.exp((values - values.max()) / 9.0)
    softmax = exp / exp.sum()
    confidence = round(float(softmax[labels.index(deficiency_type)]) * 100, 2)
    confidence = max(confidence, 40.0)

    # Severity bucket
    if deficiency_type == "Healthy":
        severity, risk_level, overall_health = "None", "Low", "Healthy"
    else:
        if affected_area_pct <= 20:
            severity = "Mild"
        elif affected_area_pct <= 50:
            severity = "Moderate"
        else:
            severity = "Severe"
        risk_level = {"Mild": "Low", "Moderate": "Medium", "Severe": "High"}[severity]
        overall_health = "Deficient"

    return {
        "deficiency_type": deficiency_type,
        "confidence": confidence,
        "confidence_tier": CONFIDENCE_TIER.get(deficiency_type, "medium"),
        "severity_level": severity,
        "affected_area_pct": affected_area_pct,
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "brown_pct": brown_pct,
        "purple_pct": purple_pct,
        "white_pct": white_pct,
        "risk_level": risk_level,
        "overall_health": overall_health,
    }


def detect_crop_type(image_path):
    """
    Best-effort crop-species guess from leaf shape alone (aspect ratio,
    solidity/compactness, and margin lobing via convexity defects).

    IMPORTANT: unlike the deficiency engine, this has no color/texture
    training data behind it at all -- it's a rough shape-bucket heuristic
    over 7 visually overlapping crop species. Treat the result as a
    starting guess, not a reliable identification. Always shown with a
    low/experimental confidence tag and is user-correctable.
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Rice", 30.0

    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_white = cv2.inRange(hsv, (0, 0, 235), (180, 20, 255))
    bg_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 18))
    leaf_mask = cv2.bitwise_not(cv2.bitwise_or(bg_white, bg_black))

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Rice", 30.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 400:
        return "Rice", 30.0

    hull = cv2.convexHull(largest)
    hull_area = max(cv2.contourArea(hull), 1)
    solidity = area / hull_area

    rect = cv2.minAreaRect(largest)
    (rw, rh) = rect[1]
    long_side, short_side = max(rw, rh), max(min(rw, rh), 1)
    aspect_ratio = long_side / short_side

    hull_indices = cv2.convexHull(largest, returnPoints=False)
    defect_count = 0
    if hull_indices is not None and len(hull_indices) > 3:
        try:
            defects = cv2.convexityDefects(largest, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    depth = defects[i, 0, 3] / 256.0
                    if depth > 8:  # ignore tiny noise dents, count real lobes/notches
                        defect_count += 1
        except cv2.error:
            defect_count = 0

    # Shape-bucket scoring per crop (rough stereotypes, not trained data)
    crop_scores = {
        "Rice":      max(0.0, aspect_ratio - 4.0) * 3 + (10 if defect_count <= 1 else 0),
        "Wheat":     max(0.0, min(aspect_ratio, 6.0) - 3.0) * 3 + (8 if defect_count <= 1 else 0),
        "Maize":     max(0.0, 4.0 - abs(aspect_ratio - 3.0)) * 3 + (6 if defect_count <= 2 else 0),
        "Cotton":    (defect_count * 6) * (1.0 if aspect_ratio < 1.8 else 0.3),
        "Groundnut": max(0.0, 2.0 - abs(aspect_ratio - 1.4)) * 5 * (solidity if solidity > 0.85 else 0.3),
        "Chilli":    max(0.0, 2.2 - abs(aspect_ratio - 2.3)) * 5 * (solidity if solidity > 0.85 else 0.4),
        "Tomato":    (defect_count * 5 + max(0.0, (0.85 - solidity)) * 40) * (1.0 if aspect_ratio < 2.5 else 0.4),
    }
    best_crop = max(crop_scores, key=crop_scores.get)
    values = np.array(list(crop_scores.values()), dtype=float)
    labels = list(crop_scores.keys())
    exp = np.exp((values - values.max()) / 6.0)
    softmax = exp / exp.sum()
    confidence = round(float(softmax[labels.index(best_crop)]) * 100, 2)
    # This classifier is fundamentally weaker evidence than the deficiency
    # engine -- cap confidence so the UI never implies false certainty.
    confidence = max(25.0, min(confidence, 65.0))
    return best_crop, confidence


def analyze_leaf_with_claude(image_path, leaf_position):
    """
    Send the leaf photo to Claude's vision API for real image-based
    diagnosis, replacing the old color-heuristic guesswork. Returns a dict
    matching the shape the rest of the app expects, or None if the API
    key isn't configured or the call fails for any reason (caller should
    fall back to the heuristic engine in that case).
    """
    if not ANTHROPIC_API_KEY:
        return None

    ext = image_path.rsplit(".", 1)[-1].lower()
    media_type = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp",
    }.get(ext, "image/jpeg")

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

    deficiency_list = "\n".join(
        f'- "{name}" ({meta["category"]}): {meta["symptom"]}'
        for name, meta in DEFICIENCY_META.items()
    )
    crop_list = ", ".join(CROPS)
    leaf_pos_text = "an OLDER / LOWER leaf" if leaf_position == "old" else "a NEWER / YOUNGER leaf"

    prompt = f"""You are an agronomy expert analyzing a photo of a crop leaf to diagnose nutrient deficiencies.

The person photographed {leaf_pos_text} on the plant. This matters: mobile nutrients (N, P, K, Mg) show symptoms on OLDER leaves first; immobile nutrients (Ca, Fe, Mn, Zn, B, Cu, Mo, S) show symptoms on NEWER leaves first.

Classify the leaf into exactly ONE of these categories:
{deficiency_list}

Also identify the crop species -- your best guess from this list if it clearly matches one: {crop_list}. If it doesn't clearly match any of those, give your best general guess of the actual species anyway.

Respond with ONLY a raw JSON object (no markdown fences, no other text) in exactly this shape:
{{
  "crop_type": "<one crop name, your best guess even if uncertain>",
  "crop_confidence": <0-100 integer, how confident you are in the crop identification>,
  "deficiency_type": "<one of the exact category names above>",
  "confidence": <0-100 integer, how confident you are in the diagnosis>,
  "severity_level": "<None, Mild, Moderate, or Severe -- None only if deficiency_type is Healthy>",
  "affected_area_pct": <0-100 integer, rough percent of visible leaf area showing symptoms>,
  "visual_symptoms": "<one or two sentences describing exactly what you observe in THIS photo that supports your diagnosis>"
}}"""

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 500,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            },
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
    except Exception as exc:
        app.logger.error(f"Claude vision analysis failed: {exc}")
        return None

    deficiency_type = parsed.get("deficiency_type", "")
    if deficiency_type not in ALL_DEFICIENCY_TYPES:
        return None  # malformed/unexpected response -- let caller fall back

    crop_type = parsed.get("crop_type", "")
    if not crop_type:
        crop_type = "Rice"

    confidence = max(0.0, min(100.0, float(parsed.get("confidence", 50))))
    crop_confidence = max(0.0, min(100.0, float(parsed.get("crop_confidence", 50))))
    severity_level = parsed.get("severity_level", "Mild")
    if severity_level not in ("None", "Mild", "Moderate", "Severe"):
        severity_level = "None" if deficiency_type == "Healthy" else "Mild"
    affected_area_pct = max(0.0, min(100.0, float(parsed.get("affected_area_pct", 0))))
    visual_symptoms = parsed.get("visual_symptoms", "") or DEFICIENCY_META.get(deficiency_type, {}).get("symptom", "")

    if deficiency_type == "Healthy":
        risk_level, overall_health = "Low", "Healthy"
    else:
        risk_level = {"Mild": "Low", "Moderate": "Medium", "Severe": "High"}.get(severity_level, "Medium")
        overall_health = "Deficient"

    confidence_tier = "high" if confidence >= 80 else "medium" if confidence >= 55 else "low"

    return {
        "crop_type": crop_type,
        "crop_confidence": round(crop_confidence, 1),
        "deficiency_type": deficiency_type,
        "confidence": round(confidence, 1),
        "confidence_tier": confidence_tier,
        "severity_level": severity_level,
        "affected_area_pct": round(affected_area_pct, 1),
        "risk_level": risk_level,
        "overall_health": overall_health,
        "visual_symptoms": visual_symptoms,
    }


def get_treatment(crop_type, deficiency_type):
    db = get_db()
    row = db.execute(
        "SELECT * FROM fertilizer_rules WHERE crop_type = ? AND deficiency_type = ?",
        (crop_type, deficiency_type)
    ).fetchone()
    if row is None:
        row = db.execute(
            "SELECT * FROM fertilizer_rules WHERE crop_type = 'All' AND deficiency_type = ?",
            (deficiency_type,)
        ).fetchone()
    if row is None:
        return {
            "immediate_action": "Consult a local agronomist for a tailored treatment plan.",
            "recommended_fertilizer": "N/A",
            "application_method": "N/A",
            "dosage": "N/A",
            "recovery_time": "N/A",
        }
    return {
        "immediate_action": row["immediate_action"],
        "recommended_fertilizer": row["recommended_fertilizer"],
        "application_method": row["application_method"],
        "dosage": row["dosage"],
        "recovery_time": row["recovery_time"],
    }


# --------------------------------------------------------------------------
# PDF report generation
# --------------------------------------------------------------------------
def generate_pdf_report(record, image_path):
    pdf_path = os.path.join(REPORT_DIR, f"{record['report_id']}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"],
                                  textColor=colors.HexColor("#1B4332"), alignment=TA_CENTER)
    sub_style = ParagraphStyle("SubStyle", parent=styles["Normal"],
                                textColor=colors.HexColor("#52734D"), alignment=TA_CENTER, fontSize=10)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#1B4332"))
    body_style = styles["Normal"]

    elements = [
        Paragraph("CropGuard AI — Diagnostic Certificate", title_style),
        Paragraph(f"Report ID: {record['report_id']}  |  Generated: {record['created_at']}", sub_style),
        Spacer(1, 10 * mm),
    ]

    if image_path and os.path.exists(image_path):
        try:
            elements.append(RLImage(image_path, width=70 * mm, height=70 * mm))
            elements.append(Spacer(1, 6 * mm))
        except Exception:
            pass

    leaf_pos_label = "Older / lower leaf" if record.get("leaf_position") == "old" else "Newer / younger leaf"
    summary_data = [
        ["Crop Type", record["crop_type"]],
        ["Leaf Position", leaf_pos_label],
        ["Diagnosis", record["deficiency_type"]],
        ["AI Confidence", f"{record['confidence']}% ({record.get('confidence_tier','medium')} confidence)"],
        ["Severity", record["severity_level"]],
        ["Affected Leaf Area", f"{record['affected_area_pct']}%"],
        ["Risk Level", record["risk_level"]],
        ["Overall Health", record["overall_health"]],
    ]
    summary_table = Table(summary_data, colWidths=[55 * mm, 100 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F0E4")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1B4332")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD9C6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [Paragraph("Diagnostic Summary", h2_style), Spacer(1, 3 * mm), summary_table, Spacer(1, 6 * mm)]

    spectral_data = [
        ["Green (Chlorophyll)", f"{record['green_pct']}%"],
        ["Yellow (Chlorosis)", f"{record['yellow_pct']}%"],
        ["Brown (Necrosis)", f"{record['brown_pct']}%"],
        ["Purple (Anthocyanin)", f"{record['purple_pct']}%"],
        ["White (Severe chlorosis)", f"{record.get('white_pct', 0)}%"],
    ]
    spectral_table = Table(spectral_data, colWidths=[55 * mm, 100 * mm])
    spectral_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD9C6")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [Paragraph("HSV Spectral Analysis", h2_style), Spacer(1, 3 * mm), spectral_table, Spacer(1, 6 * mm)]

    elements.append(Paragraph("Treatment & Fertilizer Action Plan", h2_style))
    elements.append(Spacer(1, 3 * mm))
    treatment_lines = [
        f"<b>Visual Symptoms:</b> {record['visual_symptoms']}",
        f"<b>Immediate Action:</b> {record['immediate_action']}",
        f"<b>Recommended Fertilizer:</b> {record['recommended_fertilizer']}",
        f"<b>Application Method:</b> {record['application_method']}",
        f"<b>Dosage:</b> {record['dosage']}",
        f"<b>Expected Recovery Time:</b> {record['recovery_time']}",
    ]
    for line in treatment_lines:
        elements.append(Paragraph(line, body_style))
        elements.append(Spacer(1, 2 * mm))

    elements.append(Spacer(1, 10 * mm))
    if record.get("diagnosis_method") == "claude_vision":
        footnote_text = (
            "This diagnosis was generated by Claude Vision AI analyzing the actual leaf photo, and is intended as "
            "a decision-support aid. For high-value crops or persistent symptoms, confirm with a certified agronomist."
        )
    else:
        footnote_text = (
            "AI vision analysis was unavailable for this scan, so this report used a fallback rule-based "
            "computer-vision pipeline (HSV color analysis, not a trained model) -- treat it as lower-confidence. "
            "For high-value crops or persistent symptoms, confirm with a certified agronomist."
        )
    elements.append(Paragraph(
        footnote_text,
        ParagraphStyle("Footnote", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(elements)
    return pdf_path


# --------------------------------------------------------------------------
# Public routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", crops=CROPS)


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html", crops=CROPS)

    leaf_position = request.form.get("leaf_position", "").strip()
    file = request.files.get("leaf_image")

    if leaf_position not in ("old", "young"):
        flash("Please tell us whether this is an older/lower leaf or a newer/younger leaf.", "danger")
        return redirect(url_for("detect"))

    if not file or file.filename == "":
        flash("Please choose or capture a leaf image.", "danger")
        return redirect(url_for("detect"))

    if not allowed_file(file.filename):
        flash("Please upload a clear JPG, JPEG, PNG, or WEBP image of the leaf.", "danger")
        return redirect(url_for("detect"))

    report_id = new_report_id()
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = secure_filename(f"{report_id}.{ext}")
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        spectral = analyze_leaf_image(save_path, leaf_position=leaf_position)
    except Exception as exc:
        app.logger.error(f"Leaf analysis failed for {filename}: {exc}")
        flash("Please make sure the image contains a clear crop leaf and try again.", "danger")
        return redirect(url_for("detect"))

    # Primary diagnosis: Claude's vision model actually looks at the photo.
    # Falls back to the color-heuristic engine only if no API key is set
    # or the API call fails, so the app still works either way.
    claude_result = analyze_leaf_with_claude(save_path, leaf_position)
    used_ai_vision = claude_result is not None

    if used_ai_vision:
        analysis = {
            "deficiency_type": claude_result["deficiency_type"],
            "confidence": claude_result["confidence"],
            "confidence_tier": claude_result["confidence_tier"],
            "severity_level": claude_result["severity_level"],
            "affected_area_pct": claude_result["affected_area_pct"],
            "risk_level": claude_result["risk_level"],
            "overall_health": claude_result["overall_health"],
            "green_pct": spectral["green_pct"],
            "yellow_pct": spectral["yellow_pct"],
            "brown_pct": spectral["brown_pct"],
            "purple_pct": spectral["purple_pct"],
            "white_pct": spectral["white_pct"],
        }
        crop_type = claude_result["crop_type"] if claude_result["crop_type"] in CROPS else "Rice"
        crop_confidence = claude_result["crop_confidence"]
        visual_symptoms_override = claude_result["visual_symptoms"]
    else:
        analysis = spectral
        try:
            crop_type, crop_confidence = detect_crop_type(save_path)
        except Exception:
            crop_type, crop_confidence = "Rice", 30.0
        visual_symptoms_override = None

    # Manual override: if the user picked a specific crop from the dropdown
    # instead of leaving it on "Auto-detect", trust their choice -- they
    # know their own field better than any guess.
    manual_crop = request.form.get("crop_type", "").strip()
    if manual_crop and manual_crop in CROPS:
        crop_type = manual_crop
        crop_confidence = 100.0  # user-confirmed, not a guess

    treatment = get_treatment(crop_type, analysis["deficiency_type"])
    meta = DEFICIENCY_META.get(analysis["deficiency_type"], {})

    record = {
        "report_id": report_id,
        "crop_type": crop_type,
        "crop_confidence": crop_confidence,
        "leaf_position": leaf_position,
        "image_filename": filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "visual_symptoms": visual_symptoms_override or meta.get("symptom", ""),
        "diagnosis_method": "claude_vision" if used_ai_vision else "heuristic",
        **analysis,
        **treatment,
    }

    db = get_db()
    db.execute("""
        INSERT INTO predictions (
            report_id, crop_type, crop_confidence, leaf_position, image_filename, created_at, deficiency_type, confidence,
            confidence_tier, severity_level, affected_area_pct, green_pct, yellow_pct, brown_pct, purple_pct,
            white_pct, visual_symptoms, immediate_action, recommended_fertilizer, application_method,
            dosage, recovery_time, risk_level, overall_health, diagnosis_method
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record["report_id"], record["crop_type"], record["crop_confidence"], record["leaf_position"], record["image_filename"], record["created_at"],
        record["deficiency_type"], record["confidence"], record["confidence_tier"], record["severity_level"],
        record["affected_area_pct"], record["green_pct"], record["yellow_pct"], record["brown_pct"], record["purple_pct"],
        record["white_pct"], record["visual_symptoms"], record["immediate_action"], record["recommended_fertilizer"],
        record["application_method"], record["dosage"], record["recovery_time"],
        record["risk_level"], record["overall_health"], record["diagnosis_method"]
    ))
    db.commit()

    return redirect(url_for("results", report_id=report_id))


@app.route("/results/<report_id>")
def results(report_id):
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE report_id = ?", (report_id,)).fetchone()
    if row is None:
        flash("Report not found.", "danger")
        return redirect(url_for("detect"))
    emoji = DEFICIENCY_META.get(row["deficiency_type"], {}).get("emoji", "🌿")
    category = DEFICIENCY_META.get(row["deficiency_type"], {}).get("category", "—")
    return render_template("results.html", r=row, emoji=emoji, category=category)


@app.route("/report/<report_id>/pdf")
def report_pdf(report_id):
    db = get_db()
    row = db.execute("SELECT * FROM predictions WHERE report_id = ?", (report_id,)).fetchone()
    if row is None:
        flash("Report not found.", "danger")
        return redirect(url_for("detect"))
    image_path = os.path.join(UPLOAD_DIR, row["image_filename"])
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)  # in case the host's ephemeral disk was reset
        pdf_path = generate_pdf_report(dict(row), image_path)
        return send_file(pdf_path, as_attachment=True, download_name=f"{report_id}_CropGuard_Report.pdf")
    except Exception as exc:
        app.logger.error(f"PDF generation failed for {report_id}: {exc}")
        flash("Unable to generate the PDF report right now. Please try again in a moment.", "danger")
        return redirect(url_for("results", report_id=report_id))


@app.route("/reports")
def reports():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50").fetchall()
    return render_template("reports.html", rows=rows)


@app.route("/dashboard")
def dashboard():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    healthy = db.execute("SELECT COUNT(*) c FROM predictions WHERE overall_health='Healthy'").fetchone()["c"]
    deficient = total - healthy
    recent = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 8").fetchall()
    return render_template("dashboard.html", total=total, healthy=healthy, deficient=deficient, recent=recent)


# --------------------------------------------------------------------------
# Admin routes
# --------------------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if user is None or not check_password_hash(user["password"], password):
        flash("Invalid username or password.", "danger")
        return redirect(url_for("admin_login"))

    session["admin_id"] = user["id"]
    session["admin_username"] = user["username"]
    flash("Welcome back, " + user["username"] + "!", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    healthy = db.execute("SELECT COUNT(*) c FROM predictions WHERE overall_health='Healthy'").fetchone()["c"]
    deficient = total - healthy

    today = datetime.now().strftime("%Y-%m-%d")
    daily = db.execute(
        "SELECT COUNT(*) c FROM predictions WHERE created_at LIKE ?", (today + "%",)
    ).fetchone()["c"]

    top_def_row = db.execute("""
        SELECT deficiency_type, COUNT(*) c FROM predictions
        WHERE deficiency_type != 'Healthy'
        GROUP BY deficiency_type ORDER BY c DESC LIMIT 1
    """).fetchone()
    top_deficiency = top_def_row["deficiency_type"] if top_def_row else "N/A"

    top_crop_row = db.execute("""
        SELECT crop_type, COUNT(*) c FROM predictions
        GROUP BY crop_type ORDER BY c DESC LIMIT 1
    """).fetchone()
    top_crop = top_crop_row["crop_type"] if top_crop_row else "N/A"

    deficiency_dist = db.execute("""
        SELECT deficiency_type, COUNT(*) c FROM predictions GROUP BY deficiency_type
    """).fetchall()

    severity_dist = db.execute("""
        SELECT severity_level, COUNT(*) c FROM predictions GROUP BY severity_level
    """).fetchall()

    crop_filter = request.args.get("crop", "")
    severity_filter = request.args.get("severity", "")
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if crop_filter:
        query += " AND crop_type = ?"
        params.append(crop_filter)
    if severity_filter:
        query += " AND severity_level = ?"
        params.append(severity_filter)
    query += " ORDER BY id DESC LIMIT 200"
    logs = db.execute(query, params).fetchall()

    return render_template(
        "admin_dashboard.html",
        total=total, healthy=healthy, deficient=deficient, daily=daily,
        top_deficiency=top_deficiency, top_crop=top_crop,
        deficiency_dist=deficiency_dist, severity_dist=severity_dist,
        logs=logs, crops=CROPS, crop_filter=crop_filter, severity_filter=severity_filter
    )


@app.route("/admin/export/csv")
@admin_required
def admin_export_csv():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    if rows:
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow(list(r))
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cropguard_predictions.csv"}
    )


@app.route("/admin/fertilizers", methods=["GET", "POST"])
@admin_required
def admin_fertilizers():
    db = get_db()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add" or action == "update":
            rule_id = request.form.get("id")
            data = (
                request.form.get("crop_type"),
                request.form.get("deficiency_type"),
                request.form.get("immediate_action"),
                request.form.get("recommended_fertilizer"),
                request.form.get("application_method"),
                request.form.get("dosage"),
                request.form.get("recovery_time"),
            )
            if action == "add":
                try:
                    db.execute("""INSERT INTO fertilizer_rules
                        (crop_type, deficiency_type, immediate_action, recommended_fertilizer,
                         application_method, dosage, recovery_time) VALUES (?,?,?,?,?,?,?)""", data)
                    flash("Fertilizer rule added.", "success")
                except sqlite3.IntegrityError:
                    flash("A rule for this crop + deficiency already exists.", "danger")
            else:
                db.execute("""UPDATE fertilizer_rules SET
                    crop_type=?, deficiency_type=?, immediate_action=?, recommended_fertilizer=?,
                    application_method=?, dosage=?, recovery_time=? WHERE id=?""", data + (rule_id,))
                flash("Fertilizer rule updated.", "success")
            db.commit()
        elif action == "delete":
            db.execute("DELETE FROM fertilizer_rules WHERE id = ?", (request.form.get("id"),))
            db.commit()
            flash("Fertilizer rule deleted.", "info")
        return redirect(url_for("admin_fertilizers"))

    rules = db.execute("SELECT * FROM fertilizer_rules ORDER BY crop_type, deficiency_type").fetchall()
    return render_template("admin_fertilizers.html", rules=rules, crops=["All"] + CROPS, deficiencies=ALL_DEFICIENCY_TYPES)


@app.route("/admin/password", methods=["GET", "POST"])
@admin_required
def admin_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (session["admin_id"],)).fetchone()

        if not check_password_hash(user["password"], current):
            flash("Current password is incorrect.", "danger")
        elif len(new) < 8:
            flash("New password must be at least 8 characters.", "danger")
        elif new != confirm:
            flash("New password and confirmation do not match.", "danger")
        else:
            db.execute("UPDATE users SET password = ? WHERE id = ?",
                       (generate_password_hash(new), user["id"]))
            db.commit()
            flash("Password updated successfully.", "success")
        return redirect(url_for("admin_password"))

    return render_template("admin_password.html")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
else:
    init_db()
