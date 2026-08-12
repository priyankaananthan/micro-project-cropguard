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
            overall_health TEXT
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

    # --- Leaf mask (exclude background only) ---
    # A strict saturation floor would wrongly exclude pale/white chlorotic
    # leaf tissue (e.g. severe Iron deficiency turns leaves near-white).
    # Instead, only exclude near-pure-white paper/overexposure and
    # near-pure-black shadow -- everything else counts as leaf.
    bg_white = cv2.inRange(hsv, (0, 0, 235), (180, 20, 255))
    bg_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 18))
    leaf_mask = cv2.bitwise_not(cv2.bitwise_or(bg_white, bg_black))
    leaf_pixels = max(int(cv2.countNonZero(leaf_mask)), 1)

    # --- Color-band masks within the leaf ---
    green_mask = cv2.bitwise_and(cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)), leaf_mask)
    yellow_mask = cv2.bitwise_and(cv2.inRange(hsv, (18, 40, 60), (34, 255, 255)), leaf_mask)
    brown_mask = cv2.bitwise_and(cv2.inRange(hsv, (5, 40, 20), (20, 255, 180)), leaf_mask)
    purple_mask = cv2.bitwise_and(cv2.inRange(hsv, (125, 25, 20), (160, 255, 200)), leaf_mask)
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
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, core_mask)
    vein_mask = cv2.bitwise_and(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1), core_mask)
    interveinal_mask = cv2.bitwise_and(core_mask, cv2.bitwise_not(vein_mask))

    vein_px = max(int(cv2.countNonZero(vein_mask)), 1)
    interveinal_px = max(int(cv2.countNonZero(interveinal_mask)), 1)
    vein_green_pct = cv2.countNonZero(cv2.bitwise_and(green_mask, vein_mask)) / vein_px * 100
    interveinal_yellow_pct = cv2.countNonZero(cv2.bitwise_and(yellow_mask, interveinal_mask)) / interveinal_px * 100
    # Real-world photos have natural vein/lighting texture that Canny picks
    # up even on ordinary leaves. Gate on both components: below the
    # threshold, treat it as noise (score = 0); at or above it, use the
    # full raw signal so genuinely strong patterns aren't watered down.
    if vein_green_pct >= 35.0 and interveinal_yellow_pct >= 30.0:
        interveinal_chlorosis_score = round(min(100.0, (vein_green_pct * interveinal_yellow_pct) / 100), 2)
    else:
        interveinal_chlorosis_score = 0.0

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
    margin_brown_pct = cv2.countNonZero(cv2.bitwise_and(brown_mask, margin_mask)) / margin_px * 100
    interior_brown_pct = cv2.countNonZero(cv2.bitwise_and(brown_mask, interior_mask)) / interior_px * 100
    scorch_score = round(max(0.0, margin_brown_pct - interior_brown_pct), 2)

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
        "Nitrogen": max(0.0, yellow_pct - interveinal_chlorosis_score * 0.5) * (1.0 if old else 0.4),
        "Phosphorus": purple_pct * 1.3,
        "Potassium": (scorch_score * 1.4 + brown_pct * 0.4) * (1.0 if old else 0.5),
        "Magnesium": interveinal_chlorosis_score * (1.0 if old else 0.3),
        "Calcium": distortion_score * 2.8 * (1.0 if young else 0.5) * non_green_factor,
        "Sulfur": max(0.0, yellow_pct - interveinal_chlorosis_score * 0.3) * (1.0 if young else 0.4),
        "Iron": (interveinal_chlorosis_score * (1.15 if young else 0.3)) + white_pct * 0.9,
        "Manganese": (interveinal_chlorosis_score * (0.9 if young else 0.3)) + speckle_score * 2.0,
        "Zinc": distortion_score * 0.9 + interveinal_chlorosis_score * 0.5,
        "Boron": (distortion_score * 1.1 + brown_pct * 0.4) * (1.0 if young else 0.6),
        "Copper": distortion_score * 1.5 * copper_purity * (1.0 if young else 0.5),
        "Molybdenum": (distortion_score * 1.3 + scorch_score * 0.7) * (1.0 if old else 0.5),
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
    elements.append(Paragraph(
        "This report is generated by a rule-based computer-vision pipeline (HSV color analysis, not a trained "
        "neural network) and is intended as a decision-support aid only. Shape/texture-based diagnoses "
        "(Calcium, Copper, Molybdenum, Boron, Zinc) are lower-confidence than color-based ones. For high-value "
        "crops or persistent symptoms, confirm with a certified agronomist.",
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
        analysis = analyze_leaf_image(save_path, leaf_position=leaf_position)
    except Exception as exc:
        app.logger.error(f"Leaf analysis failed for {filename}: {exc}")
        flash("Please make sure the image contains a clear crop leaf and try again.", "danger")
        return redirect(url_for("detect"))

    try:
        crop_type, crop_confidence = detect_crop_type(save_path)
    except Exception:
        crop_type, crop_confidence = "Rice", 30.0

    # Manual override: if the user picked a specific crop from the dropdown
    # instead of leaving it on "Auto-detect", trust their choice -- they
    # know their own field better than a shape heuristic does.
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
        "visual_symptoms": meta.get("symptom", ""),
        **analysis,
        **treatment,
    }

    db = get_db()
    db.execute("""
        INSERT INTO predictions (
            report_id, crop_type, crop_confidence, leaf_position, image_filename, created_at, deficiency_type, confidence,
            confidence_tier, severity_level, affected_area_pct, green_pct, yellow_pct, brown_pct, purple_pct,
            white_pct, visual_symptoms, immediate_action, recommended_fertilizer, application_method,
            dosage, recovery_time, risk_level, overall_health
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record["report_id"], record["crop_type"], record["crop_confidence"], record["leaf_position"], record["image_filename"], record["created_at"],
        record["deficiency_type"], record["confidence"], record["confidence_tier"], record["severity_level"],
        record["affected_area_pct"], record["green_pct"], record["yellow_pct"], record["brown_pct"], record["purple_pct"],
        record["white_pct"], record["visual_symptoms"], record["immediate_action"], record["recommended_fertilizer"],
        record["application_method"], record["dosage"], record["recovery_time"],
        record["risk_level"], record["overall_health"]
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
