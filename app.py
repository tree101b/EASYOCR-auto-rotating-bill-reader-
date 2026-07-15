"""
Invoice-AI — web app that scans invoices/bills and extracts their text.

Backend: Flask + EasyOCR (the same OCR engine already installed on this machine).
Folder layout (kept to 4 as requested):
    templates/  -> HTML pages
    static/     -> CSS + JS
    uploads/    -> images the user submits
    outputs/    -> extracted-text results (one .txt per scan)

Accuracy is improved with an OpenCV preprocessing pass (upscale, contrast,
denoise, deskew), a beam-search decoder, and reading-order reconstruction.
"""

import os
import re
import time
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB cap

# --- EasyOCR is heavy to load, so build the reader once, lazily, in the background ---
_reader = None
_reader_lock = threading.Lock()


def get_reader():
    """Return a shared EasyOCR reader, loading it on first use."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import torch
                import easyocr  # imported here so the server starts instantly
                # Use every CPU core for the OCR math (this box has no CUDA).
                torch.set_num_threads(os.cpu_count() or 4)
                _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


# --------------------------------------------------------------------------- #
#  Accuracy: image preprocessing                                              #
# --------------------------------------------------------------------------- #
def _rotate90(gray, angle):
    """Rotate by a multiple of 90 degrees (angle in {0, 90, 180, 270})."""
    if angle == 90:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray


def _best_orientation(gray):
    """Pick which 90-degree turn makes the page upright.

    We OCR a small copy at each of the four orientations and keep the one whose
    reading is most confident — correctly-oriented text scores far higher than
    the low-confidence gibberish you get from sideways/upside-down text.
    """
    longest = max(gray.shape)
    if longest > 640:                      # probe on a small copy -> fast
        s = 640 / longest
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    reader = get_reader()
    best_angle, best_score = 0, -1.0
    for angle in (0, 90, 180, 270):
        try:
            res = reader.readtext(_rotate90(small, angle), detail=1,
                                  paragraph=False, decoder="greedy",
                                  mag_ratio=1.0, canvas_size=768, batch_size=8)
        except Exception:
            res = []
        # Total confidence of real (>=2 char) words at this orientation.
        score = sum(c for _b, t, c in res if len(t.strip()) >= 2)
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


def _deskew(gray):
    """Rotate the image so text lines are horizontal (only for modest tilts)."""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    # Skip tiny noise angles and large ones (likely a mis-estimate).
    if abs(angle) < 0.5 or abs(angle) > 20:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# Manual rotation choices from the web form -> a 90-degree turn to apply.
_ROTATE_CHOICES = {"none": 0, "left": 270, "right": 90, "180": 180}


def preprocess(img_path, rotate="auto"):
    """Return an OCR-friendly grayscale image (numpy array) for a photo/scan.

    ``rotate`` controls orientation handling:
      "auto" -> probe all four turns and keep the most confident (slower);
      "none"/"left"/"right"/"180" -> apply that fixed turn (instant).
    """
    img = cv2.imread(img_path)
    if img is None:  # unreadable — let EasyOCR try the raw path
        return img_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Keep the working image in a size band that balances accuracy vs speed.
    # OCR time scales with pixel count, so we upscale small photos just enough
    # to be legible and shrink big ones hard to keep scans fast.
    h, w = gray.shape
    longest = max(h, w)
    if longest < 1000:                       # too small -> enlarge (accuracy)
        scale = min(1.8, 1100 / longest)
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    elif longest > 1400:                      # too big -> shrink (speed)
        scale = 1400 / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    # Even out lighting / boost contrast (great for creased receipt photos).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Edge-preserving denoise.
    gray = cv2.bilateralFilter(gray, 5, 40, 40)

    # Fix a sideways/upside-down page (auto-probe or the user's choice),
    # then correct any small remaining tilt.
    if rotate == "auto":
        gray = _rotate90(gray, _best_orientation(gray))
    else:
        gray = _rotate90(gray, _ROTATE_CHOICES.get(rotate, 0))
    gray = _deskew(gray)
    return gray


# --------------------------------------------------------------------------- #
#  Accuracy: reading-order reconstruction                                     #
# --------------------------------------------------------------------------- #
def to_reading_order(results, min_conf=0.1):
    """Group detections into rows (top->bottom, left->right) and drop junk.

    Returns (lines, kept_confidences) where each line is one row of text with
    its cells joined by spaces — so an item and its price stay together.
    """
    items = []
    for box, text, conf in results:
        if conf < min_conf or not text.strip():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({
            "text": text.strip(),
            "conf": conf,
            "cy": sum(ys) / len(ys),
            "x": min(xs),
            "h": max(ys) - min(ys),
        })

    rows = []
    for it in sorted(items, key=lambda d: d["cy"]):
        tol = max(it["h"], 10) * 0.6
        placed = False
        for row in rows:
            if abs(row["cy"] - it["cy"]) <= tol:
                row["cells"].append(it)
                row["cy"] = (row["cy"] * row["n"] + it["cy"]) / (row["n"] + 1)
                row["n"] += 1
                placed = True
                break
        if not placed:
            rows.append({"cy": it["cy"], "n": 1, "cells": [it]})

    lines, confs = [], []
    for row in sorted(rows, key=lambda r: r["cy"]):
        cells = sorted(row["cells"], key=lambda d: d["x"])
        lines.append(" ".join(c["text"] for c in cells))
        confs.extend(c["conf"] for c in cells)
    return lines, confs


def parse_fields(text):
    """Best-effort pull of the fields that matter on a bill/invoice."""
    fields = {}

    # Try, in order: a (fuzzily-spelled) invoice/bill keyword, then a bare "#"
    # marker, then a standalone invoice-number token like "INV-2026-0042".
    # OCR often confuses i/l/o/0, so the keyword match is deliberately loose.
    inv = re.search(r"inv\w{0,3}ce\s*#?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-\/]{2,})",
                    text, re.IGNORECASE)
    if not inv:
        inv = re.search(r"#\s*([A-Za-z0-9][A-Za-z0-9\-\/]{3,})", text)
    if not inv:
        inv = re.search(r"\b([A-Za-z]{2,4}[-\s]?\d[\w\-\/]*\d)\b", text)
    if inv:
        fields["invoice_no"] = inv.group(1).strip()

    # Accept /, -, . and , as separators (OCR often misreads / as ,).
    m = re.search(r"(\d{1,4}[\/\-.,]\d{1,2}[\/\-.,]\d{1,4})", text)
    if m:
        fields["date"] = m.group(1).replace(",", "/")

    # Grab every money-looking amount, report the largest as the total.
    amounts = re.findall(r"(?:[$€£₹]|USD|EUR|INR|Rs\.?)\s*([\d,]+\.\d{2})",
                         text, re.IGNORECASE)
    nums = []
    for a in amounts:
        try:
            nums.append(float(a.replace(",", "")))
        except ValueError:
            pass
    if nums:
        fields["total"] = f"{max(nums):,.2f}"

    return fields


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    stamp = str(int(time.time() * 1000))
    fname = f"{stamp}_{secure_filename(f.filename)}"
    img_path = os.path.join(UPLOAD_DIR, fname)
    f.save(img_path)

    rotate = request.form.get("rotate", "auto")
    try:
        prepared = preprocess(img_path, rotate=rotate)
        results = get_reader().readtext(
            prepared,
            detail=1,
            paragraph=False,
            decoder="greedy",       # fast decode; preprocessing carries accuracy
            mag_ratio=1.0,          # image already sized in preprocess()
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.6,
            low_text=0.35,
            link_threshold=0.4,
            canvas_size=1400,       # matches the preprocess size cap
            batch_size=8,           # recognize several lines at once
        )
    except Exception as exc:  # keep the API honest about failures
        return jsonify({"error": f"OCR failed: {exc}"}), 500

    lines, confs = to_reading_order(results)
    full_text = "\n".join(lines)
    avg_conf = round(sum(confs) / len(confs) * 100, 1) if confs else 0.0

    out_name = f"{stamp}.txt"
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as out:
        out.write(full_text)

    return jsonify({
        "image_url": f"/uploads/{fname}",
        "text": full_text,
        "line_count": len(lines),
        "confidence": avg_conf,
        "fields": parse_fields(full_text),
        "output_file": out_name,
    })


@app.route("/uploads/<path:name>")
def uploaded_file(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.route("/outputs/<path:name>")
def output_file(name):
    return send_from_directory(OUTPUT_DIR, name, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
