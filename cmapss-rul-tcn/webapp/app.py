"""
Flask application for C-MAPSS FD001 RUL Prognostics & IIoT Priority Scheduler.

Routes:
    GET  /        – Upload form (index.html)
    POST /predict – Run inference pipeline, render results dashboard (results.html)
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Ensure project root is importable so src.* and webapp.* resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from webapp.inference import predict_pipeline

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "local-development-only")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the upload / landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload, validate, run inference, and render results."""
    # --- File presence check ---
    file = request.files.get("file")
    if file is None or file.filename == "":
        flash("Please select a valid C-MAPSS text file to upload.", "danger")
        return redirect(url_for("index"))

    # --- Dataset Selection (FD001, FD002, FD003, FD004) ---
    dataset = request.form.get("dataset", "FD001")
    if dataset not in ["FD001", "FD002", "FD003", "FD004"]:
        dataset = "FD001"

    # --- Bandwidth limit from form (default 10) ---
    try:
        bandwidth_limit = int(request.form.get("bandwidth_limit", 10))
        if bandwidth_limit < 1:
            bandwidth_limit = 10
    except (ValueError, TypeError):
        bandwidth_limit = 10

    # --- Save uploaded file ---
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        flash("Please select a valid C-MAPSS text file to upload.", "danger")
        return redirect(url_for("index"))

    upload_dir = Path(app.instance_path) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}_{safe_filename}"
    file.save(str(file_path))

    # --- Basic validation: read raw lines and check format ---
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            flash("The uploaded file is empty.", "danger")
            return redirect(url_for("index"))

        # Check column count on a sample of lines
        for line_idx, line in enumerate(lines[:5]):
            parts = line.strip().split()
            if len(parts) != 26:
                flash(
                    f"Format error on line {line_idx + 1}: expected 26 whitespace-separated columns, "
                    f"found {len(parts)}. Please upload a standard C-MAPSS file.",
                    "danger",
                )
                return redirect(url_for("index"))

            # Check that all values are numeric
            for col_idx, val in enumerate(parts):
                try:
                    float(val)
                except ValueError:
                    flash(
                        f"Non-numeric value '{val}' at line {line_idx + 1}, column {col_idx + 1}. "
                        f"C-MAPSS files must contain only numeric data.",
                        "danger",
                    )
                    return redirect(url_for("index"))

    except Exception as exc:
        flash(f"Could not read the uploaded file: {exc}", "danger")
        return redirect(url_for("index"))

    # --- Run inference pipeline ---
    try:
        result = predict_pipeline(str(file_path), dataset=dataset, bandwidth_limit=bandwidth_limit)
    except Exception as exc:
        flash(f"Inference failed: {exc}", "danger")
        return redirect(url_for("index"))

    return render_template("results.html", result=result)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )

