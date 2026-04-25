"""
=============================================================================
app.py  -  Flask Web Application for the Plagiarism Detection System
=============================================================================
Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
=============================================================================
"""

import io
import os
import sys
import base64
import uuid

import matplotlib
matplotlib.use("Agg")          # headless backend – no display needed

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request

# Make sure the detector module is importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from plagiarism_detector import PlagiarismDetector

# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload limit
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"txt", "md"}

# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(file_storage) -> str:
    """Read uploaded file content as a UTF-8 string."""
    raw = file_storage.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def heatmap_to_base64(similarity_matrix: np.ndarray, document_names: list) -> str:
    """Render heatmap to a PNG and return it as a base64 data URI."""
    import seaborn as sns

    sim_pct = similarity_matrix * 100
    n = len(document_names)
    fig_size = max(8, n * 1.4)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Mask the diagonal so it does not skew the colour scale
    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        sim_pct,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        mask=mask,
        xticklabels=document_names,
        yticklabels=document_names,
        linewidths=0.6,
        linecolor="#1e1e3a",
        cbar_kws={"label": "Similarity (%)", "shrink": 0.75},
        ax=ax,
        vmin=0,
        vmax=100,
        annot_kws={"size": 10, "weight": "bold"},
    )

    # Paint the diagonal cells a neutral colour
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=True, color="#2a2a4a", zorder=2
        ))
        ax.text(
            i + 0.5, i + 0.5, "—",
            ha="center", va="center",
            fontsize=10, color="#888", zorder=3
        )

    ax.set_title(
        "Document Similarity Matrix  (TF-IDF + Cosine Similarity)",
        fontsize=13, fontweight="bold", color="white", pad=16
    )
    ax.tick_params(colors="white", labelsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", color="white")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="white")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ===========================================================================
# Routes
# ===========================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart/form-data with:
      - files[]   : one or more uploaded text files
      - threshold : float 0-100 (the plagiarism percentage threshold)
    Returns JSON with the full analysis result.
    """
    files = request.files.getlist("files[]")
    threshold = float(request.form.get("threshold", 70)) / 100.0

    # ── Validate ──────────────────────────────────────────────────────────
    if len(files) < 2:
        return jsonify({"error": "Please upload at least 2 documents."}), 400

    docs, names = [], []
    for f in files:
        if not f or f.filename == "":
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"Unsupported file type: {f.filename}. Only .txt and .md files are allowed."}), 400
        content = read_file(f)
        if len(content.strip()) < 20:
            return jsonify({"error": f"File '{f.filename}' appears to be empty or too short."}), 400
        docs.append(content)
        names.append(os.path.splitext(f.filename)[0])   # strip extension for display

    if len(docs) < 2:
        return jsonify({"error": "Could not read enough valid documents."}), 400

    # ── Run Detection ─────────────────────────────────────────────────────
    detector = PlagiarismDetector(threshold=threshold)
    sim_matrix = detector.calculate_similarity(docs)
    sim_df     = detector.get_similarity_dataframe(sim_matrix, names)
    flagged    = detector.detect_plagiarism(names, sim_matrix)

    # ── Build matrix payload ──────────────────────────────────────────────
    matrix_rows = []
    for i, row_name in enumerate(names):
        cells = []
        for j, col_name in enumerate(names):
            score = round(float(sim_matrix[i][j]) * 100, 1)
            cells.append({
                "score":     score,
                "is_self":   i == j,
                "is_flagged": i != j and score >= threshold * 100
            })
        matrix_rows.append({"name": row_name, "cells": cells})

    # ── Sentence comparison for top flagged pair ──────────────────────────
    sentence_results = []
    if flagged:
        top = flagged[0]
        i1  = names.index(top["doc1"])
        i2  = names.index(top["doc2"])
        sent_df = detector.sentence_level_comparison(
            docs[i1], docs[i2], sentence_threshold=0.50
        )
        if not sent_df.empty:
            sentence_results = sent_df.head(10).to_dict(orient="records")

    # ── Heatmap ───────────────────────────────────────────────────────────
    heatmap_b64 = heatmap_to_base64(sim_matrix, names)

    # ── Stats ─────────────────────────────────────────────────────────────
    n           = len(names)
    total_pairs = n * (n - 1) // 2
    scores_off_diag = [
        float(sim_matrix[i][j]) * 100
        for i in range(n) for j in range(i + 1, n)
    ]
    avg_score = round(sum(scores_off_diag) / len(scores_off_diag), 1) if scores_off_diag else 0
    max_score = round(max(scores_off_diag), 1) if scores_off_diag else 0

    return jsonify({
        "names":            names,
        "matrix":           matrix_rows,
        "flagged":          flagged,
        "sentence_results": sentence_results,
        "heatmap":          heatmap_b64,
        "stats": {
            "total_docs":    n,
            "total_pairs":   total_pairs,
            "flagged_count": len(flagged),
            "clean_count":   total_pairs - len(flagged),
            "threshold_pct": round(threshold * 100, 0),
            "avg_score":     avg_score,
            "max_score":     max_score,
        },
        "top_pair": flagged[0] if flagged else None,
    })


# ===========================================================================
if __name__ == "__main__":
    print("\n  Plagiarism Detection Web App")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
