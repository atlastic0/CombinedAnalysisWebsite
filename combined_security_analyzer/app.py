from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
from pathlib import Path
import threading
import joblib
from catboost import CatBoostClassifier

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

# Saved smishing model assets. Keep both files beside app.py.
MODEL_PATH = BASE_DIR / "smishing_model.cbm"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
model = None
vectorizer = None
training_error = None
model_lock = threading.Lock()


def load_smishing_model():
    """Load the saved weights; do not retrain during normal app startup."""
    global model, vectorizer, training_error
    with model_lock:
        if model is not None:
            return
        try:
            if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
                raise FileNotFoundError(
                    "Saved smishing model files are missing. Place smishing_model.cbm "
                    "and tfidf_vectorizer.joblib beside app.py."
                )
            loaded_model = CatBoostClassifier()
            loaded_model.load_model(str(MODEL_PATH))
            loaded_vectorizer = joblib.load(VECTORIZER_PATH)
            model, vectorizer = loaded_model, loaded_vectorizer
            training_error = None
            print("Loaded saved smishing model and TF-IDF vectorizer.")
        except Exception as exc:
            model = vectorizer = None
            training_error = str(exc)


def analyze_csv(file_bytes, filename):
    text = file_bytes.decode("latin-1", errors="replace")
    data = pd.read_csv(StringIO(text), encoding="latin-1", on_bad_lines="skip")
    if len(data.columns) == 0:
        raise ValueError("The CSV does not contain any columns.")
    fields = [str(c) for c in data.columns]
    missing_by_column = data.isna().sum().to_dict()
    total_missing = int(data.isna().sum().sum())
    duplicate_values = {str(c): int(data[c].duplicated().sum()) for c in data.columns}
    duplicate_rows = int(data.duplicated().sum())

    labels = set()
    for col in data.columns:
        labels.update(data[col].dropna().astype(str).head(10000).tolist())
    detected_labels = sorted(labels) if len(labels) < 10 else []

    columns = []
    for col in data.columns:
        series = data[col]
        columns.append({
            "name": str(col), "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "duplicate_values": int(series.duplicated().sum())
        })

    preview_df = data.head(25).copy().where(pd.notna(data.head(25)), None)
    preview = []
    for row in preview_df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                value = None
            clean[str(key)] = value
        preview.append(clean)

    recommendations = []
    for col in data.columns:
        missing = int(data[col].isna().sum())
        if len(data) and missing:
            pct = missing / len(data) * 100
            recommendations.append({
                "type": "warning" if pct >= 50 else "info",
                "title": f"High missing data: {col}" if pct >= 50 else f"Missing values: {col}",
                "message": (
                    f"{pct:.1f}% of this column is missing. Consider removing it "
                    "or finding a reliable way to fill the missing values."
                    if pct >= 50 else
                    f"{missing:,} values are missing ({pct:.1f}%). Consider filling or removing them."
                )
            })
    if duplicate_rows:
        recommendations.append({
            "type": "warning", "title": "Duplicate rows detected",
            "message": f"{duplicate_rows:,} exact duplicate rows were found. Consider removing duplicate rows."
        })
    for col in data.columns:
        name = str(col).lower()
        if any(word in name for word in ["label", "class", "target", "category"]):
            continue
        if not pd.api.types.is_numeric_dtype(data[col]):
            count = int(data[col].duplicated().sum())
            if count:
                recommendations.append({
                    "type": "info", "title": f"Repeated values: {col}",
                    "message": f"{count:,} repeated values were found in this non-numeric column. Check whether they are expected."
                })
    for col in data.columns:
        if len(data) and data[col].isna().all():
            recommendations.append({
                "type": "warning", "title": f"Empty column: {col}",
                "message": "This entire column is empty. Consider removing it."
            })
    if not recommendations:
        recommendations.append({
            "type": "success", "title": "Dataset looks clean",
            "message": "No missing values, duplicate rows, or obvious data-quality problems were detected."
        })

    # Heuristic suggestions only; these do not prove which field is the correct target.
    target_suggestions = []
    row_count = max(len(data), 1)
    for col in data.columns:
        unique = int(data[col].nunique(dropna=True))
        if unique < 2 or unique > min(50, max(10, int(row_count * 0.5))):
            continue
        name = str(col).lower()
        score = 0
        if any(token in name for token in ("target", "label", "class", "outcome", "category")):
            score += 5
        if unique <= 10:
            score += 2
        if not pd.api.types.is_numeric_dtype(data[col]):
            score += 1
        target_suggestions.append({"column": str(col), "unique_values": unique, "score": score})
    target_suggestions.sort(key=lambda x: (-x["score"], x["unique_values"], x["column"]))

    return {
        "filename": filename, "rows": int(len(data)), "columns": int(len(fields)),
        "fields": fields, "total_entries": int(data.size), "total_missing": total_missing,
        "missing_by_column": {str(k): int(v) for k, v in missing_by_column.items()},
        "duplicate_rows": duplicate_rows, "duplicate_values": duplicate_values,
        "detected_labels": detected_labels, "label_count": len(labels),
        "columns_info": columns, "preview": preview,
        "recommendations": recommendations, "target_suggestions": target_suggestions
    }


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "The CSV exceeds the 150 MB upload limit."}), 413


@app.get("/")
def home():
    return render_template("home.html")


@app.get("/smishing")
def smishing_page():
    return render_template("smishing.html")


@app.get("/dataset-analyzer")
def dataset_page():
    return render_template("dataset.html")


@app.get("/api/status")
def model_status():
    if model is None and training_error is None:
        load_smishing_model()
    if training_error:
        return jsonify({"ready": False, "error": training_error}), 500
    return jsonify({"ready": model is not None})


@app.post("/api/analyze")
def analyze_message():
    if model is None and training_error is None:
        load_smishing_model()
    if training_error or model is None:
        return jsonify({"error": training_error or "The smishing model is not ready."}), 500
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("text", "")).strip()
    if not message:
        return jsonify({"error": "Paste a text message before analyzing it."}), 400
    if len(message) > 10000:
        return jsonify({"error": "The message is too long. Keep it under 10,000 characters."}), 400
    try:
        features = vectorizer.transform([message])
        prediction = int(model.predict(features).flatten()[0])
        probabilities = model.predict_proba(features)[0]
        if prediction == 1:
            classification, confidence = "Smishing", float(probabilities[1])
            explanation = "The model found patterns more consistent with smishing messages."
        else:
            classification, confidence = "Likely safe", float(probabilities[0])
            explanation = "The model found patterns more consistent with safe messages."
        return jsonify({
            "classification": classification, "confidence": round(confidence * 100, 2),
            "explanation": explanation,
            "warning": "This is a machine-learning prediction, not a guarantee. Do not click suspicious links or share sensitive information based only on this result."
        })
    except Exception as exc:
        return jsonify({"error": f"Could not analyze this message: {type(exc).__name__}: {exc}"}), 500


@app.post("/analyze")
def analyze_dataset():
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Please choose a CSV file."}), 400
    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "This analyzer accepts CSV files only."}), 400
    try:
        return jsonify(analyze_csv(uploaded.read(), uploaded.filename))
    except Exception as exc:
        return jsonify({"error": f"Could not analyze the file: {type(exc).__name__}: {exc}"}), 400

@app.route("/visualize", methods=["POST"])
def visualize():
    uploaded = request.files.get("dataset")
    target = request.form.get("target")

    if not uploaded or not target:
        return jsonify({
            "error": "Please select a CSV file and target column."
        }), 400

    try:
        data = pd.read_csv(
            uploaded,
            encoding="latin-1",
            on_bad_lines="skip"
        )

        if target not in data.columns:
            return jsonify({
                "error": "The selected target column was not found."
            }), 400

        # Chart 1: Count each target value, including missing values.
        counts = data[target].astype("object").fillna("(Missing)").value_counts()

        distribution = [
            {"label": str(value), "count": int(count)}
            for value, count in counts.head(20).items()
        ]

        # Encode a binary target as 0/1 for correlation.
        target_values = data[target]
        numeric_target = None

        if pd.api.types.is_numeric_dtype(target_values):
            numeric_target = pd.to_numeric(target_values, errors="coerce")
        elif target_values.nunique(dropna=True) == 2:
            classes = list(target_values.dropna().unique())
            numeric_target = target_values.map({
                classes[0]: 0,
                classes[1]: 1
            })

        correlations = []

        # Chart 2: Correlate numeric features with the target.
        if numeric_target is not None:
            for col in data.select_dtypes(include="number").columns:
                if col == target:
                    continue

                pair = pd.DataFrame({
                    "feature": data[col],
                    "target": numeric_target
                }).dropna()

                if (
                    len(pair) < 2
                    or pair["feature"].nunique() < 2
                    or pair["target"].nunique() < 2
                ):
                    continue

                corr = pair["feature"].corr(pair["target"])

                if pd.notna(corr):
                    correlations.append({
                        "feature": str(col),
                        "correlation": round(float(corr), 4)
                    })

        correlations.sort(
            key=lambda item: abs(item["correlation"]),
            reverse=True
        )

        return jsonify({
            "target": str(target),
            "distribution": distribution,
            "correlations": correlations[:15],
            "correlation_supported": (
                numeric_target is not None
                and numeric_target.nunique() == 2
            )
        })

    except Exception as exc:
        return jsonify({
            "error": f"Visualization failed: {type(exc).__name__}: {exc}"
        }), 400

    
@app.post("/fix")
def fix_dataset():
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename or not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please provide a valid CSV file."}), 400
    try:
        data = pd.read_csv(uploaded, encoding="latin-1", on_bad_lines="skip")
        fixed = data.drop_duplicates().copy()
        output = BytesIO()
        fixed.to_csv(output, index=False)
        output.seek(0)
        safe_name = Path(uploaded.filename).stem + "_fixed.csv"
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name=safe_name)
    except Exception as exc:
        return jsonify({"error": f"Could not fix the dataset: {type(exc).__name__}: {exc}"}), 400


if __name__ == "__main__":
    load_smishing_model()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
