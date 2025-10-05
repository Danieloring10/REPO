from flask import Flask, request, jsonify, render_template, send_file
import subprocess
import os
import json
import base64
import shutil

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "dataset.csv")
JSON_FILE_PATH = os.path.join(BASE_DIR, "params.json")
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset_default.csv")
OUTPUT_ZIP = os.path.join(BASE_DIR, "modelo_generado.zip")  # Nombre del ZIP que devolveremos


@app.route("/")
def index():
    return render_template("newPage.html")


@app.route("/train", methods=["POST"])
def train():
    data = None
    file = None

    if request.content_type and "multipart/form-data" in request.content_type:
        json_data = request.form.get("json_data")
        if json_data:
            data = json.loads(json_data)

        file = request.files.get("dataset")

    # --- Caso 2: JSON puro ---
    elif request.is_json:
        payload = request.get_json()
        data = payload.get("params")

        csv_base64 = payload.get("csv_base64")
        if csv_base64:
            with open(CSV_FILE_PATH, "wb") as f:
                f.write(base64.b64decode(csv_base64))
    else:
        return jsonify({"error": "Formato de solicitud no soportado"}), 400

    if not data:
        return jsonify({"error": "No se recibieron los parámetros"}), 400

    batch_allowed = [8, 16, 32, 64]
    tuner_allowed = ["random", "bayesian", "hyperband"]

    if data["batch_size"] not in batch_allowed:
        return jsonify({"error": "Batch size inválido"}), 400
    if data["tuner"] not in tuner_allowed:
        return jsonify({"error": "Tuner inválido"}), 400
    if not (1 <= data["epochs"] <= 100):
        return jsonify({"error": "Epoch fuera de rango (1–100)."}), 400
    if not (0.1 <= data["test_size"] <= 0.5):
        return jsonify({"error": "Test size fuera de rango (0.1–0.5)."}), 400

    with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    if file:
        file.save(CSV_FILE_PATH)
        dataset_usado = CSV_FILE_PATH
    elif not os.path.exists(CSV_FILE_PATH):
        shutil.copy(DEFAULT_DATASET, CSV_FILE_PATH)
        dataset_usado = DEFAULT_DATASET
    else:
        dataset_usado = CSV_FILE_PATH

    cmd = ["python", "GenerarModelo.py"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return {'error': f'Command failed: {e}'}, 500

    if not os.path.exists(OUTPUT_ZIP):
        return jsonify({"error": "No se generó el archivo ZIP"}), 500

    return send_file(
        OUTPUT_ZIP,
        mimetype="application/zip",
        as_attachment=True,
        download_name="modelo_generado.zip"
    )

if __name__ == "__main__":
    app.run(debug=True)
