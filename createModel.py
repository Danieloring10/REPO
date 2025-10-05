from flask import Flask, request, jsonify, render_template
import os
import json
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    """
    Recibe por POST:
      - Un archivo CSV (opcional)
      - Un JSON con parámetros del entrenamiento (json_data)
    Todo dentro del mismo FormData.
    """
    # --- Recuperar los datos del JSON ---
    json_data = request.form.get("json_data")
    if not json_data:
        return jsonify({"error": "No se recibió la información del JSON"}), 400

    data = json.loads(json_data)

    # --- Validar los parámetros ---
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

    # --- Procesar el archivo CSV (si se envió) ---
    file = request.files.get("dataset")
    if file:
        dataset_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(dataset_path)

        # Leer el CSV directamente (puedes usar pandas, numpy, etc.)
        df = pd.read_csv(dataset_path)
        dataset_info = {
            "filas": len(df),
            "columnas": list(df.columns)
        }
    else:
        dataset_info = "Se usará el dataset por defecto en el servidor."

    # --- Simular respuesta del entrenamiento ---
    response = {
        "mensaje": "✅ Datos recibidos correctamente.",
        "parametros": data,
        "dataset_info": dataset_info
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
