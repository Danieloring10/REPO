from flask import Flask, render_template, request, Response, jsonify
import pandas as pd
import requests
import io

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if file:
            df = pd.read_csv(file)
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data received"}), 400
            df = pd.DataFrame(data)

        # Convertir a CSV temporal
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Enviar al modelo remoto
        model_url = "https://narrowly-nacreous-fernanda.ngrok-free.dev/"
        response = requests.post(
            model_url,
            files={'file': ('input.csv', csv_buffer.getvalue(), 'text/csv')},
            timeout=600  # Esperar hasta 10 minutos
        )

        if response.status_code != 200:
            return jsonify({
                "error": f"Model server error {response.status_code}",
                "details": response.text
            }), 500
        print("Response from remote model:\n", response.text[:1000])
        return Response(
            response.text,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=results.csv"}
        )

    except requests.Timeout:
        return jsonify({"error": "The model server took too long to respond"}), 504
    except Exception as e:
        print("⚠️ Server error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
