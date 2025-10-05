from flask import Flask, request, Response
import subprocess
import pandas as pd
import os

app = Flask(__name__)


@app.route('/req', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400

    file = request.files['file']

    input_path = "input_table.csv"
    output_path = "./out/predictions_outputs.csv"
    file.save(input_path)

    cmd = "./run.sh"
    try:
        subprocess.run(["bash", cmd], check=True)
    except subprocess.CalledProcessError as e:
        return {'error': f'Command failed: {e}'}, 500

    if not os.path.exists(output_path):
        return {'error': 'Output file not found'}, 500

    with open(output_path, 'r') as f:
        output_csv = f.read()


    response = Response(
        output_csv,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=results.csv'}
    )


    os.remove(input_path)
    os.remove(output_path)

    return response


if __name__ == '__main__':
    app.run(debug=True)
