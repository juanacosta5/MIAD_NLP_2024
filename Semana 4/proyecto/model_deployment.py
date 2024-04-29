from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Cargar el modelo al inicio de la aplicación
clf = joblib.load(r'C:\Users\jacosta\Documents\GitHub\Mis Repos\MIAD_NLP_2024\Semana 4\proyecto\stacking_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos de la solicitud como JSON
        data = request.get_json()

        # Convertir a DataFrame
        input_data = pd.DataFrame([data])

        # Realizar predicción
        prediction = clf.predict(input_data)

        # Devolver resultado como JSON
        return jsonify({'prediction': prediction.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
