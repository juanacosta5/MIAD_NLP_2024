#-------------------------------------------------------------------------
#-------------------- Importacion de librerias ---------------------------
#-------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from math import sqrt
from flask import Flask, request, jsonify
#-------------------------------------------------------------------------
#------------------------ Cargar base de datos ---------------------------
#-------------------------------------------------------------------------

dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')

#-------------------------------------------------------------------------
#------------------------ Se hace el encoder -----------------------------
#-------------------------------------------------------------------------
for col in ['State', 'Make', 'Model']:
    le = LabelEncoder()
    dataTraining[col] = le.fit_transform(dataTraining[col])

#-------------------------------------------------------------------------
#------------------------ Se separa la BD --------------------------------
#-------------------------------------------------------------------------
#Separar
X = dataTraining.drop('Price', axis=1)
y = dataTraining['Price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

#-------------------------------------------------------------------------
#------------------------ Se estandariza  --------------------------------
#-------------------------------------------------------------------------

cols_to_standardize = ['State', 'Make', 'Model']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_train_scaled[cols_to_standardize] = scaler.fit_transform(X_train_scaled[cols_to_standardize])

X_val_scaled = X_val.copy() 
X_val_scaled[cols_to_standardize] = scaler.transform(X_val_scaled[cols_to_standardize])

#-------------------------------------------------------------------------
#------------------------ Modelo ensamble --------------------------------
#-------------------------------------------------------------------------

# Define los hiperparámetros para XGBoost y RandomForest
xgb_params = {'n_estimators': 641,
              'max_depth': 10,
              'learning_rate': 0.05064281254955496,
              'subsample': 0.9668967113742012,
              'colsample_bytree': 0.8712694851316382,
              'min_child_weight': 1,
              'reg_lambda': 4,
              'reg_alpha': 6}

rf_params = {'n_estimators': 186,
             'max_depth': 19,
             'min_samples_split': 19,
             'min_samples_leaf': 2}

# Crea los estimadores base
xgb_estimator = ('xgboost', XGBRegressor(**xgb_params))
rf_estimator = ('random_forest', RandomForestRegressor(**rf_params))

# Crea el modelo de stacking
stacking_model = StackingRegressor(
    estimators=[xgb_estimator, rf_estimator],
    final_estimator=LinearRegression()
)
# Entrena el modelo de stacking
stacking_model.fit(X_train_scaled, y_train) 

#-------------------------------------------------------------------------
#--------------------------- Creacion API --------------------------------
#-------------------------------------------------------------------------

app = Flask(__name__)

# Carga el modelo previamente entrenado
stacking_model = None
scaler = None
le = None

@app.before_first_request
def load_model():
    global stacking_model, scaler, le
    # Cargar el modelo de stacking solo dentro de esta función
    stacking_model = StackingRegressor(
        estimators=[('xgboost', XGBRegressor(**xgb_params)),
                    ('random_forest', RandomForestRegressor(**rf_params))],
        final_estimator=LinearRegression()
    )
    
    # Cargar el codificador de etiquetas
    le = LabelEncoder()
    
    # Cargar el escalador
    scaler = StandardScaler()
    
    # (Opcional) Cargar datos adicionales o configuraciones necesarias

# Endpoint para predecir el precio
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de la solicitud
        data = request.json
        
        # Convertir los datos en un DataFrame
        df = pd.DataFrame(data)
        
        # Codificar las variables categóricas
        for col in ['State', 'Make', 'Model']:
            df[col] = le.transform(df[col])
        
        # Escalar las características
        cols_to_standardize = ['State', 'Make', 'Model']
        df_scaled = df.copy()
        df_scaled[cols_to_standardize] = scaler.transform(df_scaled[cols_to_standardize])
        
        # Realizar la predicción
        prediction = stacking_model.predict(df_scaled)
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)