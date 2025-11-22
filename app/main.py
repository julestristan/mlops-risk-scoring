# app/main.py
import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from schemas.input import CreditFeatures
import os
app = FastAPI(title="API de Scoring de Risque Crédit")
# Le chemin est relatif au dossier 'app' dans le conteneur Docker
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# Chargement du modèle/pipeline au démarrage de l'API
try:
    model = joblib.load(MODEL_PATH)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"ATTENTION : Erreur lors du chargement du modèle : {e}")
    model = None

@app.get("/")
def home():
    return {"message": "API de Scoring de Risque en ligne. Consultez /docs."}

@app.post("/predict")
def predict_risk(features: CreditFeatures):
    """Calcule la probabilité de risque de défaut de crédit (classe 1)."""
    if model is None:
         return {"error": "Modèle non disponible."}
         
    try:
        # Convertit les données Pydantic en DataFrame
        data_point = features.model_dump()
        df = pd.DataFrame([data_point])

        # Le modèle (pipeline) applique le pré-traitement puis fait la prédiction
        # prediction_proba[1] est la probabilité de la classe 1 (Risque/Bad)
        prediction_proba = model.predict_proba(df)[0]
        risk_score = prediction_proba[1] 

        return {
            "prediction_proba_default": float(risk_score),
            "risk_label": "Haut Risque" if risk_score > 0.5 else "Faible Risque",
        }

    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {e}"}