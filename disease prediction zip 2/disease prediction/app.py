from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import List
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Medicure AI Diagnosis API")

# filenames expected in same folder
MODEL_PATH = BASE_DIR / "disease_model_enhanced.joblib"
CSV_PATH = BASE_DIR / "disease_symptoms_extended.csv"

print("MODEL_PATH:", MODEL_PATH)
print("CSV_PATH:", CSV_PATH)

# Load model and feature list at startup
model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
# features are CSV columns except 'disease'
FEATURES = [c for c in df.columns if c.lower() != "disease"]


# utility: normalize a user symptom string so it matches feature names
def normalize_symptom(sym: str) -> str:
    return sym.strip().lower().replace(" ", "_").replace("-", "_")


class SymptomsIn(BaseModel):
    symptoms: List[str]


# ✅ Root route for browser testing
@app.get("/")
def read_root():
    return {
        "message": "Welcome to MediCure Prediction API",
        "endpoints": ["/api/symptoms", "/api/predict"],
    }


@app.get("/api/symptoms")
def get_symptoms():
    """Return readable symptom list for the Android app (spaces instead of underscores)."""
    readable = [f.replace("_", " ") for f in FEATURES]
    return {"symptoms": readable}


@app.post("/api/predict")
def predict(req: SymptomsIn):
    """Accepts JSON {"symptoms": [...]} and returns a list of formatted predictions.

    Returns: {"predictions": ["Disease - 95.3%", ...]}.
    """
    selected_norm = set(normalize_symptom(s) for s in req.symptoms)

    # build feature vector in same order as FEATURES
    x = [[1 if feat in selected_norm else 0 for feat in FEATURES]]

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            classes = model.classes_
            preds = sorted(
                [
                    (cls, round(prob * 100, 1))
                    for cls, prob in zip(classes, probs)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            formatted = [f"{cls} - {prob}%" for cls, prob in preds if prob > 0]
            return {"predictions": formatted}
        else:
            pred = model.predict(x)[0]
            return {"predictions": [pred]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
