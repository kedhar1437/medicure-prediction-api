from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from pathlib import Path

# Base setup
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load model (enhanced joblib)
MODEL_PATH = BASE_DIR / "disease_model_enhanced.joblib"
print("Loading model from:", MODEL_PATH)

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
features = model_data["features"]
classes = model_data.get("classes", [])
symptom_choices = sorted([s.replace("_", " ") for s in features])
disease_info = model_data.get("disease_info", {})
symptom_descriptions = model_data.get("symptom_descriptions", {})


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render home page with symptom selection form"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "symptom_choices": symptom_choices
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    symptom1: str = Form(...),
    symptom2: str = Form(None),
    symptom3: str = Form(None),
    symptom4: str = Form(None),
    symptom5: str = Form(None)
):
    try:
        # Get selected symptoms
        selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s]

        # Prepare input data (one-hot encoding of selected symptoms)
        symptom_values = {s.replace(" ", "_"): 1 for s in selected_symptoms}
        input_data = pd.DataFrame([{f: symptom_values.get(f, 0) for f in features}])

        # Predict probabilities
        probabilities = model.predict_proba(input_data)[0]
        results = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:5]

        # Format predictions
        predictions = []
        for disease, prob in results:
            score = int(prob * 100)

            if score >= 80:
                confidence = "Very High"
            elif score >= 60:
                confidence = "High"
            elif score >= 40:
                confidence = "Moderate"
            else:
                confidence = "Low"

            predictions.append({
                "disease": disease,
                "description": disease_info.get(disease, "No description available"),
                "probability": f"{prob*100:.1f}%",
                "confidence": confidence,
                "score": score,
                "score_color": "green" if score >= 60 else "orange" if score >= 40 else "red"
            })

        return templates.TemplateResponse("result.html", {
            "request": request,
            "predictions": predictions,
            "selected_symptoms": selected_symptoms
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="info")
