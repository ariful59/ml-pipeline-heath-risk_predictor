import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from src.pipeline.predict_pipeline import PredictPipeline


app = FastAPI(
    title="Health Risk Predictor",
    version="1.0.0",
    description="Web UI and API for health risk prediction",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
PREDICT_PIPELINE = PredictPipeline()


def normalize_prediction(pred) -> str:
    try:
        return RISK_LABELS[int(pred)]
    except Exception:
        return str(pred)


class PredictionRequest(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    Gender: str
    Symptoms: str = ""
    Medical_History: str = ""
    Medications: str = ""
    Lab_Reports: str = ""
    Lifestyle: str = ""
    Doctor_Notes: str = ""
    Diagnosis: str = ""
    Patient_ID: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": None,
            "form": {},
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    symptoms: str = Form(""),
    medical_history: str = Form(""),
    medications: str = Form(""),
    lab_reports: str = Form(""),
    lifestyle: str = Form(""),
    doctor_notes: str = Form(""),
    diagnosis: str = Form(""),
):
    try:
        input_data = {
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
            "Medical_History": medical_history,
            "Medications": medications,
            "Lab_Reports": lab_reports,
            "Lifestyle": lifestyle,
            "Doctor_Notes": doctor_notes,
            "Diagnosis": diagnosis,
        }

        prediction = PREDICT_PIPELINE.predict(input_data)
        result = normalize_prediction(prediction)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "result": result,
                "form": {
                    "age": age,
                    "gender": gender,
                    "symptoms": symptoms,
                    "medical_history": medical_history,
                    "medications": medications,
                    "lab_reports": lab_reports,
                    "lifestyle": lifestyle,
                    "doctor_notes": doctor_notes,
                    "diagnosis": diagnosis,
                },
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict_api(payload: PredictionRequest):
    try:
        prediction = PREDICT_PIPELINE.predict(payload.dict())
        result = normalize_prediction(prediction)
        return {
            "success": True,
            "prediction": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
