"""
main.py
--------
FastAPI microservice for the LLM-Powered Clinical Decision Support System.
Serves real-time patient risk alerts via REST API.

Endpoints:
  GET  /health                     - Health check
  GET  /patients/{patient_id}      - Get patient profile
  POST /risk-alert                 - Get risk alert for a patient
  POST /batch-evaluate             - Evaluate a batch of patients
  GET  /metrics                    - Latest pipeline metrics

Author: Jahnav Jayanth Reddy Kukkala
"""

import json
import sys
import os
import time
from typing import List, Optional

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from pipelines.embeddings import ClinicalNotesIndex
from pipelines.rag_pipeline import ClinicalRAGPipeline
from tracking.mlflow_tracking import ExperimentTracker, PipelineEvaluator


# ── Load data + build pipeline on startup ─────────────────────────────────────

def load_pipeline():
    index = ClinicalNotesIndex().build("data/clinical_notes.json")
    pipeline = ClinicalRAGPipeline(notes_index=index)
    with open("data/patients.json") as f:
        patients = json.load(f)
    patient_map = {p["patient_id"]: p for p in patients}
    tracker = ExperimentTracker("clinical-decision-support")
    return pipeline, patient_map, tracker


# ── Pydantic models ────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class RiskAlertRequest(BaseModel):
        patient_id: str
        top_k: Optional[int] = 3

    class BatchEvaluateRequest(BaseModel):
        patient_ids: List[str]
        top_k: Optional[int] = 3

    class InlinePatientRequest(BaseModel):
        """For ad-hoc queries without a stored patient_id."""
        age: int
        gender: str
        diagnosis: str
        medications: List[str] = []
        allergies: List[str] = []
        prev_admissions: int = 0
        comorbidities: int = 0
        vitals: dict = {}

    # ── App ─────────────────────────────────────────────────────────────────────

    app = FastAPI(
        title="Clinical Decision Support API",
        description=(
            "LLM-Powered Clinical Decision Support System. "
            "Uses RAG over clinical notes to generate grounded patient risk alerts."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load on startup
    pipeline, patient_map, tracker = load_pipeline()
    evaluator = PipelineEvaluator(tracker)
    _latest_metrics: dict = {}

    # ── Routes ──────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {
            "status":       "healthy",
            "patients_loaded": len(patient_map),
            "model":        "rag-clinical-v1",
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    @app.get("/patients/{patient_id}")
    def get_patient(patient_id: str):
        patient = patient_map.get(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
        return patient

    @app.post("/risk-alert")
    def get_risk_alert(req: RiskAlertRequest):
        """
        Main endpoint — generates a clinical risk alert for a stored patient.
        Retrieves relevant historical context via RAG and returns a grounded recommendation.
        """
        patient = patient_map.get(req.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {req.patient_id} not found.")

        result = pipeline.run(patient, top_k=req.top_k)
        return result

    @app.post("/risk-alert/inline")
    def get_risk_alert_inline(req: InlinePatientRequest):
        """
        Ad-hoc risk alert for a patient profile provided directly in the request.
        Useful for real-time queries without pre-stored records.
        """
        import math
        log_odds = (-3.2 + 0.03 * req.age + 0.28 * req.prev_admissions
                    + 0.18 * req.comorbidities + 0.09 * len(req.medications))
        risk_score = round(1 / (1 + math.exp(-log_odds)), 4)
        risk_tier = "HIGH" if risk_score >= 0.4 else ("MEDIUM" if risk_score >= 0.2 else "LOW")

        patient = {
            "patient_id":      "INLINE-001",
            "age":             req.age,
            "gender":          req.gender,
            "diagnosis":       req.diagnosis,
            "medications":     req.medications,
            "allergies":       req.allergies,
            "prev_admissions": req.prev_admissions,
            "comorbidities":   req.comorbidities,
            "vitals":          req.vitals,
            "risk_score":      risk_score,
            "risk_tier":       risk_tier,
        }
        return pipeline.run(patient)

    @app.post("/batch-evaluate")
    def batch_evaluate(req: BatchEvaluateRequest):
        """Evaluate a batch of patients and return aggregated metrics."""
        patients = []
        missing = []
        for pid in req.patient_ids:
            p = patient_map.get(pid)
            if p:
                patients.append(p)
            else:
                missing.append(pid)

        if not patients:
            raise HTTPException(status_code=404, detail="No valid patients found.")

        results = [pipeline.run(p, top_k=req.top_k) for p in patients]
        summary = evaluator.evaluate_batch(results, run_name="api_batch_eval")

        global _latest_metrics
        _latest_metrics = summary

        return {"summary": summary, "missing_patients": missing, "results": results}

    @app.get("/metrics")
    def get_metrics():
        """Return latest pipeline evaluation metrics."""
        if not _latest_metrics:
            return {"message": "No evaluations run yet. Call /batch-evaluate first."}
        return _latest_metrics

    @app.get("/high-risk-patients")
    def get_high_risk_patients(limit: int = 20):
        """Return top N highest risk patients sorted by risk score."""
        sorted_patients = sorted(
            patient_map.values(),
            key=lambda p: p["risk_score"],
            reverse=True
        )[:limit]
        return {"count": len(sorted_patients), "patients": sorted_patients}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        # Run pipeline directly without FastAPI for testing
        print("FastAPI not installed. Running pipeline demo directly.\n")
        pipeline, patient_map, tracker = load_pipeline()

        with open("data/patients.json") as f:
            patients = json.load(f)

        high_risk = [p for p in patients if p["risk_tier"] == "HIGH"][:3]
        print(f"Running RAG pipeline on {len(high_risk)} high-risk patients...\n")

        for patient in high_risk:
            result = pipeline.run(patient)
            print(f"{'─'*55}")
            print(f"Patient : {result['patient_id']} | {result['diagnosis']}")
            print(f"Risk    : {result['risk_tier']} (score={result['risk_score']})")
            print(f"Latency : {result['latency_ms']}ms")
            print(f"Rec     : {result['recommendation'][:120]}...")
            if result["alerts"]:
                for a in result["alerts"]:
                    print(f"Alert   : {a}")
        print(f"{'─'*55}")
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
