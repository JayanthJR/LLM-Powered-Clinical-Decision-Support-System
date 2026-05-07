"""
rag_pipeline.py
----------------
RAG (Retrieval-Augmented Generation) pipeline for clinical decision support.
Retrieves relevant patient history from the vector store, constructs a grounded
prompt, and calls the LLM to generate clinical recommendations.

In production: uses OpenAI / Anthropic API.
In this repo:  uses a rule-based mock LLM so the pipeline runs without API keys.

Author: Jahnav Jayanth Reddy Kukkala
"""

import json
import time
from typing import Dict, List, Optional
from pipelines.embeddings import ClinicalNotesIndex


# ── Mock LLM (swap for real API call in production) ──────────────────────────

class MockLLM:
    """
    Rule-based mock that mimics LLM response structure.
    Replace _call_api() with openai.chat.completions.create() for production.
    """

    def generate(self, prompt: str, patient: Dict, context_notes: List[Dict]) -> Dict:
        diagnosis = patient.get("diagnosis", "unknown condition")
        risk_tier = patient.get("risk_tier", "UNKNOWN")
        age = patient.get("age", "unknown")
        meds = patient.get("medications", [])
        allergies = patient.get("allergies", [])
        prev = patient.get("prev_admissions", 0)

        # Build grounded recommendation based on retrieved context
        context_diagnoses = list({n["diagnosis"] for n in context_notes})
        similar_cases = len(context_notes)

        recommendations = []
        alerts = []

        # Risk-based recommendations
        if risk_tier == "HIGH":
            recommendations.append(
                f"PRIORITY: Schedule 48-hour post-discharge follow-up call for this {age}-year-old {diagnosis} patient."
            )
            alerts.append("HIGH READMISSION RISK — Immediate care coordinator assignment recommended.")
        elif risk_tier == "MEDIUM":
            recommendations.append(
                f"Schedule 7-day follow-up appointment. Monitor {diagnosis} progression closely."
            )

        # Prior admission history
        if prev >= 2:
            recommendations.append(
                f"Patient has {prev} prior admissions. Consider enrolling in chronic care management program."
            )
            alerts.append(f"Repeat admitter: {prev} prior hospitalizations on record.")

        # Medication safety
        if "Warfarin" in meds and "NSAIDs" in allergies:
            alerts.append("⚠️  Drug-allergy conflict detected: Warfarin + NSAID allergy — review medication plan.")

        # Context from retrieved notes
        if similar_cases > 0:
            recommendations.append(
                f"Retrieved {similar_cases} similar cases from clinical history. "
                f"Common diagnoses in similar presentations: {', '.join(context_diagnoses[:3])}."
            )

        # Default
        if not recommendations:
            recommendations.append(
                f"Patient appears stable. Standard discharge protocol applicable for {diagnosis}."
            )

        return {
            "recommendation": " ".join(recommendations),
            "alerts": alerts,
            "context_cases_used": similar_cases,
            "model": "mock-llm-v1",
            "latency_ms": 12,
        }


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class ClinicalRAGPipeline:
    """
    Full RAG pipeline:
      1. Receive patient query / profile
      2. Retrieve relevant clinical notes from vector store
      3. Build grounded prompt with retrieved context
      4. Call LLM for recommendation
      5. Return structured response with provenance
    """

    def __init__(self, notes_index: ClinicalNotesIndex, llm: Optional[object] = None):
        self.index = notes_index
        self.llm = llm or MockLLM()

    def _build_query(self, patient: Dict) -> str:
        """Construct a retrieval query from patient profile."""
        return (
            f"{patient.get('diagnosis', '')} "
            f"age {patient.get('age', '')} "
            f"medications {' '.join(patient.get('medications', []))} "
            f"risk {patient.get('risk_tier', '')}"
        )

    def _build_prompt(self, patient: Dict, context_notes: List[Dict]) -> str:
        """Build a grounded LLM prompt with retrieved context."""
        context_block = "\n".join([
            f"- [{n['risk_tier']}] {n['diagnosis']}: {n['note'][:120]}..."
            for n in context_notes
        ])

        return f"""You are a clinical decision support assistant.

PATIENT PROFILE:
- ID: {patient.get('patient_id')}
- Age: {patient.get('age')} | Gender: {patient.get('gender')}
- Primary Diagnosis: {patient.get('diagnosis')}
- Risk Tier: {patient.get('risk_tier')} (Score: {patient.get('risk_score')})
- Medications: {', '.join(patient.get('medications', []))}
- Allergies: {', '.join(patient.get('allergies', []))}
- Prior Admissions: {patient.get('prev_admissions')}
- Vitals: {json.dumps(patient.get('vitals', {}))}

RETRIEVED SIMILAR CASES ({len(context_notes)} cases):
{context_block}

Based on this patient profile and similar historical cases, provide:
1. Immediate clinical recommendations
2. Any safety alerts (drug interactions, allergy conflicts)
3. Readmission risk mitigation steps
"""

    def run(self, patient: Dict, top_k: int = 3) -> Dict:
        """
        Run full RAG pipeline for a patient.
        Returns structured clinical recommendation.
        """
        start = time.time()

        # Step 1: Build retrieval query
        query = self._build_query(patient)

        # Step 2: Retrieve relevant context
        context_notes = self.index.retrieve(query, top_k=top_k)

        # Step 3: Build grounded prompt
        prompt = self._build_prompt(patient, context_notes)

        # Step 4: Generate recommendation
        llm_response = self.llm.generate(prompt, patient, context_notes)

        latency_ms = round((time.time() - start) * 1000, 1)

        return {
            "patient_id":        patient["patient_id"],
            "risk_tier":         patient["risk_tier"],
            "risk_score":        patient["risk_score"],
            "diagnosis":         patient["diagnosis"],
            "recommendation":    llm_response["recommendation"],
            "alerts":            llm_response["alerts"],
            "retrieved_context": [
                {"note_id": n["note_id"], "diagnosis": n["diagnosis"],
                 "similarity": n["similarity_score"]}
                for n in context_notes
            ],
            "prompt_used":       prompt,
            "latency_ms":        latency_ms,
            "model":             llm_response["model"],
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    index = ClinicalNotesIndex().build("data/clinical_notes.json")
    pipeline = ClinicalRAGPipeline(notes_index=index)

    # Load a sample patient
    with open("data/patients.json") as f:
        patients = json.load(f)

    # Test on first high-risk patient
    high_risk = next(p for p in patients if p["risk_tier"] == "HIGH")
    result = pipeline.run(high_risk)

    print(f"\n{'='*60}")
    print(f"Patient : {result['patient_id']} | {result['diagnosis']}")
    print(f"Risk    : {result['risk_tier']} (score={result['risk_score']})")
    print(f"Latency : {result['latency_ms']}ms")
    print(f"\n📋 Recommendation:\n{result['recommendation']}")
    if result["alerts"]:
        print(f"\n⚠️  Alerts:")
        for a in result["alerts"]:
            print(f"   {a}")
    print(f"\n🔍 Context retrieved from {len(result['retrieved_context'])} similar cases")
    print(f"{'='*60}")
