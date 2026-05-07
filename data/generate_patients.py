"""
generate_patients.py
---------------------
Generates synthetic patient records for the Clinical Decision Support System.
Produces: data/patients.json and data/clinical_notes.json

Author: Jahnav Jayanth Reddy Kukkala
"""

import json
import random
import math
from datetime import datetime, timedelta

random.seed(42)

DIAGNOSES = ["Heart Failure", "COPD", "Diabetes Type 2", "Sepsis",
             "Pneumonia", "Renal Failure", "Stroke", "Hip Fracture"]
MEDICATIONS = ["Metformin", "Lisinopril", "Atorvastatin", "Amlodipine",
               "Omeprazole", "Aspirin", "Furosemide", "Metoprolol",
               "Warfarin", "Insulin", "Prednisone", "Albuterol"]
ALLERGIES = ["Penicillin", "Sulfa", "NSAIDs", "Latex", "Codeine", "None"]
PHYSICIANS = ["Dr. Smith", "Dr. Patel", "Dr. Johnson", "Dr. Lee", "Dr. Garcia"]

CLINICAL_NOTE_TEMPLATES = [
    "Patient presents with {complaint}. Vitals stable. {diagnosis} exacerbation noted. "
    "Recommended {action}. Follow-up in {days} days.",
    "Admitted for {complaint}. History of {diagnosis}. "
    "Labs show {lab_finding}. Initiated {action}.",
    "Routine checkup. Patient reports {complaint}. {diagnosis} well-controlled on current regimen. "
    "Adjusted {action} dosage.",
    "Emergency visit for {complaint}. {diagnosis} suspected. "
    "Imaging ordered. {action} prescribed pending results.",
]

COMPLAINTS = ["shortness of breath", "chest pain", "fatigue", "dizziness",
              "elevated blood glucose", "swelling in legs", "confusion", "fever"]
ACTIONS = ["medication adjustment", "IV fluids", "oxygen therapy",
           "dietary counseling", "physical therapy", "specialist referral"]
LAB_FINDINGS = ["elevated creatinine", "low hemoglobin", "high HbA1c",
                "elevated troponin", "low potassium", "elevated BNP"]


def generate_patient(pid: int) -> dict:
    age = max(18, min(95, int(random.gauss(63, 15))))
    diagnosis = random.choice(DIAGNOSES)
    num_meds = random.randint(2, 8)
    prev_admissions = random.choices([0, 1, 2, 3, 4], weights=[35, 30, 20, 10, 5])[0]
    comorbidities = random.randint(0, 4)

    log_odds = (-3.2 + 0.03 * age + 0.28 * prev_admissions
                + 0.18 * comorbidities + 0.09 * num_meds)
    risk_score = round(1 / (1 + math.exp(-log_odds)), 4)
    risk_tier = "HIGH" if risk_score >= 0.4 else ("MEDIUM" if risk_score >= 0.2 else "LOW")

    admit_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 364))

    return {
        "patient_id": f"PAT-{pid:04d}",
        "name": f"Patient {pid:04d}",
        "age": age,
        "gender": random.choice(["M", "F"]),
        "diagnosis": diagnosis,
        "medications": random.sample(MEDICATIONS, num_meds),
        "allergies": random.sample(ALLERGIES, random.randint(1, 2)),
        "prev_admissions": prev_admissions,
        "comorbidities": comorbidities,
        "attending_physician": random.choice(PHYSICIANS),
        "admit_date": admit_date.strftime("%Y-%m-%d"),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "vitals": {
            "blood_pressure": f"{random.randint(110, 160)}/{random.randint(70, 100)}",
            "heart_rate": random.randint(55, 110),
            "temperature": round(random.uniform(97.5, 101.5), 1),
            "oxygen_saturation": random.randint(92, 100),
        }
    }


def generate_clinical_note(patient: dict) -> dict:
    template = random.choice(CLINICAL_NOTE_TEMPLATES)
    note_text = template.format(
        complaint=random.choice(COMPLAINTS),
        diagnosis=patient["diagnosis"],
        action=random.choice(ACTIONS),
        days=random.randint(3, 14),
        lab_finding=random.choice(LAB_FINDINGS),
    )
    return {
        "note_id": f"NOTE-{patient['patient_id']}",
        "patient_id": patient["patient_id"],
        "date": patient["admit_date"],
        "physician": patient["attending_physician"],
        "note": note_text,
        "diagnosis": patient["diagnosis"],
        "risk_tier": patient["risk_tier"],
    }


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    patients = [generate_patient(i) for i in range(1, 501)]
    notes = [generate_clinical_note(p) for p in patients]

    with open("data/patients.json", "w") as f:
        json.dump(patients, f, indent=2)

    with open("data/clinical_notes.json", "w") as f:
        json.dump(notes, f, indent=2)

    high_risk = sum(1 for p in patients if p["risk_tier"] == "HIGH")
    print(f"✅ Generated {len(patients)} patient records")
    print(f"   High Risk : {high_risk}")
    print(f"   Medium    : {sum(1 for p in patients if p['risk_tier'] == 'MEDIUM')}")
    print(f"   Low       : {sum(1 for p in patients if p['risk_tier'] == 'LOW')}")
    print(f"✅ Generated {len(notes)} clinical notes")
    print(f"   Saved → data/patients.json")
    print(f"   Saved → data/clinical_notes.json")
