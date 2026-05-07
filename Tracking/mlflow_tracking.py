"""
mlflow_tracking.py
-------------------
Tracks RAG pipeline performance metrics.
Logs latency, retrieval quality, alert counts, and risk distribution.

In production: logs to a real MLflow tracking server.
In this repo:  writes structured JSON logs locally, mimicking MLflow's API.

Author: Jahnav Jayanth Reddy Kukkala
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict


# ── Local MLflow-style Tracker ────────────────────────────────────────────────

class ExperimentTracker:
    """
    Mimics MLflow's tracking API locally.
    Logs runs, metrics, and params to tracking/runs/.
    Swap log_metric() / log_param() for mlflow equivalents in production.
    """

    def __init__(self, experiment_name: str = "clinical-decision-support"):
        self.experiment_name = experiment_name
        self.run_dir = f"tracking/runs/{experiment_name}"
        os.makedirs(self.run_dir, exist_ok=True)
        self._current_run: Dict = {}
        self._run_id: str = ""

    def start_run(self, run_name: str = "") -> "ExperimentTracker":
        self._run_id = f"run_{int(time.time())}"
        self._current_run = {
            "run_id":          self._run_id,
            "run_name":        run_name or self._run_id,
            "experiment":      self.experiment_name,
            "start_time":      datetime.now().isoformat(),
            "status":          "RUNNING",
            "params":          {},
            "metrics":         {},
            "tags":            {},
        }
        print(f"🚀 MLflow run started: {self._run_id}")
        return self

    def log_param(self, key: str, value: Any):
        self._current_run["params"][key] = value

    def log_metric(self, key: str, value: float):
        self._current_run["metrics"][key] = round(value, 4)

    def set_tag(self, key: str, value: str):
        self._current_run["tags"][key] = value

    def end_run(self):
        self._current_run["status"] = "FINISHED"
        self._current_run["end_time"] = datetime.now().isoformat()
        path = f"{self.run_dir}/{self._run_id}.json"
        with open(path, "w") as f:
            json.dump(self._current_run, f, indent=2)
        print(f"✅ Run logged → {path}")
        return self._current_run

    def load_runs(self) -> List[Dict]:
        runs = []
        for fname in os.listdir(self.run_dir):
            if fname.endswith(".json"):
                with open(f"{self.run_dir}/{fname}") as f:
                    runs.append(json.load(f))
        return sorted(runs, key=lambda r: r["start_time"], reverse=True)


# ── Pipeline Evaluator ────────────────────────────────────────────────────────

class PipelineEvaluator:
    """
    Evaluates RAG pipeline performance across a batch of patients.
    Computes:
      - Mean latency
      - Alert rate (% of patients with safety alerts)
      - Risk tier distribution
      - Mean retrieval similarity score
      - Coverage (% of patients where context was retrieved)
    """

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def evaluate_batch(self, results: List[Dict], run_name: str = "batch_eval") -> Dict:
        self.tracker.start_run(run_name)

        n = len(results)
        latencies       = [r["latency_ms"] for r in results]
        has_alerts      = [1 if r["alerts"] else 0 for r in results]
        risk_counts     = defaultdict(int)
        sim_scores      = []
        context_counts  = []

        for r in results:
            risk_counts[r["risk_tier"]] += 1
            for ctx in r.get("retrieved_context", []):
                sim_scores.append(ctx["similarity"])
            context_counts.append(len(r.get("retrieved_context", [])))

        mean_latency    = sum(latencies) / n
        alert_rate      = sum(has_alerts) / n * 100
        mean_similarity = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        coverage        = sum(1 for c in context_counts if c > 0) / n * 100

        # Log params
        self.tracker.log_param("n_patients",     n)
        self.tracker.log_param("top_k_retrieval", 3)
        self.tracker.log_param("model",           results[0]["model"] if results else "unknown")

        # Log metrics
        self.tracker.log_metric("mean_latency_ms",       mean_latency)
        self.tracker.log_metric("p95_latency_ms",        sorted(latencies)[int(0.95 * n)])
        self.tracker.log_metric("alert_rate_pct",        alert_rate)
        self.tracker.log_metric("mean_retrieval_similarity", mean_similarity)
        self.tracker.log_metric("context_coverage_pct",  coverage)
        self.tracker.log_metric("high_risk_pct",         risk_counts["HIGH"] / n * 100)
        self.tracker.log_metric("medium_risk_pct",       risk_counts["MEDIUM"] / n * 100)
        self.tracker.log_metric("low_risk_pct",          risk_counts["LOW"] / n * 100)

        self.tracker.set_tag("stage", "evaluation")
        run = self.tracker.end_run()

        summary = {
            "n_patients":             n,
            "mean_latency_ms":        round(mean_latency, 1),
            "alert_rate_pct":         round(alert_rate, 1),
            "mean_retrieval_similarity": round(mean_similarity, 4),
            "context_coverage_pct":   round(coverage, 1),
            "risk_distribution":      dict(risk_counts),
            "run_id":                 run["run_id"],
        }

        print(f"\n{'='*55}")
        print(f" PIPELINE EVALUATION SUMMARY")
        print(f"{'='*55}")
        print(f" Patients evaluated    : {n}")
        print(f" Mean latency          : {summary['mean_latency_ms']} ms")
        print(f" Alert rate            : {summary['alert_rate_pct']}%")
        print(f" Retrieval similarity  : {summary['mean_retrieval_similarity']}")
        print(f" Context coverage      : {summary['context_coverage_pct']}%")
        print(f" Risk distribution     : {dict(risk_counts)}")
        print(f"{'='*55}\n")

        return summary


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import json
    from pipelines.embeddings import ClinicalNotesIndex
    from pipelines.rag_pipeline import ClinicalRAGPipeline

    index    = ClinicalNotesIndex().build("data/clinical_notes.json")
    pipeline = ClinicalRAGPipeline(notes_index=index)
    tracker  = ExperimentTracker("clinical-decision-support")
    evaluator = PipelineEvaluator(tracker)

    with open("data/patients.json") as f:
        patients = json.load(f)

    # Evaluate on 50 patients
    sample   = patients[:50]
    results  = [pipeline.run(p) for p in sample]
    summary  = evaluator.evaluate_batch(results, run_name="eval_50_patients")
