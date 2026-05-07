"""
embeddings.py
--------------
Builds and queries a FAISS vector index over clinical notes.
Uses TF-IDF style embeddings (no external API needed) with cosine similarity.

Author: Jahnav Jayanth Reddy Kukkala
"""

import json
import math
import struct
import os
from collections import Counter
from typing import List, Dict, Tuple


# ── Lightweight TF-IDF Vectorizer ────────────────────────────────────────────

class TFIDFVectorizer:
    """
    Minimal TF-IDF vectorizer — no sklearn dependency.
    Builds vocabulary from corpus, transforms docs to weighted vectors.
    """

    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        for ch in ".,;:!?()[]{}\"'":
            text = text.replace(ch, " ")
        return [t for t in text.split() if len(t) > 2]

    def fit(self, documents: List[str]):
        n_docs = len(documents)
        doc_freq: Counter = Counter()
        all_tokens = []

        for doc in documents:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
            all_tokens.extend(self._tokenize(doc))

        # Pick top features by document frequency
        top_terms = [term for term, _ in doc_freq.most_common(self.max_features)]
        self.vocab = {term: idx for idx, term in enumerate(top_terms)}

        # Compute IDF
        for term, df in doc_freq.items():
            if term in self.vocab:
                self.idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

        self._fitted = True
        return self

    def transform(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")

        tokens = self._tokenize(text)
        tf: Counter = Counter(tokens)
        n_tokens = max(len(tokens), 1)

        vector = [0.0] * len(self.vocab)
        for term, count in tf.items():
            if term in self.vocab:
                tf_val = count / n_tokens
                idf_val = self.idf.get(term, 1.0)
                vector[self.vocab[term]] = tf_val * idf_val

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "idf": self.idf}, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self._fitted = True
        return self


# ── FAISS-style In-Memory Vector Store ───────────────────────────────────────

class VectorStore:
    """
    Lightweight FAISS-inspired vector store.
    Stores embeddings in memory, supports cosine similarity search.
    In production this would wrap actual FAISS for million-scale retrieval.
    """

    def __init__(self):
        self.vectors: List[List[float]] = []
        self.metadata: List[Dict] = []

    def add(self, vector: List[float], metadata: Dict):
        self.vectors.append(vector)
        self.metadata.append(metadata)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[float, Dict]]:
        """Return top_k most similar documents by cosine similarity."""
        scores = []
        for idx, vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(query_vector, vec))
            scores.append((dot, self.metadata[idx]))

        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]

    def __len__(self):
        return len(self.vectors)


# ── Clinical Notes Index ──────────────────────────────────────────────────────

class ClinicalNotesIndex:
    """
    Builds a searchable index over clinical notes using TF-IDF + vector store.
    Mimics what FAISS + sentence-transformers would do in production.
    """

    def __init__(self, max_features: int = 512):
        self.vectorizer = TFIDFVectorizer(max_features=max_features)
        self.store = VectorStore()
        self.notes: List[Dict] = []

    def build(self, notes_path: str) -> "ClinicalNotesIndex":
        with open(notes_path) as f:
            self.notes = json.load(f)

        # Fit vectorizer on all note texts
        texts = [n["note"] for n in self.notes]
        self.vectorizer.fit(texts)

        # Embed and index each note
        for note in self.notes:
            vector = self.vectorizer.transform(note["note"])
            self.store.add(vector, {
                "note_id":    note["note_id"],
                "patient_id": note["patient_id"],
                "diagnosis":  note["diagnosis"],
                "risk_tier":  note["risk_tier"],
                "physician":  note["physician"],
                "note":       note["note"],
                "date":       note["date"],
            })

        print(f"✅ ClinicalNotesIndex built: {len(self.store)} notes indexed")
        return self

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top_k most relevant clinical notes for a query.
        Returns list of note metadata dicts with similarity scores.
        """
        query_vec = self.vectorizer.transform(query)
        results = self.store.search(query_vec, top_k=top_k)
        return [
            {**meta, "similarity_score": round(score, 4)}
            for score, meta in results
        ]

    def retrieve_by_patient(self, patient_id: str) -> List[Dict]:
        """Retrieve all notes for a specific patient."""
        return [n for n in self.notes if n["patient_id"] == patient_id]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    index = ClinicalNotesIndex()
    index.build("data/clinical_notes.json")

    print("\n🔍 Query: 'patient with diabetes high blood glucose'")
    results = index.retrieve("patient with diabetes high blood glucose", top_k=3)
    for r in results:
        print(f"  [{r['risk_tier']}] {r['patient_id']} | {r['diagnosis']} | score={r['similarity_score']}")
        print(f"  Note: {r['note'][:100]}...")
        print()
