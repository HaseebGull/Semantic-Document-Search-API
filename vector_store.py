import faiss
import numpy as np
import json
import os

class VectorStore:
    def __init__(self, dim=1536):
        self.index_path = "data/faiss.index"
        self.meta_path = "data/meta.json"

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.metadata = []

        self._load()

    def add(self, embedding, meta):
        vec = np.array([embedding]).astype("float32")
        faiss.normalize_L2(vec)

        self.index.add(vec)
        self.metadata.append(meta)

    def search(self, embedding, k=5):
        if self.index.ntotal == 0:
            return []

        query = np.array([embedding]).astype("float32")
        faiss.normalize_L2(query)
        k = min(5, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    **self.metadata[idx],
                    "score": float(score)
                })

        return results

    def save(self):
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f)

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)