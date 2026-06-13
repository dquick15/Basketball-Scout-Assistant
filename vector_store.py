from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


class ScoutVectorStore:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.index: faiss.IndexFlatIP | None = None
        self.records: list[dict[str, object]] = []
        self.embeddings: np.ndarray | None = None
        self.backend = "openai"

    def _hash_embed_texts(self, texts: list[str], dimensions: int = 512) -> np.ndarray:
        embeddings = np.zeros((len(texts), dimensions), dtype="float32")
        for row_index, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9']+", str(text).lower())
            for token in tokens:
                bucket = hash(token) % dimensions
                embeddings[row_index, bucket] += 1.0
        faiss.normalize_L2(embeddings)
        return embeddings

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            embeddings = np.array([item.embedding for item in response.data], dtype="float32")
            faiss.normalize_L2(embeddings)
            self.backend = "openai"
            return embeddings
        except Exception:
            self.backend = "hash"
            return self._hash_embed_texts(texts)

    def build(self, records: list[dict[str, object]]) -> None:
        self.records = records
        texts = [str(record["document"]) for record in records]
        embeddings = self._embed_texts(texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.embeddings = embeddings

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if self.index is None:
            raise ValueError("Vector index has not been built.")
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "wb") as file_handle:
            pickle.dump({"records": self.records, "embeddings": self.embeddings, "backend": self.backend}, file_handle)

    def load(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as file_handle:
            payload = pickle.load(file_handle)
        self.records = payload["records"]
        self.embeddings = payload["embeddings"]
        self.backend = payload.get("backend", "openai")

    def similarity_search(self, query: str, top_k: int = 6) -> list[dict[str, object]]:
        if self.index is None:
            raise ValueError("Vector index has not been built.")
        query_embedding = self._embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)
        matches: list[dict[str, object]] = []
        for score, record_index in zip(scores[0], indices[0], strict=False):
            if record_index < 0:
                continue
            record = dict(self.records[record_index])
            record["similarity"] = float(score)
            matches.append(record)
        return matches

    def similar_players(self, player_name: str, top_k: int = 5) -> list[dict[str, object]]:
        if self.index is None or self.embeddings is None:
            raise ValueError("Vector index has not been built.")

        source_index = next(
            (index for index, record in enumerate(self.records) if str(record["Player Name"]).lower() == player_name.lower()),
            None,
        )
        if source_index is None:
            return []

        query_embedding = self.embeddings[source_index : source_index + 1]
        scores, indices = self.index.search(query_embedding, top_k + 1)
        matches: list[dict[str, object]] = []
        for score, record_index in zip(scores[0], indices[0], strict=False):
            if record_index < 0 or record_index == source_index:
                continue
            record = dict(self.records[record_index])
            record["similarity"] = float(score)
            matches.append(record)
        return matches[:top_k]
