"""Multilingual dense + BM25 hybrid retrieval."""
import json, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .config import INDEX_DIR, EMBED_MODEL

def tokens(s): return re.findall(r"\w+", s.casefold(), flags=re.UNICODE)

class HybridRetriever:
    def __init__(self, model_name=EMBED_MODEL):
        path = INDEX_DIR / "chunks.jsonl"
        if not path.exists(): raise FileNotFoundError("Build the index first: python -m indic_agri_rag.cli index")
        self.docs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.use_e5_prefix = "e5" in model_name.casefold()
        self.model = SentenceTransformer(model_name)
        # Embeddings are model-specific; separate filenames prevent accidentally
        # reusing vectors created by a different embedding model.
        cache_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        emb_path = INDEX_DIR / f"embeddings-{cache_name}.npy"
        expected_dim = self.model.get_sentence_embedding_dimension()
        if emb_path.exists():
            cached = np.load(emb_path)
            if cached.ndim == 2 and cached.shape == (len(self.docs), expected_dim):
                self.embeddings = cached
            else:
                self.embeddings = self._encode_corpus(emb_path)
        else:
            self.embeddings = self._encode_corpus(emb_path)
        self.bm25 = BM25Okapi([tokens(d["text"]) for d in self.docs])

    def _encode(self, texts, kind):
        # E5 requires task prefixes. BGE-M3 does not.
        if self.use_e5_prefix:
            texts = [f"{kind}: {text}" for text in texts]
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=(kind == "passage"),
                                 convert_to_numpy=True)

    def _encode_corpus(self, dest):
        arr = self._encode([d["text"] for d in self.docs], "passage")
        np.save(dest, arr); return arr

    def search(self, query, k=5):
        q = self._encode([query], "query")[0]
        dense = self.embeddings @ q
        sparse = np.asarray(self.bm25.get_scores(tokens(query)), dtype=float)
        # Reciprocal-rank fusion avoids mixing incomparable dense and BM25 scales.
        dense_order = np.argsort(-dense)
        sparse_order = np.argsort(-sparse)
        fused = np.zeros(len(self.docs), dtype=float)
        for rank, idx in enumerate(dense_order, start=1): fused[idx] += 1.0 / (60 + rank)
        for rank, idx in enumerate(sparse_order, start=1): fused[idx] += 1.0 / (60 + rank)
        ids = np.argsort(-fused)[:k]
        return [{**self.docs[i], "score": float(fused[i])} for i in ids]
