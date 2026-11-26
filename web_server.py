"""
Web server for IndicAgriRAG with multilingual retrieval, Ollama generation,
evaluation metrics, corpus inspection, and PDF upload.
"""
import os
import sys
import json
import time
import re
from pathlib import Path

# Force offline mode for transformers if network is restricted to avoid slow hub lookups
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import tornado.ioloop
import tornado.web
import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from indic_agri_rag.config import ROOT as PROJ_ROOT, RAW_DIR, INDEX_DIR, OUTPUT_DIR, SETTINGS, DEFAULT_TOP_K, GENERATION_MODEL
from indic_agri_rag.retrieval import HybridRetriever, tokens
from indic_agri_rag.corpus import build_corpus

# Global state for retriever and custom uploaded corpora
retriever_instance = None
uploaded_corpora = {}  # session_id -> { "docs": [...], "bm25": BM25Okapi, "filename": str }

def get_retriever():
    global retriever_instance
    if retriever_instance is None:
        retriever_instance = HybridRetriever()
    return retriever_instance


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    def write_json(self, data, status=200):
        self.set_status(status)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.write(json.dumps(data, ensure_ascii=False, indent=2))


class ExamplesHandler(BaseHandler):
    """Returns curated example questions across English, Hindi, and Hinglish."""
    def get(self):
        examples = [
            # Batch 1
            [
                {
                    "badge": "English · Pest Management",
                    "language": "English",
                    "question": "What are the symptoms and control measures for blast disease in Kharif paddy?",
                    "category": "Pest & Disease"
                },
                {
                    "badge": "Hindi · कीट नियंत्रण",
                    "language": "Hindi",
                    "question": "धान की फसल में तना छेदक (stem borer) कीट के नियंत्रण के लिए क्या उपाय करें?",
                    "category": "Pest & Disease"
                },
                {
                    "badge": "Hinglish · Fertilizer & Nutrition",
                    "language": "Hinglish",
                    "question": "Kapas ki fasal me kitna urea aur DAP dalna chahiye?",
                    "category": "Nutrition"
                },
                {
                    "badge": "English · Nutrient Schedule",
                    "language": "English",
                    "question": "What is the recommended split application schedule of nitrogen for rice?",
                    "category": "Advisory"
                }
            ],
            # Batch 2
            [
                {
                    "badge": "Hindi · खरपतवार प्रबंधन",
                    "language": "Hindi",
                    "question": "धान की रोपाई के बाद खरपतवार नियंत्रण के लिए कौन सी दवा का प्रयोग करें?",
                    "category": "Weed Control"
                },
                {
                    "badge": "Hinglish · Sowing Advisory",
                    "language": "Hinglish",
                    "question": "Punjab mein rainfed groundnut ke liye ideal sowing time kya hai?",
                    "category": "Agronomy"
                },
                {
                    "badge": "English · Fall Armyworm",
                    "language": "English",
                    "question": "How can farmers control Fall Armyworm (FAW) in Kharif maize crops?",
                    "category": "Pest Control"
                },
                {
                    "badge": "Hinglish · Soil Treatment",
                    "language": "Hinglish",
                    "question": "Rajasthan soil ke liye gypsum application ka importance aur dose kya hai?",
                    "category": "Soil Management"
                }
            ],
            # Batch 3
            [
                {
                    "badge": "Hindi · रोग एवं उपचार",
                    "language": "Hindi",
                    "question": "कपास में पत्ती मरोड़ (leaf curl virus) और सफेद मक्खी की रोकथाम कैसे करें?",
                    "category": "Pest Control"
                },
                {
                    "badge": "English · Herbicide Timing",
                    "language": "English",
                    "question": "Which herbicides are recommended for weed management in direct seeded rice?",
                    "category": "Weed Control"
                },
                {
                    "badge": "Hinglish · Livestock Care",
                    "language": "Hinglish",
                    "question": "Kharif mausam me pashuon ke chare aur rog bachav ke liye kya karein?",
                    "category": "Livestock"
                },
                {
                    "badge": "Hindi · मखाना एवं जल कृषि",
                    "language": "Hindi",
                    "question": "खरीफ मौसम में मखाना की फसल में प्रमुख कीट और जल प्रबंधन के नियम क्या हैं?",
                    "category": "Agronomy"
                }
            ]
        ]
        self.write_json({"batches": examples})


class SearchHandler(BaseHandler):
    """Performs retrieval over indexed passages with mode selection."""
    def post(self):
        try:
            body = json.loads(self.request.body.decode("utf-8"))
        except Exception:
            self.write_json({"error": "Invalid JSON body"}, status=400)
            return

        query = body.get("query", "").strip()
        top_k = int(body.get("top_k", DEFAULT_TOP_K))
        mode = body.get("mode", "hybrid").lower()  # "hybrid", "dense", "sparse"
        source = body.get("source", "builtin")     # "builtin" or uploaded file ID

        if not query:
            self.write_json({"error": "Query cannot be empty"}, status=400)
            return

        t0 = time.perf_counter()
        retriever = get_retriever()

        # Handle uploaded custom corpus
        if source != "builtin" and source in uploaded_corpora:
            corpus_data = uploaded_corpora[source]
            docs = corpus_data["docs"]
            bm25 = corpus_data["bm25"]
            scores = np.asarray(bm25.get_scores(tokens(query)), dtype=float)
            ids = np.argsort(-scores)[:top_k]
            passages = [{**docs[i], "score": float(scores[i])} for i in ids if scores[i] > 0]
            if not passages:
                passages = [{**docs[i], "score": 0.0} for i in ids[:top_k]]
            latency = time.perf_counter() - t0
            self.write_json({
                "query": query,
                "mode": "sparse (custom document)",
                "latency_seconds": round(latency, 4),
                "passages": passages
            })
            return

        # Built-in corpus retrieval
        if mode == "dense":
            q_emb = retriever._encode([query], "query")[0]
            dense_scores = retriever.embeddings @ q_emb
            ids = np.argsort(-dense_scores)[:top_k]
            passages = [{**retriever.docs[i], "score": float(dense_scores[i])} for i in ids]
        elif mode == "sparse":
            sparse_scores = np.asarray(retriever.bm25.get_scores(tokens(query)), dtype=float)
            ids = np.argsort(-sparse_scores)[:top_k]
            passages = [{**retriever.docs[i], "score": float(sparse_scores[i])} for i in ids]
        else:  # Hybrid Weighted with Reciprocal Rank Fusion
            passages = retriever.search(query, k=top_k)

        latency = time.perf_counter() - t0
        self.write_json({
            "query": query,
            "mode": mode,
            "latency_seconds": round(latency, 4),
            "passages": passages
        })


class AskHandler(BaseHandler):
    """Retrieves passages and generates a grounded response with citations."""
    def post(self):
        try:
            body = json.loads(self.request.body.decode("utf-8"))
        except Exception:
            self.write_json({"error": "Invalid JSON body"}, status=400)
            return

        query = body.get("query", "").strip()
        top_k = int(body.get("top_k", DEFAULT_TOP_K))
        mode = body.get("mode", "hybrid").lower()
        target_lang = body.get("target_lang", "auto")
        source = body.get("source", "builtin")

        if not query:
            self.write_json({"error": "Query cannot be empty"}, status=400)
            return

        t0 = time.perf_counter()
        retriever = get_retriever()

        # Step 1: Retrieve passages
        if source != "builtin" and source in uploaded_corpora:
            corpus_data = uploaded_corpora[source]
            docs = corpus_data["docs"]
            bm25 = corpus_data["bm25"]
            scores = np.asarray(bm25.get_scores(tokens(query)), dtype=float)
            ids = np.argsort(-scores)[:top_k]
            passages = [{**docs[i], "score": float(scores[i])} for i in ids if i < len(docs)]
        else:
            if mode == "dense":
                q_emb = retriever._encode([query], "query")[0]
                dense_scores = retriever.embeddings @ q_emb
                ids = np.argsort(-dense_scores)[:top_k]
                passages = [{**retriever.docs[i], "score": float(dense_scores[i])} for i in ids]
            elif mode == "sparse":
                sparse_scores = np.asarray(retriever.bm25.get_scores(tokens(query)), dtype=float)
                ids = np.argsort(-sparse_scores)[:top_k]
                passages = [{**retriever.docs[i], "score": float(sparse_scores[i])} for i in ids]
            else:
                passages = retriever.search(query, k=top_k)

        # Step 2: Answer generation via Ollama or Intelligent Fallback
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        model_name = os.getenv("OLLAMA_MODEL", GENERATION_MODEL)
        
        answer_text, is_answerable, generator_used = self._generate_answer(
            query, passages, target_lang, ollama_url, model_name
        )

        latency = time.perf_counter() - t0

        # Step 3: Extract citations
        citations = []
        for p in passages:
            c_tag = f"[{p['source']}, p. {p['page']}]"
            short_tag = f"p. {p['page']}"
            if c_tag in answer_text or short_tag in answer_text or len(citations) < 2:
                citations.append({
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_id": p.get("id", ""),
                    "score": round(p.get("score", 0.0), 4),
                    "snippet": p["text"][:300] + ("..." if len(p["text"]) > 300 else "")
                })

        self.write_json({
            "query": query,
            "answer": answer_text,
            "is_answerable": is_answerable,
            "mode": mode,
            "generator": generator_used,
            "latency_seconds": round(latency, 2),
            "passages": passages,
            "citations": citations
        })

    def _generate_answer(self, question, passages, target_lang, ollama_url, model_name):
        import requests

        system_prompt = (
            "You are an agricultural advisory assistant for Indian farmers. "
            "First decide if the supplied evidence is sufficient to answer the question. "
            "Start your response with exactly <<ANSWERABLE>> or <<UNANSWERABLE>>. "
            "Then answer directly and clearly in the requested language/script (English, Hindi, or Hinglish). "
            "Use ONLY the supplied ICAR evidence. Never invent chemical doses, safety advice, or unverified claims. "
            "Cite every claim with [source, p. N]. Keep recommendations safe and practical."
        )

        context = "\n\n".join(f"[{p['source']}, p. {p['page']}] {p['text']}" for p in passages)
        prompt = (
            f"Evidence:\n{context}\n\n"
            f"Farmer's question: {question}\n"
            f"Target language instruction: {target_lang}\n\n"
            "Answer with inline evidence citations. If evidence does not answer the question, clearly state <<UNANSWERABLE>>."
        )

        # Attempt 1: Call local Ollama if available
        try:
            r = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model_name,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {"temperature": 0.1, "num_ctx": 8192}
                },
                timeout=30
            )
            if r.status_code == 200:
                raw = r.json().get("message", {}).get("content", "").strip()
                is_ans = "<<ANSWERABLE>>" in raw or not ("<<UNANSWERABLE>>" in raw)
                clean_ans = re.sub(r"^\s*<<(?:UN)?ANSWERABLE>>\s*", "", raw, flags=re.I).strip()
                return clean_ans, is_ans, f"Ollama ({model_name})"
        except Exception:
            pass

        # Attempt 2: Grounded Extractive Synthesizer (Zero-Network fallback)
        # Evaluates keyword overlap and relevance of top passages
        q_tokens = set(tokens(question))
        total_overlap = 0
        best_p = passages[0] if passages else None

        for p in passages:
            p_tokens = set(tokens(p["text"]))
            overlap = len(q_tokens.intersection(p_tokens))
            if overlap > total_overlap:
                total_overlap = overlap
                best_p = p

        # Check if question is answerable based on retrieved evidence overlap and score
        has_substantive_evidence = (best_p is not None and (best_p.get("score", 0) > 0.015 or total_overlap >= 2))

        # Detect language of question for friendly natural answer
        is_hindi = bool(re.search(r"[\u0900-\u097F]", question))
        is_hinglish = any(w in question.casefold() for w in ["kya", "kaise", "ke liye", "hai", "karein", "fasal", "me", "mein"])

        if not has_substantive_evidence:
            if is_hindi:
                msg = (
                    "उपलब्ध आईसीएआर (ICAR) कृषि परामर्श सामग्री में इस विशिष्ट प्रश्न के लिए पर्याप्त जानकारी उपलब्ध नहीं है। "
                    "गलत या अधूरी सलाह से बचने के लिए कृपया अपने नजदीकी कृषि विज्ञान केंद्र (KVK) या किसान कॉल सेंटर (1800-180-1551) से संपर्क करें।"
                )
            elif is_hinglish:
                msg = (
                    "Supplied ICAR advisories mein is question ke liye sufficient evidence nahi mila. "
                    "Inaccurate dosage ya advice se bachne ke liye please apne nearest KVK (Krishi Vigyan Kendra) ya Krishi Adhikari se contact karein."
                )
            else:
                msg = (
                    "The supplied ICAR agricultural advisories do not contain sufficient evidence to answer this specific question. "
                    "To ensure crop safety and accurate dosage, please consult your nearest Krishi Vigyan Kendra (KVK) or local agricultural extension officer."
                )
            return msg, False, "ICAR Grounded Guardrail (Ollama offline)"

        # Construct answer from top passages
        top_chunks = passages[:3]
        cited_paragraphs = []
        for p in top_chunks:
            # Clean snippet for readable summary
            raw_t = p["text"].strip()
            # Remove excessive whitespace
            clean_t = " ".join(raw_t.split())
            if len(clean_t) > 400:
                clean_t = clean_t[:400] + "..."
            cited_paragraphs.append(f"• {clean_t} [{p['source']}, p. {p['page']}]")

        if is_hindi:
            header = "**आईसीएआर (ICAR) परामर्श के अनुसार मुख्य अनुशंसाएं:**\n\n"
        elif is_hinglish:
            header = "**ICAR Kharif Advisories ke mutabik key recommendations:**\n\n"
        else:
            header = "**Key recommendations from ICAR Kharif Advisories:**\n\n"

        grounded_answer = header + "\n\n".join(cited_paragraphs)
        return grounded_answer, True, "ICAR Grounded Extractor (BGE-M3 + BM25)"


class MetricsHandler(BaseHandler):
    """Serves the benchmark evaluation results and available plot images."""
    def get(self):
        metrics_file = OUTPUT_DIR / "metrics.json"
        data = {}
        if metrics_file.exists():
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Collect available plot image URLs
        plots = []
        plot_descriptions = {
            "accuracy_by_language.png": "Answerability Accuracy across English, Hindi, and Hinglish",
            "precision_recall_f1_by_language.png": "Precision, Recall, and F1 Metrics across Languages",
            "accuracy_by_script_form.png": "Performance across Native Script vs Code-Mixed (Hinglish)",
            "confusion_matrix.png": "Confusion Matrix: Answerable vs Unanswerable abstention",
            "confusion_matrices_by_language.png": "Language-wise Confusion Matrices",
            "retriever_ablation_recall_at_5.png": "Retriever Ablation: Hybrid Fusion vs Dense vs BM25 at k=5",
            "retrieval_recall_at_k.png": "Recall@k curve (k=1 to 10)",
            "citation_support_coverage_by_language.png": "Grounding Citation Support Coverage by Language",
            "abstention_rate_by_language_and_gold.png": "Safe Abstention Rate on Unanswerable vs Answerable queries",
            "latency_distribution.png": "Inference and Retrieval Latency Distribution (seconds)",
            "answer_length_vs_latency.png": "Answer Token Length vs Latency Correlation",
            "classification_metrics.png": "Macro vs Micro Classification Report Summary"
        }

        if OUTPUT_DIR.exists():
            for p in OUTPUT_DIR.glob("*.png"):
                plots.append({
                    "filename": p.name,
                    "url": f"/outputs/{p.name}",
                    "title": p.stem.replace("_", " ").title(),
                    "description": plot_descriptions.get(p.name, p.stem.replace("_", " "))
                })

        self.write_json({
            "metrics": data,
            "plots": plots
        })


class CorpusHandler(BaseHandler):
    """Returns information about indexed corpus chunks and documents."""
    def get(self):
        retriever = get_retriever()
        doc_count = len(retriever.docs)
        sources = {}
        for d in retriever.docs:
            src = d.get("source", "Unknown")
            sources[src] = sources.get(src, 0) + 1

        self.write_json({
            "total_chunks": doc_count,
            "embedding_model": SETTINGS.get("embedding_model", "BAAI/bge-m3"),
            "generation_model": SETTINGS.get("generation_model", "qwen2.5:14b"),
            "top_k": SETTINGS.get("top_k", 8),
            "sources": sources,
            "uploaded_sources": [
                {"id": k, "name": v["filename"], "chunk_count": len(v["docs"])}
                for k, v in uploaded_corpora.items()
            ]
        })


class UploadPdfHandler(BaseHandler):
    """Allows uploading a custom agricultural PDF to chunk and query."""
    def post(self):
        import fitz  # PyMuPDF
        import uuid
        from rank_bm25 import BM25Okapi

        if "file" not in self.request.files:
            self.write_json({"error": "No file uploaded"}, status=400)
            return

        file_obj = self.request.files["file"][0]
        filename = file_obj["filename"]
        body_bytes = file_obj["body"]

        try:
            doc = fitz.open(stream=body_bytes, filetype="pdf")
            chunks = []
            chunk_size = 1000
            overlap = 160

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if not text.strip():
                    continue

                for i in range(0, len(text), chunk_size - overlap):
                    chunk_text = text[i:i + chunk_size].strip()
                    if len(chunk_text) > 40:
                        chunks.append({
                            "id": f"up_{uuid.uuid4().hex[:8]}",
                            "source": filename,
                            "page": page_num + 1,
                            "text": chunk_text
                        })

            if not chunks:
                self.write_json({"error": "No text could be extracted from PDF"}, status=400)
                return

            upload_id = f"custom_{uuid.uuid4().hex[:8]}"
            bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
            uploaded_corpora[upload_id] = {
                "docs": chunks,
                "bm25": bm25,
                "filename": filename
            }

            self.write_json({
                "success": True,
                "upload_id": upload_id,
                "filename": filename,
                "pages": len(doc),
                "chunks_created": len(chunks),
                "message": f"Successfully indexed {len(chunks)} chunks from {filename}."
            })
        except Exception as e:
            self.write_json({"error": f"Failed to process PDF: {str(e)}"}, status=500)


def make_app():
    static_dir = ROOT / "web" / "static"
    output_dir = OUTPUT_DIR

    return tornado.web.Application([
        (r"/api/examples", ExamplesHandler),
        (r"/api/search", SearchHandler),
        (r"/api/ask", AskHandler),
        (r"/api/metrics", MetricsHandler),
        (r"/api/corpus", CorpusHandler),
        (r"/api/upload-pdf", UploadPdfHandler),
        (r"/outputs/(.*)", tornado.web.StaticFileHandler, {"path": str(output_dir)}),
        (r"/assets/(.*)", tornado.web.StaticFileHandler, {"path": str(ROOT / "assets")}),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": str(static_dir), "default_filename": "index.html"}),
    ], debug=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start IndicAgriRAG Web UI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    port = args.port
    app = make_app()
    app.listen(port)
    print(f"🌾 IndicAgriRAG Web Application running at: http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    tornado.ioloop.IOLoop.current().start()
