"""Grounded answer generation via local Ollama; citations point to source pages."""
import os, requests
from .config import GENERATION_MODEL

SYSTEM = """You are an agricultural information assistant. First decide whether the supplied evidence is sufficient to answer the farmer's specific question. Start your response with exactly <<ANSWERABLE>> or <<UNANSWERABLE>>. Then answer in the same language and script as the farmer's question when possible. Use only the supplied ICAR evidence. Do not invent chemical doses, safety advice, or facts. If evidence is insufficient, say that the supplied advisories do not contain enough information and suggest contacting a local KVK/agricultural officer. Cite every factual claim using [source, p. N]. Keep advice cautious."""

def answer_question(question, passages):
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", GENERATION_MODEL)
    context = "\n\n".join(f"[{p['source']}, p. {p['page']}] {p['text']}" for p in passages)
    prompt = f"Evidence:\n{context}\n\nFarmer question: {question}\n\nAnswer with inline evidence citations. If evidence does not answer the question, abstain clearly."
    try:
        r = requests.post(url + "/api/chat", json={"model": model, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "num_ctx": 8192}}, timeout=180)
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
    except requests.RequestException as e:
        text = "<<UNANSWERABLE>> Insufficient evidence: the local answer model is unavailable. Retrieved passages are shown for inspection. Configure Ollama to generate a grounded answer."
    return text
