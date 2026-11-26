"""Extract and chunk local PDF advisories while preserving page citations."""
import json, re, hashlib
from pathlib import Path
import fitz
from .config import RAW_DIR, INDEX_DIR

def _clean(text):
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_pages():
    records = []
    for pdf in sorted(RAW_DIR.rglob("*.pdf")):
        doc = fitz.open(pdf)
        for i, page in enumerate(doc):
            text = _clean(page.get_text("text"))
            if len(text) > 40:
                records.append({"source": pdf.name, "page": i + 1, "text": text})
        doc.close()
    if not records:
        raise FileNotFoundError(f"No extractable PDFs found under {RAW_DIR}. Add the ICAR PDFs to data/raw/.")
    return records

def chunk_pages(pages, chunk_chars=1000, overlap=160):
    chunks = []
    for rec in pages:
        text = rec["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            if end < len(text):
                boundary = text.rfind(". ", start + chunk_chars // 2, end)
                if boundary > start:
                    end = boundary + 1
            body = text[start:end].strip()
            if len(body) > 40:
                key = f"{rec['source']}|{rec['page']}|{start}|{body[:80]}"
                chunks.append({"id": hashlib.sha1(key.encode()).hexdigest()[:12],
                    "source": rec["source"], "page": rec["page"], "language": "regional/unspecified",
                    "text": body})
            if end >= len(text): break
            start = max(start + 1, end - overlap)
    return chunks

def build_corpus():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = chunk_pages(extract_pages())
    out = INDEX_DIR / "chunks.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in chunks), encoding="utf-8")
    return chunks
