from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
OUTPUT_DIR = ROOT / "outputs"
ANNOTATION_DIR = ROOT / "evaluation"

def load_settings():
    p = ROOT / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

SETTINGS = load_settings()
EMBED_MODEL = SETTINGS.get("embedding_model", "BAAI/bge-m3")
GENERATION_MODEL = SETTINGS.get("generation_model", "qwen2.5:14b")
DEFAULT_TOP_K = int(SETTINGS.get("top_k", 8))
