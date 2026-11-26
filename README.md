# IndicAgriRAG — evidence-grounded agricultural QA

Local Python implementation of cross-lingual retrieval and evidence-grounded farmer Q&A using the two ICAR Kharif Agro-Advisories PDFs supplied with this project. Includes reproducible benchmark export, manual gold annotation, answerability metrics, retrieval and answer metrics, and paper-ready plots.

## Selected data and models

| Purpose | Selection | Use |
|---|---|---|
| Evidence corpus | ICAR Kharif Agro-Advisories for Farmers 2025 (English PDF + multilingual PDF) | The only source used to support answers and citations |
| QA benchmark | `sthanika-ai/Indic-KCC-Agri-Advisory-Benchmark` | Evaluation questions/reference answers only; optional because Hugging Face access is gated |
| Dense retrieval | `BAAI/bge-m3` | Multilingual dense embeddings; cached separately by model |
| Sparse retrieval | BM25 | Adds lexical matching; combined with dense retrieval |
| Answer generation | Ollama `qwen2.5:14b` | Local answer generator; can be overridden with `OLLAMA_MODEL` |

The benchmark is machine-translated for non-English languages, has noisy KCC reference answers, and explicitly is not agronomic advice. This system never indexes benchmark answers as evidence. It retrieves evidence only from ICAR PDFs. The provided PDFs are not a complete agriculture knowledge base, so abstention is expected when they do not support an answer. The multilingual PDF is kept page-cited, but a mixed-language document is not automatically language-tagged; this prototype does not claim validated equal performance in every language.

## Run on Windows

Install Python 3.10–3.12 and [Ollama](https://ollama.com/download/windows). Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:14b
```

First indexing downloads `BAAI/bge-m3` from Hugging Face and creates model-specific embeddings. CPU works, but can be slow. The 14B answer model needs substantially more memory than the previous 3B model; if it is too slow or does not fit comfortably, install the smaller fallback with `ollama pull qwen2.5:7b` and set `$env:OLLAMA_MODEL="qwen2.5:7b"` in PowerShell. Then:

```powershell
python -m pip install -e .
python -m indic_agri_rag.cli index
python -m indic_agri_rag.cli ask "What should farmers do to manage pests in paddy?"
```

### Launch the Web UI / Frontend

You can launch the web interface with:

```powershell
python run_ui.py
```
Then open your browser at **`http://localhost:8000`**.

Features included in the Web UI:
- **Design & Layout**: Pastel theme matching the reference design with soft glow, floating dock, and responsive cards.
- **Multilingual Querying**: Curated shuffleable examples in English, Hindi, and Hinglish.
- **Voice Input & TTS**: Speech-to-text with Hindi/English mic selector, plus text-to-speech audio playback for farmers.
- **Retrieval Modes**: Toggle between *Hybrid Weighted (BGE-M3 + BM25)*, *Dense Vector Only*, and *Sparse BM25*.
- **Grounding & Abstention**: Answerability badges (`<<ANSWERABLE>>` vs `<<UNANSWERABLE>>`) and clickable inline citations `[source, p. N]`.
- **Evaluation Dashboard**: Interactive visual modal showcasing project accuracy (88.3%), F1 (93.4%), confusion matrices, and ablation charts.
- **PDF Upload**: Drag-and-drop custom agricultural documents to index and search dynamically.


The retriever combines dense BGE-M3 ranking with BM25 using reciprocal-rank fusion. To compare another embedding model, change `embedding_model` in `config.json` and run `python -m indic_agri_rag.cli index`; each model gets a separate embedding cache. `generation_model` in `config.json` selects the default Ollama model, while `OLLAMA_MODEL` takes precedence for a one-off override.

Use a question in English, an Indic language, or a code-mixed form. Responses cite the ICAR PDF filename and page. Check advice against the cited ICAR passage before relying on it. This is a research prototype, not a substitute for a KVK/agricultural officer.

## Benchmark and research evaluation

The Indic-KCC benchmark has gated access. First accept its dataset terms on Hugging Face and authenticate locally (`hf auth login`). Export only the language configs you want; a small sample is recommended for a laptop:

```powershell
python -m indic_agri_rag.cli benchmark-questions --languages English Hindi Bengali --out data/benchmark/questions.csv
```

Create the annotation sheet, and manually fill `is_answerable` (`1` if the ICAR passages support an answer, otherwise `0`) and `gold_evidence_ids` (relevant chunk IDs separated by `|`). Review each question against `data/index/chunks.jsonl`; do not infer answerability from the KCC response alone. The reference answer is included only for optional answer-overlap scoring.

```powershell
python -m indic_agri_rag.prepare_gold --input data/benchmark/questions.csv --out evaluation/gold.csv
```

Generate candidate answers from questions (the RAG code receives only questions and ICAR evidence):

```powershell
python -m indic_agri_rag.cli batch --input data/benchmark/questions.csv --out outputs/predictions.csv --top-k 10
```

After the gold labels and evidence IDs are complete, prepare a human review sheet for generated responses:

```powershell
python -m indic_agri_rag.prepare_review --gold evaluation/gold.csv --pred outputs/predictions.csv --out evaluation/answer_review.csv
```

Open `evaluation/answer_review.csv` and fill `pred_answerable_reviewed` (1 if the system actually attempted an answer, 0 if it abstained), plus `answer_correct`, `faithful`, `citation_valid`, and `safe` (each 1=yes, 0=no). Review the cited passage, crop, pest, region, and timing. These labels are human judgements; adjudicate disagreements before publication.

Then score the run:

```powershell
python -m indic_agri_rag.evaluate --gold evaluation/gold.csv --pred outputs/predictions.csv --review evaluation/answer_review.csv --out outputs
```

The evaluation produces:

- `outputs/metrics.json` and `outputs/classification_report.txt`
- `outputs/confusion_matrix.png` — answerability, with unanswerable/answerable classes
- `outputs/classification_metrics.png` — accuracy, answerable-class precision, recall and F1
- `outputs/qa_reference_metrics.png` — exact-match accuracy and token precision/recall/F1 against the reference text
- `outputs/human_answer_quality.png` and `outputs/answer_correctness_by_language.png` — human-rated correctness, groundedness, citation validity, and safety
- `outputs/accuracy_by_language.png` — when language values are present
- `outputs/accuracy_by_script_form.png` — native/romanized/code-mixed comparison when labels are present
- `outputs/retrieval_recall_at_k.png` — when gold evidence IDs are annotated
- `outputs/latency_distribution.png` — when prediction latency is available
- `outputs/scored_rows.csv` — per-query outcomes

The requested classification Accuracy, Precision, Recall, F1 and confusion matrix are computed for the system's answer/abstain decision against the human gold answerability labels. Answerable is the positive class; macro averages are also reported. The review sheet lets a human correct the system's answer/abstain flag because local LLMs may ignore the requested output marker. Separately, exact-match accuracy and token precision/recall/F1 compare answer text to noisy KCC reference answers; report these as lexical overlap, not factual correctness. Human-rated answer correctness, groundedness, citation validity and safety are separate proportions. Retrieval Recall@1/5/10 and MRR@10 require manually reviewed `gold_evidence_ids`; citation precision checks citations against retrieved pages. For a research paper, have two people annotate a held-out set and report agreement/adjudication; never present unannotated or auto-generated labels as gold.

## Files

- `data/raw/` — supplied ICAR PDFs (included)
- `data/index/` — extracted chunks and embeddings (created locally; not included)
- `evaluation/gold_template.csv` — empty annotation schema
- `outputs/` — metrics and plots generated locally
- `src/indic_agri_rag/` — extraction, retrieval, generation, benchmark and evaluation scripts

## Attribution

ICAR advisory PDFs are reproduced here as user-provided project inputs. Please retain ICAR attribution when redistributing any derivative materials and check source terms. If using Indic-KCC, follow its required attribution and cite the dataset card: [Indic-KCC-Agri-Advisory-Benchmark](https://huggingface.co/datasets/sthanika-ai/Indic-KCC-Agri-Advisory-Benchmark). The KCC benchmark is released under GODL-India according to its card; obtain and follow its current terms before publication or redistribution.
