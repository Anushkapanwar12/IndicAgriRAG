"""Command-line tools for corpus indexing, farmer QA and evaluation exports."""
import argparse, csv, json, sys
import re
from pathlib import Path
from time import perf_counter
from .config import ROOT, RAW_DIR, INDEX_DIR, OUTPUT_DIR, DEFAULT_TOP_K
from .corpus import build_corpus

def index_cmd(_):
    chunks=build_corpus(); print(f"Indexed {len(chunks)} passages in {INDEX_DIR}")
    from .retrieval import HybridRetriever
    HybridRetriever()
    print("Multilingual embedding index created.")

def ask_cmd(a):
    from .retrieval import HybridRetriever
    from .answer import answer_question
    r=HybridRetriever(); t=perf_counter(); passages=r.search(a.question,a.top_k)
    answer=answer_question(a.question,passages)
    answer = re.sub(r"^\s*<<(?:UN)?ANSWERABLE>>\s*", "", answer, flags=re.I)
    print("\n"+answer+"\n\nRetrieved evidence:")
    for p in passages: print(f"- [{p['source']}, p. {p['page']}] (chunk {p['id']}, score {p['score']:.3f}) {p['text'][:320]}...")
    print(f"\nLatency: {perf_counter()-t:.2f}s")

def export_cmd(a):
    from datasets import load_dataset
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]
    languages=a.languages or ["English","Hindi","Bengali","Gujarati","Kannada","Malayalam","Marathi","Odia","Punjabi","Tamil","Telugu"]
    for lang in languages:
        print(f"Loading {lang} test split (gated: HF access may need approval/login)...")
        ds=load_dataset("sthanika-ai/Indic-KCC-Agri-Advisory-Benchmark",lang,split="test")
        for i,row in enumerate(ds):
            rows.append({"query_id":f"{lang.lower()}-{row.get('id',i)}","question":row.get("question",row.get("text","")),
                "language":lang,"script_form":row.get("script_form",""),"query_type":row.get("query_type",""),
                "reference_answer":row.get("answer",""),"benchmark_warning":"KCC reference is noisy; benchmark only; not advisory evidence"})
    dest=Path(a.out); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} evaluation questions to {dest}. Create a separate annotated gold.csv; do not use this file as training data.")

def run_batch_cmd(a):
    import pandas as pd
    from .retrieval import HybridRetriever
    from .answer import answer_question
    inp=pd.read_csv(a.input).fillna(""); r=HybridRetriever(); rows=[]
    for _,row in inp.iterrows():
        q=str(row["question"]); t=perf_counter(); hits=r.search(q,a.top_k); text=answer_question(q,hits)
        marker=re.match(r"\s*<<(ANSWERABLE|UNANSWERABLE)>>",text,re.I)
        if marker: predicted=int(marker.group(1).upper()=="ANSWERABLE")
        else: predicted=int(not any(s in text.casefold() for s in ["not enough information","does not contain enough","unable to answer","insufficient evidence"]))
        clean_text=re.sub(r"^\s*<<(?:UN)?ANSWERABLE>>\s*", "", text, flags=re.I)
        rows.append({"query_id":row["query_id"],"pred_answer":clean_text,"pred_answerable":predicted,
            "retrieved_ids":"|".join(h["id"] for h in hits),"retrieved_citations":"|".join(f"{h['source']}:p{h['page']}" for h in hits),"latency_seconds":round(perf_counter()-t,3),
            "language":row.get("language","")})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out,index=False,encoding="utf-8-sig"); print(f"Saved {len(rows)} predictions to {a.out}")

def main():
    p=argparse.ArgumentParser(prog="indic-agri-rag"); sub=p.add_subparsers(required=True)
    x=sub.add_parser("index",help="Extract ICAR PDFs and build retrieval index"); x.set_defaults(func=index_cmd)
    x=sub.add_parser("ask",help="Ask a question against indexed ICAR evidence"); x.add_argument("question"); x.add_argument("--top-k",type=int,default=DEFAULT_TOP_K); x.set_defaults(func=ask_cmd)
    x=sub.add_parser("benchmark-questions",help="Download benchmark questions for evaluation only"); x.add_argument("--languages",nargs="+"); x.add_argument("--out",default=str(ROOT/"data"/"benchmark"/"questions.csv")); x.set_defaults(func=export_cmd)
    x=sub.add_parser("batch",help="Generate predictions for an input question CSV"); x.add_argument("--input",required=True); x.add_argument("--out",default=str(OUTPUT_DIR/"predictions.csv")); x.add_argument("--top-k",type=int,default=DEFAULT_TOP_K); x.set_defaults(func=run_batch_cmd)
    a=p.parse_args(); a.func(a)
if __name__=="__main__": main()
