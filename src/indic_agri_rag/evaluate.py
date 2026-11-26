"""Research evaluation: answerability classification, QA, retrieval and plots."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report)
from .config import ROOT, OUTPUT_DIR

def norm(s): return re.sub(r"[^\w\s]", " ", str(s).casefold()).split()
def token_prf(pred, ref):
    a, b = norm(pred), norm(ref)
    if not a or not b:
        score = float(a == b)
        return score, score, score
    from collections import Counter
    common = sum((Counter(a) & Counter(b)).values())
    p, r = common / len(a), common / len(b)
    return p, r, (2*p*r/(p+r) if p+r else 0.0)

def evaluate(gold_path, pred_path, out_dir=OUTPUT_DIR, review_path=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(gold_path).fillna("")
    pred = pd.read_csv(pred_path).fillna("")
    df = gold.merge(pred, on="query_id", suffixes=("_gold", "_pred"), validate="one_to_one")
    if df.empty: raise ValueError("No matching query_id rows between gold and prediction files")
    review = pd.read_csv(review_path).fillna("") if review_path else None
    if review is not None and "pred_answerable_reviewed" in review:
        df = df.merge(review[["query_id", "pred_answerable_reviewed"]], on="query_id", how="left", validate="one_to_one")
        if df["pred_answerable_reviewed"].astype(str).str.strip().eq("").any():
            raise ValueError("Fill pred_answerable_reviewed for every row in the review sheet before scoring")
        df["pred_answerable"] = df["pred_answerable_reviewed"]
    if "is_answerable" not in df or "pred_answerable" not in df:
        raise ValueError("Gold must contain is_answerable and predictions must contain pred_answerable")
    allowed = {"0", "1", "0.0", "1.0", "true", "false", "yes", "no", "answerable", "unanswerable"}
    invalid = [x for x in df["is_answerable"].astype(str).str.strip().str.lower().unique() if x not in allowed]
    if invalid: raise ValueError(f"Gold answerability labels are blank/invalid ({invalid}). Manually annotate every row as 0 or 1 first.")
    invalid_pred = [x for x in df["pred_answerable"].astype(str).str.strip().str.lower().unique() if x not in allowed]
    if invalid_pred: raise ValueError(f"Prediction answerability labels are blank/invalid ({invalid_pred})")
    def truth(x): return int(str(x).strip().lower() in {"1", "1.0", "true", "yes", "answerable"})
    y = df["is_answerable"].map(truth)
    guessed = df["pred_answerable"].map(truth)
    labels = [0, 1]
    acc = accuracy_score(y, guessed)
    pr, rc, f1, support = precision_recall_fscore_support(y, guessed, labels=labels, zero_division=0)
    summary = {"n": int(len(df)), "answerability_accuracy": float(acc),
        "answerable_precision": float(pr[1]), "answerable_recall": float(rc[1]),
        "answerable_f1": float(f1[1]), "macro_precision": float(pr.mean()),
        "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()),
        "confusion_matrix_labels": ["unanswerable", "answerable"],
        "confusion_matrix": confusion_matrix(y, guessed, labels=labels).tolist()}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(classification_report(y, guessed,
        labels=labels, target_names=["unanswerable", "answerable"], zero_division=0), encoding="utf-8")
    # Optional open-ended QA metrics; benchmark references are noisy and not authoritative.
    if "reference_answer" in df and "pred_answer" in df:
        eligible = df[(y == 1) & (df["reference_answer"].astype(str).str.len() > 0)]
        if len(eligible):
            em = [float(" ".join(norm(a)) == " ".join(norm(b))) for a,b in zip(eligible.pred_answer, eligible.reference_answer)]
            prs = [token_prf(a,b) for a,b in zip(eligible.pred_answer, eligible.reference_answer)]
            summary["answer_exact_match_accuracy"] = float(np.mean(em))
            summary["answer_token_precision"] = float(np.mean([x[0] for x in prs]))
            summary["answer_token_recall"] = float(np.mean([x[1] for x in prs]))
            summary["answer_token_f1"] = float(np.mean([x[2] for x in prs]))
            # Keep the earlier key for compatibility with existing reports.
            summary["answer_em"] = summary["answer_exact_match_accuracy"]
    # Retrieval metrics use manually assigned gold evidence chunk IDs, not KCC answers.
    if "gold_evidence_ids" in df and "retrieved_ids" in df:
        recalls = {1: [], 5: [], 10: []}; rr = []
        for _, row in df.iterrows():
            gold_ids = {x.strip() for x in str(row.gold_evidence_ids).split("|") if x.strip()}
            got = [x.strip() for x in str(row.retrieved_ids).split("|") if x.strip()]
            if gold_ids:
                for k in recalls: recalls[k].append(len(gold_ids & set(got[:k])) / len(gold_ids))
                rr.append(next((1/(i+1) for i,x in enumerate(got[:10]) if x in gold_ids), 0.0))
        for k, vals in recalls.items():
            if vals: summary[f"retrieval_recall_at_{k}"] = float(np.mean(vals))
        if rr: summary["retrieval_mrr_at_10"] = float(np.mean(rr))
    # Citation diagnostics: page citations should point to retrieved passages.
    if "pred_answer" in df and "retrieved_citations" in df:
        import re
        cited_total = cited_supported = answers_with_citations = 0
        for _, row in df.iterrows():
            found = re.findall(r"\[([^\]]+),\s*p(?:age)?\.?\s*(\d+)\]", str(row.pred_answer), flags=re.I)
            if found:
                answers_with_citations += int(bool(truth(row.pred_answerable)))
                retrieved = set(x.strip() for x in str(row.retrieved_citations).split("|") if x.strip())
                for source, page in found:
                    cited_total += 1
                    cited_supported += int(f"{source.strip()}:p{page}" in retrieved)
        if cited_total:
            summary["citation_precision"] = cited_supported / cited_total
            summary["citation_coverage"] = answers_with_citations / max(1, int(np.sum(guessed)))
    if "latency_seconds" in df:
        latency = pd.to_numeric(df["latency_seconds"], errors="coerce").dropna()
        if len(latency):
            summary["latency_seconds_mean"] = float(latency.mean())
            summary["latency_seconds_median"] = float(latency.median())
            summary["latency_seconds_p95"] = float(latency.quantile(.95))
    if review_path:
        reviewed = review.merge(df, on="query_id", how="inner", suffixes=("_review", ""), validate="one_to_one")
        if reviewed.empty: raise ValueError("Review file has no query_id values in common with gold/predictions")
        dims = {"answer_correct": "human_answer_correctness_rate", "faithful": "human_groundedness_rate",
                "citation_valid": "human_citation_validity_rate", "safe": "human_safety_rate"}
        quality = {}
        for col, metric in dims.items():
            if col in reviewed:
                vals = pd.to_numeric(reviewed[col], errors="coerce").dropna()
                if len(vals) != len(reviewed): raise ValueError(f"Complete every {col} label (1=yes, 0=no) before scoring")
                if len(vals):
                    if not vals.isin([0, 1]).all(): raise ValueError(f"Review field {col} must contain only 0/1 labels")
                    quality[metric] = float(vals.mean())
                    n=len(vals); se=(quality[metric]*(1-quality[metric])/n)**0.5
                    quality[metric+"_ci95"] = [max(0.0, quality[metric]-1.96*se), min(1.0, quality[metric]+1.96*se)]
        summary.update(quality)
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        available=[(k,v) for k,v in dims.items() if v in quality]
        if available:
            labels={"answer_correct":"Answer correct", "faithful":"Evidence grounded", "citation_valid":"Citation valid", "safe":"Safe"}
            fig,ax=plt.subplots(figsize=(8,4)); names=[labels[k] for k,_ in available]; vals=[quality[v] for _,v in available]
            bars=ax.bar(names,vals,color="#54a24b"); ax.set(ylim=(0,1),ylabel="Human-rated proportion",title="Human review of answer quality"); ax.bar_label(bars,fmt="%.2f",padding=3); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out_dir/"human_answer_quality.png",dpi=300); plt.close(fig)
        language_col = "language_review" if "language_review" in reviewed else ("language" if "language" in reviewed else None)
        if language_col and "answer_correct" in reviewed:
            d=reviewed.assign(answer_correct_num=pd.to_numeric(reviewed.answer_correct,errors="coerce")).groupby(language_col).answer_correct_num.mean().dropna().sort_values()
            if len(d):
                fig,ax=plt.subplots(figsize=(8,max(4,len(d)*.38))); d.plot.barh(ax=ax,color="#f58518"); ax.set(xlim=(0,1),xlabel="Human-rated correctness rate",ylabel="Language",title="Answer correctness by language"); fig.tight_layout(); fig.savefig(out_dir/"answer_correctness_by_language.png",dpi=300); plt.close(fig)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Confusion matrix and publication-friendly metric charts.
    cm = confusion_matrix(y, guessed, labels=labels)
    fig, ax = plt.subplots(figsize=(5.4, 4.5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Unanswerable", "Answerable"], yticklabels=["Unanswerable", "Answerable"], ax=ax)
    ax.set(xlabel="Predicted label", ylabel="Gold label", title="Answerability confusion matrix"); fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=300); plt.close(fig)
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    vals = [acc, pr[1], rc[1], f1[1]]
    fig, ax = plt.subplots(figsize=(7, 4)); bars=ax.bar(names, vals, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Answerability classification metrics"); ax.bar_label(bars, fmt="%.3f", padding=3); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(out_dir / "classification_metrics.png", dpi=300); plt.close(fig)
    if "language_gold" in df:
        d=df.assign(correct=(y.values==guessed.values)).groupby("language_gold").correct.mean().sort_values()
        if len(d):
            fig,ax=plt.subplots(figsize=(8,max(4,len(d)*.38))); d.plot.barh(ax=ax,color="#4c78a8"); ax.set(xlim=(0,1),xlabel="Answerability accuracy",ylabel="Language",title="Accuracy by language"); fig.tight_layout(); fig.savefig(out_dir/"accuracy_by_language.png",dpi=300); plt.close(fig)
    if "script_form" in df:
        d=df.assign(correct=(y.values==guessed.values)).groupby("script_form").correct.mean().sort_values()
        if len(d):
            fig,ax=plt.subplots(figsize=(7,4)); d.plot.bar(ax=ax,color="#72b7b2"); ax.set(ylim=(0,1),xlabel="Input form",ylabel="Answerability accuracy",title="Accuracy by script form"); fig.tight_layout(); fig.savefig(out_dir/"accuracy_by_script_form.png",dpi=300); plt.close(fig)
    if "answer_exact_match_accuracy" in summary:
        names=["Exact match", "Token precision", "Token recall", "Token F1"]
        vals=[summary["answer_exact_match_accuracy"],summary["answer_token_precision"],summary["answer_token_recall"],summary["answer_token_f1"]]
        fig,ax=plt.subplots(figsize=(8,4)); bars=ax.bar(names,vals,color=["#4c78a8","#f58518","#54a24b","#e45756"]); ax.set(ylim=(0,1),ylabel="Score",title="Reference-based QA overlap metrics"); ax.bar_label(bars,fmt="%.3f",padding=3); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(out_dir/"qa_reference_metrics.png",dpi=300); plt.close(fig)
    if "latency_seconds" in df:
        latency=pd.to_numeric(df["latency_seconds"],errors="coerce").dropna()
        if len(latency):
            fig,ax=plt.subplots(figsize=(7,4)); ax.hist(latency,bins=min(30,max(5,len(latency)//3)),color="#b279a2",edgecolor="white"); ax.set(xlabel="Seconds per query",ylabel="Count",title="End-to-end response latency"); fig.tight_layout(); fig.savefig(out_dir/"latency_distribution.png",dpi=300); plt.close(fig)
    if "gold_evidence_ids" in df and "retrieved_ids" in df:
        curve=[]
        for k in [1,2,3,4,5,10]:
            vals=[]
            for _,row in df.iterrows():
                gs={x.strip() for x in str(row.gold_evidence_ids).split("|") if x.strip()}
                got=[x.strip() for x in str(row.retrieved_ids).split("|") if x.strip()]
                if gs: vals.append(len(gs & set(got[:k]))/len(gs))
            if vals: curve.append((k,float(np.mean(vals))))
        if curve:
            fig,ax=plt.subplots(figsize=(6,4)); ax.plot([x[0] for x in curve],[x[1] for x in curve],marker="o",color="#54a24b"); ax.set(xticks=[x[0] for x in curve],ylim=(0,1),xlabel="Top-k retrieved chunks",ylabel="Evidence Recall@k",title="Evidence retrieval recall"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir/"retrieval_recall_at_k.png",dpi=300); plt.close(fig)
    pd.DataFrame({"gold":y,"predicted":guessed,"correct":y.values==guessed.values}).to_csv(out_dir/"scored_rows.csv",index=False)
    return summary

def main():
    p=argparse.ArgumentParser(); p.add_argument("--gold",default=str(ROOT/"evaluation"/"gold.csv")); p.add_argument("--pred",default=str(ROOT/"outputs"/"predictions.csv")); p.add_argument("--review",default=None); p.add_argument("--out",default=str(OUTPUT_DIR)); a=p.parse_args()
    print(json.dumps(evaluate(a.gold,a.pred,a.out,a.review),indent=2))
if __name__=="__main__": main()
