"""Create a spreadsheet template for human review of generated answers."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("--gold",default="evaluation/gold.csv"); p.add_argument("--pred",default="outputs/predictions.csv"); p.add_argument("--out",default="evaluation/answer_review.csv")
    a=p.parse_args(); gold=pd.read_csv(a.gold).fillna(""); pred=pd.read_csv(a.pred).fillna("")
    df=gold.merge(pred,on="query_id",suffixes=("_gold","_pred"),validate="one_to_one")
    language=df["language_gold"] if "language_gold" in df else df.get("language","")
    fields={"query_id":df.query_id,"language":language,"question":df.get("question",""),
        "candidate_answer":df.get("pred_answer",""),"gold_answerable":df.get("is_answerable",""),
        "system_pred_answerable":df.get("pred_answerable",""),"pred_answerable_reviewed":"","reference_answer_optional":df.get("reference_answer",""),
        "answer_correct":"","faithful":"","citation_valid":"","safe":"","reviewer_notes":""}
    out=pd.DataFrame(fields); Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False,encoding="utf-8-sig")
    print(f"Created {a.out} ({len(out)} rows). Review every row: pred_answerable_reviewed (1=system answered, 0=system abstained), answer_correct, faithful, citation_valid, safe (1=yes, 0=no). Adjudicate disagreements before publication.")
if __name__=="__main__": main()
