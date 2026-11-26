"""Create a gold annotation sheet from benchmark questions; labels must be human supplied."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/benchmark/questions.csv"); p.add_argument("--out",default="evaluation/gold.csv")
    a=p.parse_args(); df=pd.read_csv(a.input).fillna("")
    out=pd.DataFrame({"query_id":df.query_id,"question":df.question,"language":df.language,
        "script_form":df.get("script_form",""),"is_answerable":"","gold_evidence_ids":"",
        "reference_answer":df.get("reference_answer",""),"annotation_notes":""})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False,encoding="utf-8-sig")
    print(f"Created {a.out} with {len(out)} rows. Manually label is_answerable (0/1) and gold_evidence_ids (chunk IDs separated by |). Do not score until gold labels are complete.")
if __name__=="__main__": main()
