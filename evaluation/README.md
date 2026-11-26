# Gold labels for evaluation

`gold_template.csv` is a schema example. Export an evaluation question set, then use `python -m indic_agri_rag.prepare_gold` to create a populated sheet. A human annotator must fill `is_answerable` with 1 only when the ICAR corpus supports an answer and 0 when it does not. Add all judged supporting chunk IDs (from `data/index/chunks.jsonl`) to `gold_evidence_ids`, separated by `|`.

Do not leave answerability blank. The evaluator intentionally stops when labels are incomplete. The KCC benchmark's reference answer may be retained for optional reference-overlap QA scoring, but it is not the gold truth for this ICAR-grounded system.
