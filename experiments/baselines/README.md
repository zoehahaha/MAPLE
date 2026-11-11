# Baseline evaluation

The paper compares Full oracle, Exact cache, LOTUS (prefix/full),
SemWeave (0.70/0.80/0.90), GPTCache (0.90/0.975), Native Fusion,
SemReuse PR, and PREP. GPTCache (0.90) compares full requests;
GPTCache (0.975) compares operator inputs.

`registry.json` lists the configurations. `evaluate.py` measures per-query
precision and recall from supplied frozen Boolean outputs and references.
It does not make live model requests.

Supply a complete Boolean output for each comparable query-record pair.
The evaluator rejects absent or non-Boolean predictions. References can use
the bundled `oracle_answers.json` wrapper or a direct query-to-record map.

Table II reports the common certification comparison. Table III reports
quality under each method's original execution protocol. The paper values
are in `reference/tables/table_ii.csv` and `table_iii.csv`.
