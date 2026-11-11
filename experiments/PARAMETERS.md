# PREP parameter settings

`experiments/run.py` is intentionally configuration-only. It passes named
variables to the package and contains no paper-specific scientific values.
The paper configuration is [`configs/paper.json`](configs/paper.json); the
small CPU example uses [`configs/tiny_demo.json`](configs/tiny_demo.json).

`base_config` loads a JSON configuration relative to the selected file.
Selected fields override the base. CFPB and 20NG scaling configurations
inherit `paper.json`, including its failure budget and control requests.

The paper configuration fixes precision and recall at 0.90 and uses a joint
per-query failure budget of 0.05. When both prediction signs occur, PREP
splits that budget equally between the two finite-population bounds; a single
present sign receives the entire budget. Discovery purchases 2% of the input
rows per round, requires two consecutive stability passes, and stops by round
8. Scoring examples are drawn from the actual Discovery complement.

The input-condition family uses token presence, four-character prefixes,
and bounded structural cues. The candidate caps are 24 preliminary features,
12 single conditions, a six-condition conjunction frontier, and eight retained
variants per target. Planning uses 32 named scenarios. GGR completely scores
up to eight legal single-slot neighbors at a time, with 32 complete neighbor
scores and eight accepted moves as global caps. The seven anonymous controls
are charged in the complete request union.

Each workload supplies explicit, label-independent Discovery and scoring
example orders, plus a certification order for each query. Planning scenarios
use a named namespace and configured scenario count for reproducible forecasts.

Full input paths and dimensions are registered in `data/datasets.json`.
`--limit-rows` and `--limit-queries` select input prefixes for execution
checks and query scaling. Full replays use all rows and queries by default.
