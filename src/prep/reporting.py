from __future__ import annotations

import csv
from pathlib import Path
from shutil import copyfile
from typing import Iterable


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown(rows: list[dict[str, str]], fields: Iterable[str]) -> str:
    fields = list(fields)
    output = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        output.append("| " + " | ".join(row.get(field, "") for field in fields) + " |")
    return "\n".join(output) + "\n"


def regenerate(repo: Path, output: Path) -> dict[str, object]:
    reference = repo / "reference"
    output.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for number, fields in {
        "table_i": ("workload", "rows", "queries", "required_outputs", "rare_positive_queries"),
        "table_ii": ("method", "workload", "call_saving_percent", "token_saving_percent", "speedup", "miss_pr_percent", "footnote"),
        "table_iii": ("method", "workload", "miss_pr_percent", "mean_miss_pr_percent"),
        "table_iv": ("section", "method", "workload", "call_saving_percent"),
    }.items():
        values = _rows(reference / "tables" / f"{number}.csv")
        target = output / f"{number}.md"
        content = _markdown(values, fields)
        if number == "table_ii":
            content += "\nF: Full oracle evaluation on Snips and arXiv.\n"
        target.write_text(content, encoding="utf-8")
        generated.append(str(target))
    for name in ("figure3_certification", "figure4_sensitivity",
                 "figure5_variability", "figure6_20ng_scaling"):
        image = reference / "figures" / f"{name}.png"
        target = output / image.name
        if image.resolve() != target.resolve():
            copyfile(image, target)
        generated.append(str(target))
        values = _rows(reference / "figures" / f"{name}.csv")
        summary = output / f"{name}.md"
        content = _markdown(values, values[0])
        content += "\nThis table summarizes values printed in the paper. "
        content += "See the accompanying paper figure for the complete plot.\n"
        summary.write_text(content, encoding="utf-8")
        generated.append(str(summary))
    return {"status": "COMPLETE", "generated": generated}


def validate_reference(repo: Path) -> dict[str, object]:
    tables = repo / "reference" / "tables"
    table_i = _rows(tables / "table_i.csv")
    table_ii = _rows(tables / "table_ii.csv")
    table_iii = _rows(tables / "table_iii.csv")
    table_iv = _rows(tables / "table_iv.csv")
    prep = {(row["workload"], row["method"]): row for row in table_ii if row["method"] == "PREP"}
    expected = {
        "Snips": (17.10, 17.02, 1.09, 0.0),
        "arXiv": (11.63, 11.52, 1.03, 0.0),
        "Amazon-Elec": (8.71, 8.63, 1.06, 0.0),
        "20 Newsgroups": (12.26, 12.23, 1.07, 0.0),
    }
    checks: dict[str, bool] = {}
    for workload, values in expected.items():
        row = prep[(workload, "PREP")]
        actual = tuple(float(row[field]) for field in (
            "call_saving_percent", "token_saving_percent", "speedup", "miss_pr_percent"
        ))
        checks[f"paper_main::{workload}"] = actual == values
    checks["table_i_rows"] = len(table_i) == 4
    checks["table_ii_rows"] = len(table_ii) == 48
    checks["table_iii_rows"] = len(table_iii) == 40
    checks["table_iv_rows"] = len(table_iv) == 22
    checks["native_fusion_negative_tokens_preserved"] = any(
        row["method"] == "Native Fusion" and float(row["token_saving_percent"]) < 0
        for row in table_ii
    )
    checks["semreuse_full_oracle_footnote_preserved"] = sum(
        row.get("footnote") == "F" for row in table_ii if row["method"] == "SemReusePR"
    ) == 2
    gptcache = [row for row in table_iii if row["method"] == "GPTCache .90"]
    checks["table_iii_gptcache_full_request"] = (
        {row["workload"]: float(row["miss_pr_percent"]) for row in gptcache}
        == {"Snips": 100.0, "arXiv": 100.0, "Amazon-Elec": 100.0, "20 Newsgroups": 95.7}
        and all(float(row["mean_miss_pr_percent"]) == 98.9 for row in gptcache)
    )
    figures = repo / "reference" / "figures"
    checks["paper_figure_images"] = all((figures / f"{name}.png").is_file() for name in (
        "figure3_certification", "figure4_sensitivity", "figure5_variability", "figure6_20ng_scaling"
    ))
    figure5 = _rows(figures / "figure5_variability.csv")
    checks["figure5_paper_annotations"] = {
        row["workload"]: float(row["annotated_call_saving_percent"]) for row in figure5
    } == {"Snips": 17.05, "arXiv": 11.23, "Amazon-Elec": 8.71, "20 Newsgroups": 12.57}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    return result

