"""Describe paired class effects from the complete follow-up score matrix."""

import argparse
from hashlib import sha256
import json
import math
from statistics import mean, stdev

from scipy.stats import t

from _dengine_cli_paths import cli_input_path, cli_output_path

from .design import ARMS, SPEC, write_json


def interval(values):
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Require two or more finite cohort means")
    middle, df = mean(values), len(values) - 1
    half = float(t.ppf(.975, df) * stdev(values) / math.sqrt(len(values)))
    return {"mean_pp": middle, "low_pp": middle - half, "high_pp": middle + half, "df": df}


def summarize_classes(rows):
    """Summarize the dense-minus-expert change in F1 loss by digit."""
    expected = {(cohort, slot) for cohort in range(1, SPEC["cohorts"] + 1)
                for slot in range(1, SPEC["starts_per_cohort"] + 1)}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Require the complete cohort and start matrix")
    effects = {}
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("cohort")) is not int
                or type(row.get("slot")) is not int):
            raise ValueError("Require integer cohort and start identifiers")
        key = (row["cohort"], row["slot"])
        if key not in expected or key in effects:
            raise ValueError("Duplicate or unexpected cohort and start")
        scores = row.get("scores")
        if not isinstance(scores, dict) or any(arm not in scores for arm in ARMS):
            raise ValueError("Require scores for all six study arms")
        for arm in ARMS:
            metrics = scores[arm]
            values = metrics.get("f1") if isinstance(metrics, dict) else None
            if (not isinstance(values, list) or len(values) != 10
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or not 0 <= value <= 1 for value in values)):
                raise ValueError("Require ten finite F1 scores in [0, 1] for every arm")
        effects[key] = {
            reference: [
                100 * ((scores[reference + "-clean"]["f1"][digit] - scores[reference + "-proxy"]["f1"][digit])
                       - (scores["moe-clean"]["f1"][digit] - scores["moe-proxy"]["f1"][digit]))
                for digit in range(10)
            ] for reference in ("dense", "wide")
        }

    def summarize(reference, digit):
        grouped = [mean(effects[(cohort, slot)][reference][digit] for slot in range(1, SPEC["starts_per_cohort"] + 1))
                   for cohort in range(1, SPEC["cohorts"] + 1)]
        return {"cohort_means_pp": grouped, "summary": interval(grouped)}

    classes = []
    for digit in range(10):
        classes.append({"digit": digit, **summarize("dense", digit), "capacity_control": summarize("wide", digit)})
    return {
        "scope": "secondary_exploratory", "contrast": "dense F1 loss minus expert F1 loss",
        "capacity_control_contrast": "wide F1 loss minus expert F1 loss",
        "units": "F1 percentage points", "interval": "two-sided 95% Student t on cohort means",
        "interval_adjustment": "none", "primary_digit": 4,
        "cohorts": SPEC["cohorts"], "starts_per_cohort": SPEC["starts_per_cohort"],
        "matched_starts": len(rows), "classes": classes,
    }


def render_class_report(result):
    lines = [
        "# Paired class comparisons", "",
        "This secondary exploratory view describes all ten digits. The prespecified primary outcome is digit 4.", "",
        "The effects compare each dense model's F1 loss with the expert model's F1 loss, measured in percentage points. "
        "Positive values favor the expert model. Negative values favor the dense model in that comparison.", "",
        f"Each interval uses {result['cohorts']} cohort means after averaging "
        f"{result['starts_per_cohort']} matched starts within each cohort. "
        "These descriptive 95% Student t intervals are unadjusted for the twenty reported class contrasts. "
        "They describe variation across training cohorts conditional on the common evaluation set.", "",
    ]
    for title, key in (("Dense versus expert", None), ("Wide capacity control versus expert", "capacity_control")):
        lines.extend([f"## {title}", "",
                      "| Digit | Mean effect (points) | Lower 95% limit | Upper 95% limit | Degrees of freedom |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for row in result["classes"]:
            contrast = row[key] if key else row
            summary = contrast["summary"]
            lines.append(f"| {row['digit']} | {summary['mean_pp']:+.3f} | "
                         f"{summary['low_pp']:+.3f} | {summary['high_pp']:+.3f} | {summary['df']} |")
        lines.append("")
    if "input_results_sha256" in result:
        lines.extend(["", f"Input score JSON SHA-256 `{result['input_results_sha256']}`."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=cli_input_path, help="Existing study-results.json")
    parser.add_argument("out", type=cli_output_path, help="Fresh directory for the secondary report")
    args = parser.parse_args(argv)
    raw = args.results.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("Expected a study result object")
    result = summarize_classes(record.get("matched_comparisons"))
    result["input_results_sha256"] = sha256(raw).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "class-comparisons.json", result)
    (args.out / "class-comparisons.md").write_text(render_class_report(result), encoding="utf-8")
    print(f"Wrote paired class comparisons to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
