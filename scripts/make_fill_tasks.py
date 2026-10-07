"""Build the task table of the supplementary gap-fill (electricity, traffic).

Reads the persisted result shards (valid = no error AND finite metrics),
finds every protocol run (model × dataset × horizon × seed) that has no valid
result for the supplementary datasets, and writes
``configs/benchmark/fill_tasks_supplementary.tsv`` — one row per sbatch task:

    task_id  dataset  seed  horizon  batch_size  models

Why a task is a (dataset, seed, horizon, group):

* Models whose last attempt was **never run** (their arrays were cancelled
  when the scope was reduced) share one task per cell at the dataset's
  default batch size, except the slow models, which get their own task so
  that one slow model cannot starve the others of wall-time.
* Models whose last attempt ran **out of memory** are retried alone with a
  smaller batch size (``OOM_RETRY_BATCH``).  Repeating an out-of-memory run at
  the same batch size is pointless.  ``spacetimeformer`` is not retried: the
  benchmark already forces its smallest viable batch size
  (``_MODEL_BATCH_OVERRIDES``) and it still ran out of memory.

The task table is the only place that decides what is re-run, so the sbatch
script stays a thin reader.  Usage::

    .venv/bin/python scripts/make_fill_tasks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from transformer_ts_ranking.reporting.loading import build_report_data  # noqa: E402

DATASET_BATCH = {"electricity": 16, "traffic": 8}
SLOW_MODELS = {"triformer", "spacetimeformer", "quatformer", "deformable_tst",
               "pathformer", "airformer"}
# Smaller batch for models that ran out of memory at the dataset default.
OOM_RETRY_BATCH = {"airformer": 4, "deformable_tst": 4, "triformer": 4,
                   "chronos_bolt": 4, "crossformer": 4, "pathformer": 2}
NOT_RETRIED = {"spacetimeformer": "already at its forced minimum batch size and still out of memory"}
EXCLUDED = {"contiformer", "contiformer_fast", "chronos2", "lag_llama", "tpatchgnn"}


def main() -> None:
    """Write the task table and print a summary."""
    data = build_report_data(REPO / "results" / "raw", REPO / "configs" / "benchmark" / "report_scope.yaml")
    scope = data.scope
    seeds = scope["seeds"]["protocol"]
    models = sorted(set(data.lt_rows["model_name"]) - EXCLUDED)
    ok = data.lt_rows[data.lt_rows["ok"]]
    have = set(zip(ok["model_name"], ok["dataset_name"], ok["horizon"], ok["seed"]))
    last = (data.lt_rows.sort_values("run_timestamp")
            .drop_duplicates(["model_name", "dataset_name", "horizon", "seed"], keep="last")
            .set_index(["model_name", "dataset_name", "horizon", "seed"]))

    def was_oom(key: tuple) -> bool:
        """Whether the last attempt of a run ended with a CUDA out-of-memory error."""
        if key not in last.index:
            return False
        err = last.loc[key, "error"]
        return isinstance(err, str) and "out of memory" in err.lower()

    rows, skipped = [], []
    for dataset in scope["long_term"]["supplementary_datasets"]:
        for horizon in data.expected_horizons(dataset):
            for seed in seeds:
                never, oom = [], []
                for model in models:
                    key = (model, dataset, horizon, seed)
                    if key in have:
                        continue
                    if was_oom(key):
                        if model in NOT_RETRIED:
                            skipped.append((*key, NOT_RETRIED[model]))
                        else:
                            oom.append(model)
                    else:
                        never.append(model)
                fast = [m for m in never if m not in SLOW_MODELS]
                if fast:
                    rows.append((dataset, seed, horizon, DATASET_BATCH[dataset], ",".join(fast)))
                for m in never:
                    if m in SLOW_MODELS:
                        rows.append((dataset, seed, horizon, DATASET_BATCH[dataset], m))
                for m in oom:
                    rows.append((dataset, seed, horizon,
                                 min(OOM_RETRY_BATCH.get(m, 4), DATASET_BATCH[dataset]), m))

    table = pd.DataFrame(rows, columns=["dataset", "seed", "horizon", "batch_size", "models"])
    # Cheapest first: electricity before traffic, short horizons before long ones.
    table = table.sort_values(["dataset", "horizon", "seed", "models"],
                              key=lambda c: c.map({"electricity": 0, "traffic": 1}) if c.name == "dataset" else c)
    table.insert(0, "task_id", range(len(table)))
    out = REPO / "configs" / "benchmark" / "fill_tasks_supplementary.tsv"
    table.to_csv(out, sep="\t", index=False)
    n_runs = table["models"].str.count(",").add(1).sum()
    print(f"{len(table)} tasks covering {n_runs} runs -> {out.relative_to(REPO)}")
    print(f"not retried ({len(skipped)} runs): {sorted({(m, d) for m, d, *_ in skipped})}")


if __name__ == "__main__":
    main()
