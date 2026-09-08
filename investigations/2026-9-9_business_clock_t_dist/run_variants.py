"""Build and score variants.

    python run_variants.py --all
    python run_variants.py --variants baseline,rho_off
    python run_variants.py --sweep variants/sweeps/example.yaml
    python run_variants.py --variants baseline --holdout --i-know

A variant run is pure evaluation: nothing is fitted, and the register bank is
shared across every variant (see export/cache.py).

Sweep-generated variants are written to `variants/generated/`, never to the top
level of `variants/` - that keeps `fvmodel.variants.list_variants()` (the shipped
set) unchanged by running a sweep.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

INV = Path(__file__).resolve().parent
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from export.build_export import build            # noqa: E402
from fvmodel import config                       # noqa: E402
from fvmodel.variants import GENERATED, expand_sweep, list_variants  # noqa: E402

HOLDOUT_LOG = INV / "runs" / "holdout_log.tsv"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(INV), text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    w = config.read_config()["window"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--i-know", dest="i_know", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()

    if a.holdout and not a.i_know:
        raise SystemExit(
            "--holdout scores the reserved window 2026-08-26 .. 08-31. It is for the "
            "ONE chosen variant, once, at the end. If that is what you are doing, "
            "pass --i-know as well; the run is logged to runs/holdout_log.tsv.")

    if a.sweep:
        names = []
        spec = yaml.safe_load(Path(a.sweep).read_text())
        GENERATED.mkdir(parents=True, exist_ok=True)
        for v in expand_sweep(spec):
            p = GENERATED / ("%s.yaml" % v["name"])
            p.write_text(yaml.safe_dump(v, sort_keys=False))
            names.append(v["name"])
    elif a.all:
        names = list_variants()
    else:
        names = [s for s in a.variants.split(",") if s]
    if not names:
        raise SystemExit("nothing to run; pass --variants, --all or --sweep")

    t0 = w["holdout_t0"] if a.holdout else w["backtest_t0"]
    t1 = w["backtest_t1"] if a.holdout else w["select_t1"]
    tag = "holdout" if a.holdout else "select"
    print("window %s: %d .. %d, %d variant(s)" % (tag, t0, t1, len(names)))

    # baseline first, and always: every other variant is scored on ITS strike grid
    # and reported as a delta against it
    names = ["baseline"] + [n for n in names if n != "baseline"]

    results, ref_path = {}, None
    for name in names:
        start = time.time()
        path = build(name, t0, t1,
                     out_path=INV / "runs" / tag / ("%s.parquet" % name))
        took = time.time() - start
        if name == "baseline":
            ref_path = path
        print("  %-22s built in %.1f s" % (name, took))
        if not a.no_score:
            from score.scorecard import score_variant
            results[name] = score_variant(name, path, ref_path=ref_path)
            results[name]["build_seconds"] = round(took, 1)

    if not a.no_score:
        out = INV / "runs" / ("scores_%s.json" % tag)
        out.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print("wrote %s" % out)

    if a.holdout:
        HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not HOLDOUT_LOG.exists()
        with open(HOLDOUT_LOG, "a", encoding="utf-8") as fh:
            if new:
                fh.write("utc\tgit_sha\tvariant\tt0\tt1\tlog_loss\tnear30_log_loss\n")
            for name in names:
                r = results.get(name, {})
                fh.write("%s\t%s\t%s\t%d\t%d\t%.6f\t%.6f\n" % (
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    _git_sha(), name, t0, t1,
                    r.get("log_loss", float("nan")),
                    r.get("near30_log_loss", float("nan"))))
        print("logged %d holdout run(s) to %s" % (len(names), HOLDOUT_LOG))


if __name__ == "__main__":
    main()
