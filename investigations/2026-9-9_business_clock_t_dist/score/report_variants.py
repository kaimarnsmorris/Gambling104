"""report_variants.html: baseline and every shipped variant, on the selection window.

Sorted by the near-strike last-30 s log-loss with the pooled log-loss beside it, which
is the brief's ordering: the cell where the model is actually asked a hard question,
with the pooled number next to it so a variant that wins the cell by losing everywhere
else is visible.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

INV = Path(__file__).resolve().parents[1]
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from fvmodel.variants import read_variant           # noqa: E402

SCORES = INV / "runs" / "scores_select.json"
OUT = INV / "report_variants.html"

# Manual findings: for each shipped variant, whether the observed scorecard result
# matches the expectation written in its own variants/*.yaml `notes` field, checked
# against the real panel (see task-10-report.md for the full derivation of each).
# Per task instructions, a disagreement is not smoothed over - it is written here.
FINDINGS = {
    "alpha_half": (
        "No detectable effect: p_model is bit-for-bit identical to baseline. The "
        "order-book imbalance feeding alpha_scale is uniformly zero across this "
        "panel's book capture, so there is nothing for alpha_scale to scale - "
        "confirming variants/README.md's own caution that the book captures do not "
        "overlap the Chainlink history used here."),
    "alpha_x2": (
        "Same as alpha_half: p_model is bit-for-bit identical to baseline because "
        "the order-book imbalance input is uniformly zero over this panel."),
    "no_alpha": (
        "Confirmed, more strongly than the note anticipated: m_Y and p_model are "
        "bit-for-bit identical to baseline and the pooled log-loss delta is exactly "
        "0.0 - not because a nonzero alpha effect happened to cancel, but because "
        "alpha_scale=1 already computes to zero imbalance on this panel."),
    "basis_alt": (
        "Partially contradicts its note. m_Y is bit-for-bit untouched and eps_bar "
        "does move, as predicted. But 'carry' is also bit-for-bit identical to "
        "baseline, contradicting the note's explicit claim that basis_alt moves "
        "'eps_bar and the carry ... not just the tracked level in isolation.'"),
    "eps_x0": (
        "Direction confirmed, wording overstated: E[resid^2/Var_Y] in the 1-5 s "
        "bucket rises from 0.241 (baseline) to 0.312, and log-loss worsens near "
        "expiry. But the ratio never crosses 1.0, so 'above one' is not literally "
        "true here - it is a rise toward one, not past it."),
    "eps_x2": (
        "CONTRADICTS its note: 'Should lose against baseline if the fitted "
        "sigma_eps is right.' It wins on every metric here (near30 -0.0309, pooled "
        "-0.00059) - one of the two largest near30 improvements of the 16 shipped "
        "variants. Read with eps_x0's loss, this points at the fitted sigma_eps "
        "being too small on this real 5 m panel."),
    "fat_tails": (
        "Confirms its note precisely: level_by_tte and qlike_excess_by_tte are "
        "bit-for-bit identical to baseline at every horizon - kappa_tail rescales "
        "sigma to hold Var(z) exactly fixed, so the second-moment metrics cannot "
        "move by construction - while log-loss shifts a little, a pure shape "
        "effect exactly as claimed."),
    "thin_tails": (
        "Confirms its note precisely, the same way fat_tails does: level_by_tte "
        "and qlike_excess_by_tte are bit-for-bit identical to baseline at every "
        "horizon, while log-loss moves a little on its own."),
    "market_observable": (
        "CONTRADICTS its note: 'Expected to lose on every calibration metric, most "
        "of all in the last 30 s.' Instead near30 log-loss IMPROVES by -0.0301 "
        "(third-best of the 16 shipped variants) even though pooled log-loss "
        "worsens slightly (+0.0037). The cell the note singles out as showing the "
        "lag edge 'most of all' is exactly the one where this variant does best."),
    "normal_tail": (
        "THE HEADLINE FINDING, and it is a scorecard limitation, not a model one. "
        "The note expects a clear near-expiry loss; every scorecard number here is "
        "bit-for-bit identical to baseline. score_variant recomputes p via a "
        "hardcoded stats.t.cdf(nu, mu, sigma_t) (score/scorecard.py::_p_at), and "
        "tail_family='normal' only flips an internal .family flag in "
        "fvmodel/overrides.py::_scaled_tail without touching (nu, mu, sigma_t) - so "
        "this scorecard cannot see the override at all. The already-exported "
        "p_model column, built by the real family-aware pipeline, DOES differ "
        "meaningfully from baseline (mean |delta p_model| = 0.0251, max = 0.127); "
        "the model is doing its job, this report just never reads the column that "
        "would show it."),
    "rho_off": (
        "Direction confirmed, threshold not met: level_by_tte rises above baseline "
        "at all six tte buckets, but only the longest (121-300 s: 1.139) actually "
        "crosses 1.0 - the other five (0.255, 0.329, 0.490, 0.678, 0.849) stay "
        "below one. 'Rises above one across the board' is not literally true here."),
    "short_vol_minus20": (
        "CONTRADICTS its own falsifiability clause: 'If it moves both [near30 and "
        "pooled], the knob is not doing what it says.' It moves both, by the "
        "largest margins of any shipped variant - near30 +0.0358 (worst of the "
        "16), pooled +0.0051 (second-worst after rho_off)."),
    "shrink_05": (
        "As the note anticipates, the interesting comparison is against "
        "short_vol_minus20, not baseline: shrink_05 is the milder, more "
        "calibration-friendly remedy for the same short-horizon complaint - "
        "smaller near30 damage (+0.0174 vs +0.0358) and one of only two shipped "
        "variants that actually improves pooled log-loss (-0.0017)."),
    "vol_minus10": (
        "The predicted diagnostic fires: the two sides of the vol_plus10/"
        "vol_minus10 pair are not symmetric. vol_minus10 worsens both near30 "
        "(+0.0148) and pooled (+0.0022) log-loss. Per the note, 'if one side wins "
        "clearly, the shipped variance level is off in that direction' - it does, "
        "pointing at the shipped vol level being too low."),
    "vol_plus10": (
        "The predicted diagnostic fires: vol_plus10 improves both near30 (-0.0168) "
        "and pooled (-0.0009) log-loss while vol_minus10 worsens both. The shipped "
        "variance level appears too low on this real panel, consistent with "
        "eps_x2's finding above."),
}

CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --line:#d8d8d8;
        --muted:#666; --good:#0a7; --bad:#c33; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#141414; --line:#333; --muted:#999; } }
body { margin:0 auto; padding:2rem 1.25rem; max-width:70rem; color:var(--fg);
       background:var(--bg); font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }
h1 { font-size:1.5rem; margin:0 0 .25rem; }
.sub { color:var(--muted); margin:0 0 2rem; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { padding:.4rem .6rem; border-bottom:1px solid var(--line); text-align:right;
        white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { border-bottom:2px solid var(--line); font-weight:600; }
tr.base { font-weight:600; }
.d-good { color:var(--good); } .d-bad { color:var(--bad); }
.note { margin:.4rem 0 1.4rem; color:var(--muted); max-width:60rem; }
h3 { margin:1.6rem 0 .2rem; font-size:1rem; }
.flag { color:var(--bad); font-weight:600; }
"""


def _delta(v, b, lower_is_better=True):
    if v != v or b != b:
        return ""
    d = v - b
    cls = "d-good" if (d < 0) == lower_is_better and d != 0 else (
        "d-bad" if d != 0 else "")
    return '<span class="%s">%+.4f</span>' % (cls, d)


def main() -> None:
    scores = json.loads(SCORES.read_text())
    base = scores["baseline"]
    rows = sorted(scores.items(),
                  key=lambda kv: (kv[1].get("near30_log_loss") or 9e9))

    head = ("<tr><th>variant</th><th>near-strike last 30 s</th><th>Δ</th>"
            "<th>pooled log-loss</th><th>Δ</th><th>grid log-loss</th>"
            "<th>Brier</th><th>markets</th><th>build s</th></tr>")
    body = []
    for name, r in rows:
        body.append(
            '<tr class="%s"><td>%s</td><td>%.5f</td><td>%s</td><td>%.5f</td>'
            '<td>%s</td><td>%.5f</td><td>%.5f</td><td>%d</td><td>%.0f</td></tr>'
            % ("base" if name == "baseline" else "", html.escape(name),
               r["near30_log_loss"], _delta(r["near30_log_loss"],
                                            base["near30_log_loss"]),
               r["log_loss"], _delta(r["log_loss"], base["log_loss"]),
               r["log_loss_grid"], r["brier"], r["n_markets"],
               r.get("build_seconds", 0)))

    notes = []
    for name, r in rows:
        if name == "baseline":
            continue
        v = read_variant(name)
        d_near = r["near30_log_loss"] - base["near30_log_loss"]
        d_pool = r["log_loss"] - base["log_loss"]
        finding = FINDINGS.get(name, "")
        notes.append(
            "<h3>%s</h3><p class='note'><em>Expected:</em> %s<br><em>Observed:</em> "
            "near-strike last-30 s log-loss %+.5f, pooled %+.5f against baseline. "
            "Variance level by remaining time: %s.</p>"
            "%s"
            % (html.escape(name), html.escape(v["notes"].strip()), d_near, d_pool,
               html.escape(json.dumps({k: round(x, 3)
                                       for k, x in r["level_by_tte"].items()})),
               ("<p class='note flag'><em>Finding:</em> %s</p>" % html.escape(finding)
                if finding else "")))

    OUT.write_text(
        "<title>Fair-value variants</title><style>%s</style>"
        "<h1>Fair-value variants</h1>"
        "<p class='sub'>Selection window 2026-08-14 to 2026-08-25, real 5 m markets "
        "and real strikes. The holdout 08-26 to 08-31 is not scored here. Sorted by "
        "near-strike last-30 s log-loss; lower is better throughout.</p>"
        "<div class='wrap'><table><thead>%s</thead><tbody>%s</tbody></table></div>"
        "<h2>Per variant</h2>%s"
        % (CSS, head, "".join(body), "".join(notes)), encoding="utf-8")
    print("wrote %s: %d variants" % (OUT, len(rows)))


if __name__ == "__main__":
    main()
