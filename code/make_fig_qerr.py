"""Figure: in-region cardinality error under deletion replay (single panel, log y).

Reads results/e5_dbtask.csv and writes figures/fig_qerr.pdf.  The three references are
the base model, the exact-retrain target for the same tenant deletion, and the two
operators replayed k = 1..5 times.
"""
from __future__ import annotations

import csv
import io
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results", "e5_dbtask.csv")
OUT = os.path.join(HERE, "figures", "fig_qerr.pdf")


def load():
    rows = list(csv.DictReader(io.open(CSV, newline="")))
    series = defaultdict(lambda: defaultdict(list))   # condition -> k -> [median]
    retrain = []
    base = []
    for r in rows:
        k = int(r["k"])
        if r["experiment"] == "replay":
            if r["region"] != "in":
                continue
            series[r["condition"]][k].append(float(r["qerr_median"]))
        elif r["condition"].startswith("base_in_region"):
            base.append(float(r["qerr_median"]))
        elif r["condition"].startswith("retrain_tenant") and r["region"] == "in":
            retrain.append(float(r["qerr_median"]))
    return series, base, retrain


def main():
    series, base, retrain = load()
    ok = sorted(series["finetune"])
    ng = sorted(series["neggradplus"])

    def mid(d, k):
        v = d[k]
        return sum(v) / len(v)

    plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.45, 1.80))
    ax.plot([0] + ok, [mid(series["finetune"], 0) if 0 in series["finetune"]
                       else sum(base) / len(base)] + [mid(series["finetune"], k) for k in ok],
            marker="s", ms=2.6, lw=1.0, color="#1f77b4", label="fine-tune")
    ax.plot([0] + ng, [sum(base) / len(base)] + [mid(series["neggradplus"], k) for k in ng],
            marker="o", ms=2.6, lw=1.0, color="#d62728", label="NegGrad+")
    lo, hi = min(retrain), max(retrain)
    ax.axhspan(lo, hi, color="0.75", alpha=0.55, lw=0, label="exact retrain (target)")
    ax.axhline(sum(base) / len(base), color="0.35", ls=":", lw=0.9, label="base model")

    ax.set_yscale("log")
    ax.set_xlabel("times the same tenant deletion is replayed $k$")
    ax.set_ylabel("in-region median Q-error")
    ax.set_xticks([0] + ng)
    ax.set_xticklabels(["0"] + [str(k) for k in ng])
    ax.set_xlim(-0.3, max(ng) + 0.4)
    ax.grid(axis="y", lw=0.3, color="0.85", which="both")
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right", handlelength=1.6,
              borderpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT)
    print("wrote", OUT)
    for k in [0] + ng:
        ft = mid(series["finetune"], k) if k else sum(base) / len(base)
        ngv = mid(series["neggradplus"], k) if k else sum(base) / len(base)
        print(f"  k={k} fine-tune={ft:.3f} negrad+={ngv:.3f}")
    print(f"  retrain band {lo:.3f}-{hi:.3f}, base {sum(base)/len(base):.3f}")


if __name__ == "__main__":
    main()
