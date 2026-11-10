"""Generate all paper figures from the result CSVs."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from su import config as C

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.titlesize": 8,
    "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODE_LABEL = {"stateless": "stateless", "key": "key-dedup", "payload": "payload-dedup",
              "semantic": "delta only", "su_opt": "optimistic", "su": "SemUnlearn"}
MODE_COLOR = {"stateless": "#999999", "key": "#e08214", "payload": "#b2abd2",
              "semantic": "#80cdc1", "su_opt": "#d6604d", "su": "#2166ac"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  figure", name)


def fig_dar():
    p = os.path.join(C.RESULTS, "e7_stats.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    ax = axes[0]
    dd = d[d.k == 2].reset_index(drop=True)
    idx = np.arange(len(dd))
    ax.bar(idx - 0.21, dd.dar_old, width=0.4, color="#d6604d",
           label="all-pairs (uncalibrated)")
    ax.bar(idx + 0.21, dd.dar_loo, width=0.4, yerr=dd.dar_loo_se.fillna(0),
           color="#2166ac", capsize=1.8, label=r"LOO (calibrated)")
    ax.axhline(1.0, ls="--", c="k", lw=0.7)
    ax.axhline(1 - 1.0 / 6, ls=":", c="#d6604d", lw=0.8)
    ax.text(len(dd) - 0.6, 1 - 1.0 / 6 + 0.06, r"$1-1/n$", color="#d6604d", fontsize=6)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{m[:4]}\ns{s}" for m, s in zip(dd.method, dd.seed)], fontsize=6)
    ax.set_ylabel(r"duplicate amplification, $k=2$")
    ax.set_title("estimator calibration")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for method, g in d.groupby("method"):
        m = g.groupby("k").agg(a=("a_paired", "mean"), lo=("a_paired_lo", "mean"),
                               hi=("a_paired_hi", "mean"))
        ax.plot(m.index, m.a, marker="o", ms=3, label=method)
        ax.fill_between(m.index, m.lo, m.hi, alpha=0.18)
    ax.axhline(1.0, ls="--", c="k", lw=0.7)
    ax.text(2.05, 1.05, "single-request noise floor", fontsize=6)
    ax.set_xlabel("replay count $k$")
    ax.set_ylabel(r"paired drift $A_{\mathrm{paired}}$")
    ax.set_title("replay drift beyond the noise floor")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_dar")


def fig_semantics():
    p = os.path.join(C.RESULTS, "e1_semantics.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    scen = list(dict.fromkeys(d.scenario))
    modes = [m for m in MODE_LABEL if m in set(d["mode"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    w = 0.8 / len(modes)
    short = [s.split("_", 1)[1].replace("_", " ") for s in scen]
    for ax, metric, ylab, title in (
            (axes[0], "physical_runs", "unlearning jobs executed", "recomputation"),
            (axes[1], "js_to_retrainA", "JS divergence", "distance to retrain oracle")):
        for j, m in enumerate(modes):
            vals = [d[(d.scenario == s) & (d["mode"] == m)][metric].mean() for s in scen]
            ax.bar(np.arange(len(scen)) + j * w - 0.4, vals, width=w,
                   color=MODE_COLOR[m], label=MODE_LABEL[m])
        ax.set_xticks(np.arange(len(scen)))
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=5.6)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False, ncol=2, fontsize=5.8)
    fig.tight_layout()
    _save(fig, "fig_semantics")


def fig_concurrency():
    p = os.path.join(C.RESULTS, "e3_concurrency.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, scen, metric, ylab, title in (
            (axes[0], "disjoint", "physical_runs", "unlearning jobs", "disjoint effects"),
            (axes[1], "identical", "throughput_rps", "requests / s", "identical effects")):
        for m in [x for x in MODE_LABEL if x in set(d["mode"])]:
            g = d[(d.scenario == scen) & (d["mode"] == m)].groupby("concurrency")[metric].mean()
            ax.plot(g.index, g.values, marker="o", ms=3, color=MODE_COLOR[m], label=MODE_LABEL[m])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("concurrent requests")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    _save(fig, "fig_concurrency")


def fig_db():
    p = os.path.join(C.RESULTS, "e5_dbtask.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    ax = axes[0]
    r = d[(d.experiment == "replay") & (d.region == "in")]
    for cond, g in r.groupby("condition"):
        m = g.groupby("k").qerr_median.mean()
        ax.plot(m.index, m.values, marker="o", ms=3, label=cond)
    ref = d[(d.experiment == "reference") & (d.region == "in")]
    if len(ref):
        ax.axhline(ref.qerr_median.mean(), ls="--", c="k", lw=0.8,
                   label="exact retrain")
    b = d[(d.experiment == "reference") & (d.condition == "base_in_region0")]
    ax.set_yscale("log")
    ax.set_xlabel("replay count $k$")
    ax.set_ylabel("median Q-error (deleted region)")
    ax.set_title("replayed deletion request")
    ax.legend(frameon=False, fontsize=6)
    ax.grid(alpha=0.25)
    ax = axes[1]
    s = d[(d.experiment == "service") & (d.region == "in")]
    if len(s):
        g = s.groupby("condition").agg(q=("qerr_median", "mean"))
        idx = np.arange(len(g))
        ax.bar(idx, g.q, color=[MODE_COLOR.get(x, "#555") for x in g.index], width=0.6)
        ax.set_xticks(idx)
        ax.set_xticklabels([MODE_LABEL.get(x, x) for x in g.index], rotation=15, fontsize=6)
        ax.set_ylabel("median Q-error (deleted region)")
        ax.set_title("after 3 retries + 1 overlapping request")
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, "fig_db")


def fig_batching():
    p = os.path.join(C.RESULTS, "e8_batching_window.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    g = d[d["mode"] == "su"].groupby("window_s").agg(r=("physical_runs", "mean"),
                                                     p=("p50_ms", "mean"))
    ax.plot(g.index * 1000, g.r, marker="o", ms=3, color="#2166ac", label="jobs")
    ax.set_xlabel("batching window (ms)")
    ax.set_ylabel("unlearning jobs")
    ax.set_title("latency/work trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_batching")


def fig_sensitivity():
    p = os.path.join(C.RESULTS, "e6_sensitivity.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    op = d[d.sweep == "operator"].copy().reset_index(drop=True)
    op["label"] = (op["method"].str.replace("finetune_l1", "FT+L1")
                   .str.replace("finetune_noisy", "FT+noise")
                   .str.replace("finetune", "FT")
                   .str.replace("neggradplus", "NG+")
                   .str.replace("salun", "SalUn"))
    op["label"] = op["label"] + " " + op["param"].fillna("").astype(str)
    ax.barh(np.arange(len(op)), op.a_paired, color="#2166ac", height=0.62)
    ax.set_yticks(np.arange(len(op)))
    ax.set_yticklabels(op.label, fontsize=6)
    ax.axvline(1.0, ls="--", c="k", lw=0.7)
    ax.set_xlabel(r"paired drift $A_{\mathrm{paired}}$, $k=2$")
    ax.set_title("operator and hyperparameter")
    ax.grid(alpha=0.25, axis="x")
    ax = axes[1]
    het = d[d.sweep == "heterogeneity"].reset_index(drop=True)
    if len(het):
        ax.bar(np.arange(len(het)), het.a_paired, color="#2166ac", width=0.55)
        ax.axhline(1.0, ls="--", c="k", lw=0.7)
        ax.set_xticks(np.arange(len(het)))
        ax.set_xticklabels([f"set{i}" for i in range(len(het))], fontsize=6)
        ax.set_ylabel(r"$A_{\mathrm{paired}}$")
        ax.set_title("disjoint deletion sets of equal size")
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, "fig_sensitivity")


if __name__ == "__main__":
    for f in (fig_dar, fig_semantics, fig_concurrency, fig_db, fig_batching,
              fig_sensitivity):
        try:
            f()
        except Exception as e:
            print(f"  !! {f.__name__}: {e}")
