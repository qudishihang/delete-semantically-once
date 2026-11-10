"""Regenerate Table I (calibrated replay statistics) from results/e7_stats.csv."""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "results", "e7_stats.csv"))

rows = []
for _, r in d.iterrows():
    op = "NegGrad+" if r["method"] == "neggradplus" else "fine-tune"
    acc = f"{r['test_acc_k1']*100:0.2f}$\\to${r['test_acc_k']*100:0.2f}"
    rows.append(
        f"{int(r['seed'])} & {op} & {int(r['k'])} & {r['v_stoch_loo']*1000:.2f} & "
        f"{r['dar_loo']:.2f}$\\pm${r['dar_loo_se']:.2f} & {r['a_paired']:.2f} & "
        f"{r['p_paired_exceeds_stoch']:.3f} & {acc} \\\\")
body = "\n".join(rows)

tex = os.path.join(HERE, "..", "paper", "main.tex")
s = io.open(tex, encoding="utf-8").read()
start = s.find("\\midrule\n0 & NegGrad+ & 2")
end = s.find("\\bottomrule", start)
assert start != -1 and end != -1
s = s[:start] + "\\midrule\n" + body + "\n" + s[end:]
io.open(tex, "w", encoding="utf-8", newline="\n").write(s)
print("wrote", len(rows), "rows")
