import sys

sys.path.insert(0, "code")
import numpy as np

from su import metrics as MET
from su.dbtask_rows import RowDensity, build, estimate_rows, train_row_model

data = build()
print("rows", len(data["country"]), "test queries", len(data["q_te"]))
print("card_te stats: median", np.median(data["card_te"]), "mean", data["card_te"].mean(),
      "max", data["card_te"].max(), "zeros", (data["card_te"] == 0).mean())
m = train_row_model(data["country"], data["a_bin"], data["d_bin"], seed=0, epochs=40)
est = estimate_rows(m, data["q_te"], n_rows=len(data["country"]))
s = MET.q_error_stats(est, data["card_te"])
print({k: round(v, 3) for k, v in s.items()})
print("sample est/card:", list(zip(np.round(est[:5], 1), data["card_te"][:5])))
