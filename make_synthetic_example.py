"""Writes examples/synthetic_field.xlsx (Layout A: injectors + producers + distances)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crm_ml_hybrid import make_mock_field  # noqa: E402


def block(names, values, unit):
    """Header row (names), a unit row, then numeric rows - the layout the loader expects."""
    head = pd.DataFrame([names, [unit] * len(names)])
    head.columns = range(len(names))
    body = pd.DataFrame(values)
    body.columns = range(len(names))
    return pd.concat([head, body], ignore_index=True)


time, inj, prod, dist, _ = make_mock_field(n_steps=400)
out = Path(__file__).with_name("synthetic_field.xlsx")
prods = [f"P-{j + 1:02d}" for j in range(prod.shape[1])]
injs = [f"I-{i + 1:02d}" for i in range(inj.shape[1])]
with pd.ExcelWriter(out) as xw:
    pd.concat([pd.DataFrame([["Time"], ["[day]"]]), pd.DataFrame(time)], ignore_index=True
              ).to_excel(xw, sheet_name="Time", header=False, index=False)
    block(prods, prod, "[STB/day]").to_excel(xw, sheet_name="q_Observed", header=False, index=False)
    block(injs, inj, "[STB/day]").to_excel(xw, sheet_name="w_Injection", header=False, index=False)
    block(prods, dist, "[ft]").to_excel(xw, sheet_name="Distances", header=False, index=False)
print("wrote", out)
