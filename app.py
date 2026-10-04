"""Streamlit GUI for the CRM-ML hybrid production forecaster.

Launch with:
    uv run streamlit run app.py
    (or)  streamlit run app.py

Workflow in the GUI: upload workbook -> check the data preview -> choose forecast
length / models / evaluations -> Run -> browse the paper-style charts and tables ->
download everything as a ZIP.
"""

from __future__ import annotations

import io
import tempfile
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from crm_ml_hybrid import ML_MODELS
from data_io import FieldData, load_from_bytes, write_template
from pipeline import StudyOutput, export_outputs, make_figures, run_pipeline

warnings.filterwarnings("ignore")
st.set_page_config(page_title="CRM-ML Production Forecasting", layout="wide")

FIG_TITLES = {
    "crm_fit": "CRM fit and forecast (≈ paper Fig. 9 / 18)",
    "crm_parameters": "CRM parameters λij, τij (≈ Fig. 8 / 17)",
    "approaches_forecast": "All approaches - forecast period (≈ Fig. 11 / 14)",
    "approaches_entire": "All approaches - entire record (≈ Fig. 10 / 19)",
    "mae_by_evaluation_forecast": "MAE per evaluation - forecast (≈ Fig. 12 / 15 / 21)",
    "mae_by_evaluation_entire": "MAE per evaluation - entire record (≈ Fig. 20)",
    "average_mae_forecast": "Average MAE - forecast (≈ Fig. 13b / 16 / 22b)",
    "average_mae_entire": "Average MAE - entire record (≈ Fig. 13a / 22a)",
    "ranking": "Ranking ladder (≈ Fig. 23)",
}


@st.cache_data(show_spinner=False)
def _load(content: bytes) -> FieldData:
    """Cached workbook loader."""
    return load_from_bytes(content)


def _template_bytes() -> bytes:
    """Builds the blank Layout-A template workbook in memory."""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_template(Path(tmp) / "template.xlsx")
        return path.read_bytes()


def _zip_outputs(out: StudyOutput) -> bytes:
    """Exports all charts/tables to a temp folder and returns them as a ZIP."""
    with tempfile.TemporaryDirectory() as tmp:
        files = export_outputs(out, tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)
        return buf.getvalue()


# ------------------------------- sidebar ----------------------------------- #
st.title("CRM-ML hybrid production forecasting")
st.caption("Capacitance-Resistance Model + machine learning (Ogali & Orodu, 2025)")

with st.sidebar:
    st.header("1. Data")
    upload = st.file_uploader("Excel workbook (.xlsx)", type=["xlsx"])
    st.download_button("Download blank template", _template_bytes(), "crm_ml_template.xlsx",
                       help="Layout A: Time, q_Observed, w_Injection, Distances")

    st.header("2. Settings")
    n_forecast = st.number_input("Forecast length (samples)", 1, 1000, 12)
    n_evals = st.slider("Evaluations (ELM / MLP)", 1, 20, 5,
                        help="The paper uses 20. Fewer is faster.")
    models = st.multiselect("ML models to combine with CRM", list(ML_MODELS),
                            default=list(ML_MODELS))
    trim = st.checkbox("Skip each producer's leading zero-rate period", value=True)
    n_starts = st.slider("CRM optimiser restarts", 1, 8, 3,
                         help="Full mode only. More restarts = better optimum, slower.")
    run = st.button("Run study", type="primary", use_container_width=True)

# -------------------------------- main ------------------------------------- #
if upload is None:
    st.info("Upload a workbook in the sidebar to begin. See the README section "
            "'Importing your own data' for the expected layout.")
    st.stop()

try:
    data = _load(upload.getvalue())
except Exception as exc:  # show a friendly message instead of a traceback
    st.error(f"Could not read the workbook: {exc}")
    st.stop()

mode_text = ("**Full mode** - the CRM will be calibrated from injection and production data."
             if data.mode == "full" else
             "**Pre-computed CRM mode** - no injection data found, so the supplied `q_CRM` "
             "series is used as the CRM and as a feature for the hybrids.")
st.markdown(mode_text)
for note in data.notes:
    st.caption(f"ℹ️ {note}")

tab_data, tab_results = st.tabs(["Data preview", "Results"])
with tab_data:
    st.dataframe(data.summary(), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for j, name in enumerate(data.producer_names):
        ax.plot(data.time, data.production[:, j], label=name, lw=1.2)
    ax.set_xlabel(f"Time [{data.time_unit}]" if data.time_unit else "Time")
    ax.set_ylabel(f"q [{data.rate_unit}]" if data.rate_unit else "q")
    ax.legend(fontsize=7, ncol=3)
    st.pyplot(fig)
    plt.close(fig)
    if n_forecast >= len(data.time) - 10:
        st.warning("Forecast length leaves fewer than 10 historical samples.")

if run:
    if not models:
        st.error("Select at least one ML model.")
        st.stop()
    with st.spinner("Running... (CRM calibration and model training)"):
        try:
            st.session_state["out"] = run_pipeline(
                data, int(n_forecast), int(n_evals), models, trim_inactive=trim,
                crm_kwargs=dict(n_starts=int(n_starts)))
        except Exception as exc:
            st.error(f"Study failed: {exc}")
            st.stop()

with tab_results:
    out: StudyOutput | None = st.session_state.get("out")
    if out is None:
        st.write("Press **Run study** in the sidebar.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Ranking - forecast period")
            st.dataframe(out.ranking("forecast").round(3), use_container_width=True)
        with c2:
            st.subheader("Ranking - entire record")
            st.dataframe(out.ranking("entire").round(3), use_container_width=True)
        st.download_button("Download all charts + tables (ZIP)", _zip_outputs(out),
                           "crm_ml_results.zip", mime="application/zip")
        figs = make_figures(out)
        tabs = st.tabs([FIG_TITLES.get(k, k) for k in figs])
        for tab, (name, fig) in zip(tabs, figs.items()):
            with tab:
                st.pyplot(fig)
                plt.close(fig)
