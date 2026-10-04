"""Command-line runner: load an Excel workbook, run the study, export charts + tables.

Examples:
    python run_study.py CRM_Volve3_Dataset.xlsx --forecast 12 --evals 10 --out results_volve
    python run_study.py examples/synthetic_field.xlsx --forecast 60 --evals 20 --out results_syn
"""

from __future__ import annotations

import argparse
import warnings

from crm_ml_hybrid import ML_MODELS
from data_io import load_field_data
from pipeline import export_outputs, run_pipeline


def main() -> None:
    """Parses arguments, runs the study and writes the outputs."""
    ap = argparse.ArgumentParser(description="CRM-ML hybrid production forecasting study")
    ap.add_argument("workbook", help="Excel workbook (see README 'Importing your own data')")
    ap.add_argument("--forecast", type=int, default=12, help="forecast length in samples")
    ap.add_argument("--evals", type=int, default=20, help="evaluations for ELM/MLP (paper: 20)")
    ap.add_argument("--models", nargs="+", default=list(ML_MODELS), choices=list(ML_MODELS))
    ap.add_argument("--no-trim", action="store_true",
                    help="do not skip each producer's leading zero-rate period")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--quiet", action="store_true", help="hide progress output")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    data = load_field_data(args.workbook)
    print(f"Loaded {len(data.time)} samples, {len(data.producer_names)} producers, "
          f"mode = {data.mode}")
    for note in data.notes:
        print("  note:", note)
    out = run_pipeline(data, args.forecast, args.evals, args.models,
                       trim_inactive=not args.no_trim, verbose=not args.quiet)
    paths = export_outputs(out, args.out)
    print(f"\nForecast-period ranking:\n{out.ranking('forecast').round(3)}")
    print(f"\nWrote {len(paths)} files to {args.out}/")


if __name__ == "__main__":
    main()
