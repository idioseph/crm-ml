"""Data import utilities for the CRM-ML hybrid forecaster.

Two input layouts are supported.

Layout A - "full" (paper workflow; CRM is calibrated by this package)
    Excel workbook (or folder of CSVs) with these sheets/files:
        Time          : one column, time stamps
        q_Observed    : K columns, one per producer (observed liquid rate)
        w_Injection   : I columns, one per injector (injection rate)
        Distances     : (optional) I rows x K columns injector-producer distances
        BHP           : (optional) K columns producer bottom-hole pressures

Layout B - "precomputed CRM" (e.g. CRM_Volve3_Dataset.xlsx)
    Excel workbook with sheets:
        Time, q_Observed, q_CRM
    No injection data, so the CRM cannot be re-calibrated; the supplied q_CRM
    series is used as the physics model.

In every sheet the first rows may contain *header* rows: the well name in row 1
followed by optional descriptive rows (e.g. "Oil + Water", "[bbl/mth]").  The
loader keeps the first row as the name and drops any further leading non-numeric
rows automatically.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path, BinaryIO]

SHEET_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "t", "date", "days", "months"],
    "production": ["q_observed", "q_obs", "production", "q_production", "observed", "q"],
    "injection": ["w_injection", "injection", "w_inj", "w", "inj"],
    "distances": ["distances", "distance", "x_ij", "dist"],
    "bhp": ["bhp", "pwf", "p_wf"],
    "crm": ["q_crm", "crm"],
}


@dataclass
class FieldData:
    """Validated field data ready for the forecasting study.

    Attributes:
        time: Time stamps, shape (N,).
        production: Observed producer rates, shape (N, K).
        producer_names: Names of the K producers.
        injection: Injection rates, shape (N, I), or None (layout B).
        injector_names: Names of the I injectors (empty if none).
        distances: Injector-producer distances, shape (I, K), or None.
        bhp: Producer BHPs, shape (N, K), or None.
        q_crm: Pre-computed CRM rates, shape (N, K), or None (layout A).
        time_unit: Unit string read from the file (e.g. "mth"), if any.
        rate_unit: Rate unit string read from the file (e.g. "bbl/mth"), if any.
        notes: Human readable remarks collected while loading.
    """

    time: np.ndarray
    production: np.ndarray
    producer_names: List[str]
    injection: Optional[np.ndarray] = None
    injector_names: List[str] = field(default_factory=list)
    distances: Optional[np.ndarray] = None
    bhp: Optional[np.ndarray] = None
    q_crm: Optional[np.ndarray] = None
    time_unit: str = ""
    rate_unit: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        """``"full"`` if injection data is available, else ``"precomputed"``."""
        return "full" if self.injection is not None else "precomputed"

    def first_active_index(self) -> np.ndarray:
        """Index of the first non-zero observed rate for every producer, shape (K,)."""
        nz = self.production > 0
        return np.where(nz.any(axis=0), nz.argmax(axis=0), 0)

    def summary(self) -> pd.DataFrame:
        """Per-producer summary table (active window, peak rate, zero count)."""
        first = self.first_active_index()
        rows = []
        for j, name in enumerate(self.producer_names):
            nz = np.nonzero(self.production[:, j])[0]
            rows.append({
                "producer": name,
                "first_active_step": int(first[j]),
                "last_active_step": int(nz.max()) if nz.size else -1,
                "active_steps": int(nz.size),
                "peak_rate": float(self.production[:, j].max()),
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sheet reading
# --------------------------------------------------------------------------- #


def _find_sheet(names: List[str], key: str) -> Optional[str]:
    """Finds the sheet whose (case-insensitive) name matches an alias of ``key``."""
    lowered = {n.strip().lower(): n for n in names}
    for alias in SHEET_ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_block(raw: pd.DataFrame) -> tuple[pd.DataFrame, List[str], List[str]]:
    """Splits a raw sheet into numeric data, column names and header notes.

    Row 0 is taken as the column names. Any following rows whose cells are not
    all numeric (units, fluid labels, blanks) are treated as header notes.

    Args:
        raw: Sheet read with ``header=None``.

    Returns:
        ``(numeric_df, names, notes)`` where ``numeric_df`` has float columns.
    """
    raw = raw.dropna(how="all").reset_index(drop=True)
    names = [str(c).strip() for c in raw.iloc[0].tolist()]
    body = raw.iloc[1:].reset_index(drop=True)
    numeric = body.apply(pd.to_numeric, errors="coerce")
    is_data = numeric.notna().all(axis=1)
    if not is_data.any():
        raise ValueError("No numeric rows found below the header.")
    first_data = int(is_data.values.argmax())
    notes = [" | ".join(str(v) for v in body.iloc[r].tolist()) for r in range(first_data)]
    data = numeric.iloc[first_data:].reset_index(drop=True)
    data.columns = names
    return data, names, notes


def _extract_unit(notes: List[str]) -> str:
    """Pulls a bracketed unit such as ``[bbl/mth]`` out of header notes."""
    for note in notes:
        for part in note.split("|"):
            part = part.strip()
            if part.startswith("[") and part.endswith("]"):
                return part[1:-1]
    return ""


def load_field_data(source: PathLike, sheet_map: Optional[Dict[str, str]] = None) -> FieldData:
    """Loads an Excel workbook into :class:`FieldData`.

    Args:
        source: Path or file-like object of an ``.xlsx`` workbook.
        sheet_map: Optional explicit mapping ``{"time": "MySheet", "production": ...,
            "injection": ..., "distances": ..., "bhp": ..., "crm": ...}`` for
            workbooks whose sheet names differ from the defaults.

    Returns:
        A validated :class:`FieldData`.

    Raises:
        ValueError: If mandatory sheets are missing or shapes are inconsistent.
    """
    if hasattr(source, "seek"):
        source.seek(0)
    xls = pd.ExcelFile(source)
    names = xls.sheet_names
    sm = {k: (sheet_map or {}).get(k) or _find_sheet(names, k) for k in SHEET_ALIASES}

    for mandatory in ("time", "production"):
        if sm[mandatory] is None:
            raise ValueError(f"Could not find a '{mandatory}' sheet. Sheets present: {names}. "
                             f"Pass sheet_map={{'{mandatory}': '<sheet name>'}}.")

    notes: List[str] = []
    t_df, _, t_notes = _parse_block(xls.parse(sm["time"], header=None))
    time = t_df.iloc[:, 0].to_numpy(float)
    time_unit = _extract_unit(t_notes)

    q_df, prod_names, q_notes = _parse_block(xls.parse(sm["production"], header=None))
    production = q_df.to_numpy(float)
    rate_unit = _extract_unit(q_notes)

    data = FieldData(time=time, production=production, producer_names=prod_names,
                     time_unit=time_unit, rate_unit=rate_unit, notes=notes)

    if sm["injection"] is not None:
        w_df, inj_names, _ = _parse_block(xls.parse(sm["injection"], header=None))
        data.injection, data.injector_names = w_df.to_numpy(float), inj_names
    if sm["distances"] is not None:
        d_df, _, _ = _parse_block(xls.parse(sm["distances"], header=None))
        data.distances = d_df.to_numpy(float)
    if sm["bhp"] is not None:
        b_df, _, _ = _parse_block(xls.parse(sm["bhp"], header=None))
        data.bhp = b_df.to_numpy(float)
    if sm["crm"] is not None:
        c_df, _, _ = _parse_block(xls.parse(sm["crm"], header=None))
        data.q_crm = c_df.to_numpy(float)

    validate(data)
    if data.injection is None and data.q_crm is not None:
        notes.append("No injection sheet found: running in 'precomputed CRM' mode "
                     "(the supplied q_CRM series is used as the physics model).")
    return data


def validate(data: FieldData) -> None:
    """Checks shapes, ordering and finite values; fills sensible defaults.

    Args:
        data: Data to validate (modified in place: default distances may be added).

    Raises:
        ValueError: On any inconsistency that would break the model.
    """
    n = data.time.shape[0]
    if data.production.shape[0] != n:
        raise ValueError(f"q_Observed has {data.production.shape[0]} rows but Time has {n}.")
    if not np.all(np.diff(data.time) > 0):
        raise ValueError("Time must be strictly increasing.")
    if data.injection is None and data.q_crm is None:
        raise ValueError("Provide either an injection sheet (full mode) or a q_CRM sheet "
                         "(precomputed mode).")
    for label, arr in (("production", data.production), ("injection", data.injection),
                       ("bhp", data.bhp), ("q_crm", data.q_crm)):
        if arr is None:
            continue
        if arr.shape[0] != n:
            raise ValueError(f"'{label}' has {arr.shape[0]} rows, expected {n}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"'{label}' contains NaN/inf - fill or remove them first.")
    k = data.production.shape[1]
    if data.q_crm is not None and data.q_crm.shape[1] != k:
        raise ValueError("q_CRM must have the same number of columns as q_Observed.")
    if data.bhp is not None and data.bhp.shape[1] != k:
        raise ValueError("BHP must have one column per producer.")
    if data.injection is not None:
        i = data.injection.shape[1]
        if not data.injector_names:
            data.injector_names = [f"I-{n_ + 1:02d}" for n_ in range(i)]
        if data.distances is None:
            # No coordinates supplied: use unit distances (feature becomes uninformative).
            data.distances = np.ones((i, k))
            data.notes.append("No distances provided: injector-producer distances set to 1.")
        if data.distances.shape != (i, k):
            raise ValueError(f"Distances must be {i} x {k} (injectors x producers), "
                             f"got {data.distances.shape}.")


def write_template(path: Union[str, Path], n_steps: int = 36, n_injectors: int = 3,
                   n_producers: int = 2) -> Path:
    """Writes an empty Layout-A workbook that users can fill with their own data.

    Args:
        path: Destination ``.xlsx`` file.
        n_steps: Number of example rows.
        n_injectors: Number of injector columns.
        n_producers: Number of producer columns.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(0)
    prod = [f"P-{j + 1:02d}" for j in range(n_producers)]
    inj = [f"I-{i + 1:02d}" for i in range(n_injectors)]

    def sheet(cols: List[str], values: np.ndarray, unit: str) -> pd.DataFrame:
        head = pd.DataFrame([cols, ["Oil + Water"] * len(cols) if unit != "[ft]" else [""] * len(cols),
                             [unit] * len(cols)])
        head.columns = range(len(cols))
        body = pd.DataFrame(values, columns=range(len(cols)))
        return pd.concat([head, body], ignore_index=True)

    time_df = pd.concat([pd.DataFrame([["Time"], ["[mth]"], [""]]),
                         pd.DataFrame(np.arange(1, n_steps + 1))], ignore_index=True)
    out = Path(path)
    with pd.ExcelWriter(out) as xw:
        time_df.to_excel(xw, sheet_name="Time", header=False, index=False)
        sheet(prod, rng.uniform(1e3, 2e3, (n_steps, n_producers)), "[bbl/mth]").to_excel(
            xw, sheet_name="q_Observed", header=False, index=False)
        sheet(inj, rng.uniform(1e3, 2e3, (n_steps, n_injectors)), "[bbl/mth]").to_excel(
            xw, sheet_name="w_Injection", header=False, index=False)
        sheet(prod, rng.uniform(500, 3000, (n_injectors, n_producers)), "[ft]").to_excel(
            xw, sheet_name="Distances", header=False, index=False)
    return out


def load_from_bytes(content: bytes, **kwargs) -> FieldData:
    """Convenience wrapper for GUI uploads (bytes -> :class:`FieldData`)."""
    return load_field_data(io.BytesIO(content), **kwargs)


def from_arrays(
    time: Sequence[float],
    production: Union[np.ndarray, pd.DataFrame],
    injection: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    distances: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    bhp: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    q_crm: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    producer_names: Optional[Sequence[str]] = None,
    injector_names: Optional[Sequence[str]] = None,
    time_unit: str = "",
    rate_unit: str = "",
) -> FieldData:
    """Builds :class:`FieldData` straight from NumPy arrays / pandas DataFrames (e.g. CSVs).

    DataFrame column names are used as well names when explicit names are not given.

    Args:
        time: Time stamps, length N.
        production: Observed rates, (N, K).
        injection: Injection rates, (N, I), or None.
        distances: Injector-producer distances, (I, K), or None.
        bhp: Producer BHPs, (N, K), or None.
        q_crm: Pre-computed CRM rates, (N, K), or None.
        producer_names: Optional producer labels.
        injector_names: Optional injector labels.
        time_unit: Time unit label for plots.
        rate_unit: Rate unit label for plots.

    Returns:
        A validated :class:`FieldData`.
    """
    def arr(x):
        return None if x is None else np.asarray(x, dtype=float)

    if producer_names is None and isinstance(production, pd.DataFrame):
        producer_names = [str(c) for c in production.columns]
    if injector_names is None and isinstance(injection, pd.DataFrame):
        injector_names = [str(c) for c in injection.columns]
    prod = arr(production)
    prod = prod.reshape(-1, 1) if prod.ndim == 1 else prod
    data = FieldData(
        time=np.asarray(time, dtype=float), production=prod,
        producer_names=list(producer_names) if producer_names is not None
        else [f"P-{j + 1:02d}" for j in range(prod.shape[1])],
        injection=arr(injection), injector_names=list(injector_names or []),
        distances=arr(distances), bhp=arr(bhp), q_crm=arr(q_crm),
        time_unit=time_unit, rate_unit=rate_unit)
    validate(data)
    return data
