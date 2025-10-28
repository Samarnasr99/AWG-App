# awg_app.py — Streamlit app for AWG predictions (ML-direct + KNN)
# - Airflow for KPIs is fixed internally at 90,000 m³/h (no UI text)
# - Area removed from UI & outputs
# - Wind speed input remains for ML/KNN
# - Default models & dataset auto-downloaded from GitHub Releases
# - Robust header aliasing for KNN inputs
# - Horizontal bar charts with optional ±% error bars (ML vs KNN)
# - Safe dataframe column selection to avoid KeyErrors

import io
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional imports
try:
    import shap
except Exception:
    shap = None

# XGBoost fallback (for sklearn-wrapper quirks)
try:
    import xgboost as xgb
except Exception:
    xgb = None

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
import joblib
import requests

st.set_page_config(page_title="AWG Dual Calculator (KNN + ML)", layout="wide")

# ---------------------- Constants / Mappings ----------------------

# fixed airflow for KPI metrics (not shown in UI)
_AIRFLOW_M3PH = 90_000.0

EXPECTED_INPUTS = [
    'Wind_in_temperature (°C)',
    'Wind_in_rh (%)',
    'Wind_in_speed (m/s)',
]

TARGETS = [
    'Water production (L/h)',
    'Power (kW)',
    'DO (mg/L)',
    'ph',
    'Conductivity (µS/cm)',
    'Turbidity (NTU)',
    'SEC (kWh/L)',
    'COP',
    'Efficiency (%)'
]

RAW_TARGET_TO_STD = {
    'Water production (L)': 'Water production (L/h)',
    'real_time_power (kW)': 'Power (kW)',
    'DO': 'DO (mg/L)',
    'ph': 'ph',
    'Conductivity': 'Conductivity (µS/cm)',
    'water_turb': 'Turbidity (NTU)',
}
STD_TO_RAW_TARGET = {v: k for k, v in RAW_TARGET_TO_STD.items()}

DERIVED_METRICS = ['SEC (kWh/L)', 'COP', 'Airborne water (kg/h)', 'Capture efficiency (%)']

# ---- GitHub Release assets ----
DEFAULT_MODEL_URLS = {
    "scaler_final_model_set": os.environ.get("SCALER_URL", st.secrets.get("SCALER_URL", "https://github.com/Samarnasr99/AWG-App/releases/download/Models/scaler_final_model_set.pkl")),
    "Water production (L)":   os.environ.get("WATER_URL",  st.secrets.get("WATER_URL",  "https://github.com/Samarnasr99/AWG-App/releases/download/Models/Water.production.L._final_model.pkl")),
    "real_time_power (kW)":   os.environ.get("POWER_URL",  st.secrets.get("POWER_URL",  "https://github.com/Samarnasr99/AWG-App/releases/download/Models/real_time_power.kW._final_model.pkl")),
    "DO":                     os.environ.get("DO_URL",     st.secrets.get("DO_URL",     "https://github.com/Samarnasr99/AWG-App/releases/download/Models/DO_final_model.pkl")),
    "ph":                     os.environ.get("PH_URL",     st.secrets.get("PH_URL",     "https://github.com/Samarnasr99/AWG-App/releases/download/Models/ph_final_model.pkl")),
    "Conductivity":           os.environ.get("COND_URL",   st.secrets.get("COND_URL",   "https://github.com/Samarnasr99/AWG-App/releases/download/Models/Conductivity_final_model.pkl")),
    "water_turb":             os.environ.get("TURB_URL",   st.secrets.get("TURB_URL",   "https://github.com/Samarnasr99/AWG-App/releases/download/Models/water_turb_final_model.pkl")),
}

DEFAULT_DATA_URL = os.environ.get(
    "DATA_URL",
    st.secrets.get("DATA_URL", "https://github.com/Samarnasr99/AWG-App/releases/download/Models/Search_and_fit._AWG.xlsx")
)

# ---------------------- Caching helpers ----------------------

try:
    from streamlit.runtime.caching import cache_resource as st_cache_resource
except Exception:
    st_cache_resource = st.cache_resource

try:
    from streamlit.runtime.caching import cache_data as st_cache_data
except Exception:
    st_cache_data = st.cache_data

@st_cache_resource(show_spinner=False)
def _download_joblib(url: str):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return joblib.load(BytesIO(r.content))

@st_cache_data(show_spinner=False)
def _download_table(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    buf = BytesIO(resp.content)
    if url.lower().endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf, sheet_name=0)

def _autofill_models_if_none(
    models_dict: Dict[str, BaseEstimator],
    scaler_obj: Optional[StandardScaler]
) -> Tuple[Dict[str, BaseEstimator], Optional[StandardScaler]]:
    """If user didn't upload, pull artifacts from GitHub Release."""
    have_any = (models_dict and len(models_dict) > 0) or (scaler_obj is not None)
    if have_any:
        return models_dict, scaler_obj

    fetched_models: Dict[str, BaseEstimator] = {}
    fetched_scaler: Optional[StandardScaler] = None

    for key_raw, url in DEFAULT_MODEL_URLS.items():
        if not url:
            continue
        try:
            obj = _download_joblib(url)
            if key_raw.startswith("scaler") or (hasattr(obj, "mean_") and hasattr(obj, "scale_")):
                fetched_scaler = obj
            else:
                fetched_models[key_raw] = obj
        except Exception as e:
            st.warning(f"Could not fetch '{key_raw}' from release: {e}")

    return (fetched_models if fetched_models else models_dict,
            fetched_scaler if fetched_scaler is not None else scaler_obj)

# ---------------------- Utilities ----------------------

def calc_metrics(temp_c: float, rh_pct: float, speed_ms: float,
                 water_lph: float, power_kw: float):
    """Derived KPIs with airflow fixed at 90,000 m³/h."""
    h_fg_kj_per_kg = 2260.0
    p_sat_kpa = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    p_vapor_kpa = (rh_pct / 100.0) * p_sat_kpa
    abs_humidity_g_per_m3 = (216.7 * p_vapor_kpa) / (temp_c + 273.15)
    abs_humidity_kg_per_m3 = abs_humidity_g_per_m3 / 1000.0

    air_volume_m3ph = _AIRFLOW_M3PH  # fixed
    total_water_air_kgph = air_volume_m3ph * abs_humidity_kg_per_m3

    sec = (power_kw) / (water_lph) if water_lph > 0 else float("inf")
    cop = (water_lph * h_fg_kj_per_kg) / (power_kw * 3600.0) if power_kw > 0 else 0.0
    efficiency = (water_lph / (total_water_air_kgph * 1.0)) * 100.0 if total_water_air_kgph > 0 else 0.0
    return sec, cop, total_water_air_kgph, efficiency

def nice_float(x, nd=3):
    try:
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(f"{x:.{nd}f}")
    except Exception:
        return x

def _read_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(uploaded_file, sheet_name=0)
    return pd.read_csv(uploaded_file)

# ----- column aliasing to avoid KNN KeyError -----
COL_ALIASES = {
    'Wind_in_temperature (°C)': {
        'wind_in_temperature (°c)', 'wind_in_temperature', 'wind_in_temp',
        'windintemperature', 'temperature (°c)', 'temperature', 'temp', 't', 'temp_c'
    },
    'Wind_in_rh (%)': {
        'wind_in_rh (%)', 'wind_in_rh', 'windinrh', 'wind_rh',
        'rh (%)', 'relative humidity (%)', 'relative humidity', 'rh', 'humidity'
    },
    'Wind_in_speed (m/s)': {
        'wind_in_speed (m/s)', 'wind_in_speed', 'windspeed', 'wind_speed',
        'wind speed (m/s)', 'wind speed', 'speed (m/s)', 'speed', 'v', 'u'
    },
}

def _normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants of the three input columns to expected names."""
    if df is None or df.empty:
        return df

    def norm(s: str) -> str:
        return s.lower().strip().replace(' ', '').replace('_', '')

    norm_map = {norm(c): c for c in df.columns}
    renames: Dict[str, str] = {}

    for target, aliases in COL_ALIASES.items():
        if target in df.columns:
            continue
        found_col = None

        for alias in aliases:
            if alias in norm_map:
                found_col = norm_map[alias]; break
            alias_norm = alias.lower().strip()
            for c in df.columns:
                if c.lower().strip() == alias_norm:
                    found_col = c; break
            if found_col: break
            alias_key = norm(alias)
            if alias_key in norm_map:
                found_col = norm_map[alias_key]; break

        if not found_col:
            wanted = norm(target)
            for c in df.columns:
                if norm(c) == wanted:
                    found_col = c; break
        if not found_col:
            for c in df.columns:
                cn = norm(c)
                if 'temperature' in target.lower() and ('temp' in cn or 'temperature' in cn):
                    found_col = c; break
                if 'rh' in target.lower() and ('rh' in cn or 'humidity' in cn):
                    found_col = c; break
                if 'speed' in target.lower() and ('speed' in cn):
                    found_col = c; break

        if found_col:
            renames[found_col] = target

    if renames:
        df = df.rename(columns=renames)
    return df

def _pack_predictions_row(temp, rh, speed, outputs_dict):
    water = outputs_dict.get('Water production (L/h)', 0.0) or 0.0
    power = outputs_dict.get('Power (kW)', 0.0) or 0.0
    sec, cop, airborne, eff = calc_metrics(temp, rh, speed, water, power)
    row = {
        'Wind_in_temperature (°C)': temp,
        'Wind_in_rh (%)': rh,
        'Wind_in_speed (m/s)': speed,
    }
    row.update(outputs_dict)
    row.update({
        'SEC (kWh/L)': sec,
        'COP': cop,
        'Airborne water (kg/h)': airborne,
        'Capture efficiency (%)': eff
    })
    return row

def _predict_direct(models: Dict[str, BaseEstimator],
                    scaler: Optional[StandardScaler],
                    features: np.ndarray,
                    feature_names: List[str]) -> Dict[str, float]:
    X = features.reshape(1, -1)
    if scaler is not None:
        try:
            check_is_fitted(scaler)
            X = scaler.transform(X)
        except Exception:
            pass

    preds = {}
    for out_name_std, raw_name in STD_TO_RAW_TARGET.items():
        est = models.get(raw_name)
        if est is None:
            continue
        y_val = None
        try:
            y_val = est.predict(X)[0]
        except Exception:
            try:
                if hasattr(est, "get_booster") and xgb is not None:
                    booster = est.get_booster()
                    try:
                        y_pred = booster.inplace_predict(X)
                    except Exception:
                        y_pred = booster.predict(xgb.DMatrix(X))
                    y_val = float(y_pred[0])
            except Exception:
                raise
        if y_val is not None:
            preds[out_name_std] = float(y_val)
    return preds

def _knn_match(df: pd.DataFrame,
               temp: float, rh: float, speed: float,
               tol_temp: float, tol_rh: float, tol_speed: float,
               k_max: int = 5):
    filt = (
        (df['Wind_in_temperature (°C)'].between(temp - tol_temp, temp + tol_temp)) &
        (df['Wind_in_rh (%)'].between(rh - tol_rh, rh + tol_rh)) &
        (df['Wind_in_speed (m/s)'].between(speed - tol_speed, speed + tol_speed))
    )
    dff = df.loc[filt].copy()
    if dff.empty:
        base = df[EXPECTED_INPUTS].to_numpy()
        scaler = StandardScaler().fit(base)
        tree = KDTree(scaler.transform(base))
        d = scaler.transform([[temp, rh, speed]])
        _, ind = tree.query(d, k=min(k_max, base.shape[0]))
        sel = df.iloc[ind[0].tolist()]
    else:
        sel = dff

    outs, stds = {}, {}
    for std_name in TARGETS:
        raw_col = None
        for k, v in RAW_TARGET_TO_STD.items():
            if v == std_name:
                raw_col = k
                break
        col_to_use = raw_col if (raw_col and raw_col in df.columns) else std_name
        if col_to_use not in sel.columns:
            continue
        vals = sel[col_to_use].astype(float).dropna()
        if len(vals) == 0:
            continue
        outs[std_name] = float(vals.mean())
        stds[std_name] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return outs, stds, len(sel)

def _build_feature_vector(temp, rh, speed):
    temp_rh_interaction = temp * rh
    speed_squared = speed ** 2
    temp_rh_ratio = temp / (rh + 1e-6)
    log_rh = np.log1p(rh)
    log_speed = np.log1p(speed)
    turb_rolling = 0.0
    feats = np.array(
        [temp, log_rh, log_speed, temp_rh_interaction, speed_squared, temp_rh_ratio, turb_rolling],
        dtype=float
    )
    names = ['temp', 'log_rh', 'log_speed', 'temp_x_rh', 'speed_sq', 'temp/rh', 'turb_roll']
    return feats, names

def _normalize_stem_to_raw(stem: str) -> Optional[str]:
    s = stem.replace('.', ' ').replace('__', '_').replace('_final_model', '').strip()
    candidates = [
        ('Water production (L)', ['Water production (L)']),
        ('real_time_power (kW)', ['real time power (kW)', 'real_time_power (kW)', 'real time power kW']),
        ('DO', ['DO']),
        ('ph', ['ph', 'pH', 'PH']),
        ('Conductivity', ['Conductivity']),
        ('water_turb', ['water turb', 'water_turb', 'turbidity']),
    ]
    s_low = s.lower()
    for raw_key, aliases in candidates:
        for a in aliases:
            if s_low == a.lower():
                return raw_key
    if 'water' in s_low and 'production' in s_low and '(l)' in s_low:
        return 'Water production (L)'
    if 'power' in s_low and 'kw' in s_low:
        return 'real_time_power (kW)'
    if s_low in {'do'}: return 'DO'
    if s_low in {'ph', 'p h'}: return 'ph'
    if 'conduct' in s_low: return 'Conductivity'
    if 'turb' in s_low: return 'water_turb'
    return None

def _load_model_artifacts(uploaded_files: List):
    models: Dict[str, BaseEstimator] = {}
    scaler: Optional[StandardScaler] = None
    if not uploaded_files:
        return models, scaler
    for f in uploaded_files:
        if not f.name.endswith('.pkl'):
            continue
        obj = joblib.load(f)
        if isinstance(obj, StandardScaler) or (hasattr(obj, "mean_") and hasattr(obj, "scale_")):
            scaler = obj
            continue
        stem = Path(f.name).stem
        raw_key = _normalize_stem_to_raw(stem) or stem
        models[raw_key] = obj
    return models, scaler

# ---------------------- Plot helpers ----------------------
_ORDER_FOR_PLOT = [
    'Water production (L/h)', 'Power (kW)', 'DO (mg/L)', 'ph',
    'Conductivity (µS/cm)', 'Turbidity (NTU)', 'SEC (kWh/L)', 'COP', 'Efficiency (%)'
]

def _build_plot_payload(row: dict) -> Dict[str, float]:
    return {k: row.get(k) for k in _ORDER_FOR_PLOT if k in row}

def _bar_chart_with_optional_error(values_ml: Dict[str, float],
                                   values_knn: Optional[Dict[str, float]] = None):
    """Draw horizontal bars for ML values; if KNN provided, add ±% error bars
    where pct = |ML - KNN| / ML * 100, and annotate 'value ±pct%'."""
    labels, nums, xerr, ann = [], [], None, []
    for k in _ORDER_FOR_PLOT:
        if k in values_ml and values_ml[k] is not None:
            try:
                v = float(values_ml[k])
            except Exception:
                continue
            labels.append(k); nums.append(v)
            if values_knn and k in values_knn and values_knn[k] is not None:
                try:
                    kn = float(values_knn[k])
                    pct = 0.0 if v == 0 else abs(v - kn) / abs(v) * 100.0
                    err_abs = abs(v) * (pct / 100.0)
                except Exception:
                    pct, err_abs = 0.0, 0.0
            else:
                pct, err_abs = 0.0, 0.0
            ann.append(pct)
            # collect xerr per-bar
            if xerr is None: xerr = []
            xerr.append(err_abs)

    if not nums:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(labels))[::-1]
    if xerr is not None:
        ax.barh(y, nums, xerr=xerr, capsize=4, edgecolor="black")
    else:
        ax.barh(y, nums, edgecolor="black")
    ax.set_yticks(y, labels)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    # annotate "value  ±p%"
    for yi, v, p in zip(y, nums, ann):
        tail = f"  {v:.2f}" + (f"  ±{p:.1f}%" if p > 0 else "")
        ax.text(v, yi, tail, va="center", ha="left", fontsize=9)

    plt.tight_layout()
    return fig

# ---------------------- Sidebar ----------------------

st.sidebar.header("Configuration")

with st.sidebar.expander("Upload ML-direct model artifacts (.pkl)", expanded=False):
    up_models = st.file_uploader(
        "Upload scaler and regressors (*.pkl): scaler_final_model_set.pkl, Water production (L)_final_model.pkl, real_time_power (kW)_final_model.pkl, DO_final_model.pkl, ph_final_model.pkl, Conductivity_final_model.pkl, water_turb_final_model.pkl",
        type=["pkl"], accept_multiple_files=True
    )
    models_dict, scaler_obj = _load_model_artifacts(up_models)
    st.caption(f"Loaded {len(models_dict)} models; Scaler: {'Yes' if scaler_obj is not None else 'No'}")

with st.sidebar.expander("Upload historical dataset for KNN (.csv/.xlsx)", expanded=False):
    hist_file = st.file_uploader("Upload historical AWG dataset", type=["csv", "xlsx", "xlsm", "xls"], accept_multiple_files=False)
    hist_df = None
    if hist_file is not None:
        try:
            hist_df = _read_dataframe(hist_file)
            hist_df = _normalize_input_columns(hist_df)
            st.caption(f"Dataset shape: {hist_df.shape}. Columns: {list(hist_df.columns)[:8]} ...")
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")

# ---- Auto-fill default dataset if none uploaded ----
if hist_df is None and DEFAULT_DATA_URL:
    try:
        hist_df = _download_table(DEFAULT_DATA_URL)
        hist_df = _normalize_input_columns(hist_df)
        st.caption(f"Loaded default dataset from release: {hist_df.shape}")
    except Exception as e:
        st.warning(f"No uploaded dataset and default download failed: {e}")

with st.sidebar.expander("KNN settings", expanded=False):
    tol_temp = st.number_input("Temperature window (±°C)", 0.1, 20.0, 1.0, 0.1)
    tol_rh = st.number_input("RH window (±%)", 0.5, 40.0, 2.0, 0.5)
    tol_speed = st.number_input("Speed window (±m/s)", 0.05, 5.0, 0.5, 0.05)
    k_max = st.slider("Max neighbors (fallback)", 1, 15, 5, 1)

# ---- Auto-fill models from Release if nothing uploaded ----
models_dict, scaler_obj = _autofill_models_if_none(models_dict, scaler_obj)

st.title("AWG Dual Calculator: KNN matcher + ML direct predictor")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single-point", "Compare (KNN vs ML)", "Batch", "Explain/AD", "Downloads"])

# --- Single-point
with tab1:
    st.subheader("Single-point prediction")
    c1, c2, c3, c4 = st.columns(4)
    temp = c1.number_input("Wind_in_temperature (°C)", -10.0, 60.0, 35.0, 0.1)
    rh = c2.number_input("Wind_in_rh (%)", 1.0, 100.0, 70.0, 0.1)
    speed = c3.number_input("Wind_in_speed (m/s)", 0.0, 30.0, 0.5, 0.1)
    do_calc = c4.button("Predict", use_container_width=True)

    if do_calc:
        rows = []
        feats, feat_names = _build_feature_vector(temp, rh, speed)
        ml_row = None
        knn_row = None

        if models_dict:
            ml_out = _predict_direct(models_dict, scaler_obj, feats, feat_names)
            ml_row = _pack_predictions_row(temp, rh, speed, ml_out) | {'Model': 'ML-direct'}
            rows.append(ml_row)

        if hist_df is not None:
            try:
                outs, stds, nsel = _knn_match(hist_df, temp, rh, speed, tol_temp, tol_rh, tol_speed, k_max)
                knn_row = _pack_predictions_row(temp, rh, speed, outs) | {'Model': f'KNN (n={nsel})'}
                for k, v in stds.items():
                    if k in outs and outs[k] != 0:
                        knn_row[f"{k} STD%"] = 100.0 * v / abs(outs[k])
                rows.append(knn_row)
            except Exception as e:
                st.error(f"KNN error: {e}")

        if not rows:
            st.warning("Please upload either model artifacts (for ML-direct) or ensure the default models/dataset loaded.")
        else:
            order_cols = ['Model'] + EXPECTED_INPUTS + TARGETS + DERIVED_METRICS
            extra = [c for c in rows[0].keys() if c.endswith('STD%')]
            df_show = pd.DataFrame(rows)[order_cols + extra]
            df_show = df_show.applymap(lambda z: nice_float(z, 3) if isinstance(z, (float, np.floating)) else z)
            st.dataframe(df_show, use_container_width=True)
            st.session_state['last_single_df'] = df_show

            # ---- Plot: ML with ±% error vs KNN if available
            vals_ml = _build_plot_payload(ml_row or rows[0])
            vals_knn = _build_plot_payload(knn_row) if knn_row else None
            fig = _bar_chart_with_optional_error(vals_ml, vals_knn)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)

# --- Compare
with tab2:
    st.subheader("Side-by-side comparison")
    c1, c2, c3, c4 = st.columns(4)
    temp_c = c1.number_input("Temperature (°C)", -10.0, 60.0, 35.0, 0.1, key="cmp_t")
    rh_c   = c2.number_input("RH (%)", 1.0, 100.0, 70.0, 0.1, key="cmp_rh")
    spd_c  = c3.number_input("Speed (m/s)", 0.0, 30.0, 0.5, 0.1, key="cmp_spd")
    do_cmp = c4.button("Compare", use_container_width=True, key="cmp_btn")

    if do_cmp:
        rows = []
        ml_row = None
        if models_dict:
            feats, feat_names = _build_feature_vector(temp_c, rh_c, spd_c)
            ml_out = _predict_direct(models_dict, scaler_obj, feats, feat_names)
            ml_row = _pack_predictions_row(temp_c, rh_c, spd_c, ml_out) | {'Model': 'ML-direct'}
            rows.append(ml_row)
        knn_row = None
        if hist_df is not None:
            outs, stds, nsel = _knn_match(hist_df, temp_c, rh_c, spd_c, tol_temp, tol_rh, tol_speed, k_max)
            knn_row = _pack_predictions_row(temp_c, rh_c, spd_c, outs) | {'Model': f'KNN (n={nsel})'}
            rows.append(knn_row)

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df[['Model'] + TARGETS + DERIVED_METRICS], use_container_width=True)
            if len(rows) == 2:
                deltas = {
                    k: (rows[0].get(k, np.nan) if rows[0].get(k) is not None else np.nan)
                       - (rows[1].get(k, np.nan) if rows[1].get(k) is not None else np.nan)
                    for k in TARGETS + DERIVED_METRICS
                }
                st.caption("ML-direct minus KNN (positive means ML higher):")
                st.dataframe(pd.DataFrame([deltas]), use_container_width=True)
            st.session_state['last_compare_df'] = df

            # ---- Plot: ML with ±% error vs KNN
            vals_ml = _build_plot_payload(ml_row or rows[0])
            vals_knn = _build_plot_payload(knn_row) if knn_row else None
            fig = _bar_chart_with_optional_error(vals_ml, vals_knn)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Provide both a dataset and model artifacts to compare.")

# --- Batch
with tab3:
    st.subheader("Batch predictions")
    st.write("Upload a CSV/XLSX with columns: 'Wind_in_temperature (°C)', 'Wind_in_rh (%)', 'Wind_in_speed (m/s)'.")
    batch_file = st.file_uploader("Batch file", type=["csv", "xlsx", "xls"], key="batch")
    mode = st.radio("Compute with", ["ML-direct", "KNN", "Both"], horizontal=True)
    run_batch = st.button("Run batch", use_container_width=True)

    if run_batch and batch_file is not None:
        try:
            df_in = _read_dataframe(batch_file)
            df_in = _normalize_input_columns(df_in)
            out_rows = []
            for _, r in df_in.iterrows():
                t, h, s = float(r[EXPECTED_INPUTS[0]]), float(r[EXPECTED_INPUTS[1]]), float(r[EXPECTED_INPUTS[2]])
                if mode in ["ML-direct", "Both"] and models_dict:
                    feats, feat_names = _build_feature_vector(t, h, s)
                    outs = _predict_direct(models_dict, scaler_obj, feats, feat_names)
                    out_rows.append(_pack_predictions_row(t, h, s, outs) | {'Model': 'ML-direct'})
                if mode in ["KNN", "Both"] and hist_df is not None:
                    outs, stds, nsel = _knn_match(hist_df, t, h, s, tol_temp, tol_rh, tol_speed, k_max)
                    out_rows.append(_pack_predictions_row(t, h, s, outs) | {'Model': f'KNN (n={nsel})'})
            if out_rows:
                df_out = pd.DataFrame(out_rows)
                st.dataframe(df_out, use_container_width=True)
                st.session_state['last_batch_df'] = df_out
            else:
                st.warning("Nothing computed. Ensure required inputs and uploads are provided.")
        except Exception as e:
            st.error(f"Batch error: {e}")

# --- Explainability / AD
with tab4:
    st.subheader("Explainability and Applicability Domain (AD)")
    st.write("Upload an optional training feature table to enable percentile-based AD flags and SHAP (for ML-direct).")
    feat_file = st.file_uploader("Training features CSV/XLSX (optional)", type=["csv", "xlsx", "xls"], key="featupload")
    train_feats = None
    if feat_file is not None:
        try:
            train_feats = _read_dataframe(feat_file)
            train_feats = _normalize_input_columns(train_feats)
            st.caption(f"Training feature table shape: {train_feats.shape}")
        except Exception as e:
            st.error(f"Could not read features: {e}")

    c1, c2, c3 = st.columns(3)
    t_e = c1.number_input("Temperature (°C) for explain", -10.0, 60.0, 35.0, 0.1, key="exp_t")
    rh_e = c2.number_input("RH (%) for explain", 1.0, 100.0, 70.0, 0.1, key="exp_rh")
    sp_e = c3.number_input("Speed (m/s) for explain", 0.0, 30.0, 0.5, 0.1, key="exp_sp")

    colA, colB = st.columns(2)
    if colA.button("Compute AD flags", use_container_width=True):
        if train_feats is not None:
            bounds = {}
            for name in EXPECTED_INPUTS:
                if name in train_feats.columns:
                    lo, hi = np.percentile(train_feats[name].dropna().to_numpy(), [1, 99])
                    bounds[name] = (lo, hi)
            ad_df = pd.DataFrame(bounds, index=['p01', 'p99']).T
            st.dataframe(ad_df, use_container_width=True)
            flags = {nm: (val := {'Wind_in_temperature (°C)': t_e, 'Wind_in_rh (%)': rh_e, 'Wind_in_speed (m/s)': sp_e}[nm]) < bounds[nm][0] or val > bounds[nm][1]
                     for nm in bounds}
            st.info(f"Out-of-domain flags (1%/99% bounds): {flags}")
        else:
            st.warning("Upload training features to enable AD bounds.")

    if colB.button("Compute SHAP for ML-direct", use_container_width=True):
        if shap is None:
            st.error("SHAP not installed. Please ensure 'shap' is in requirements.")
        elif not models_dict:
            st.warning("Upload ML models first or ensure auto-download succeeded.")
        else:
            try:
                feats, feat_names = _build_feature_vector(t_e, rh_e, sp_e)
                X = feats.reshape(1, -1)
                if scaler_obj is not None:
                    try:
                        check_is_fitted(scaler_obj)
                        X = scaler_obj.transform(X)
                    except Exception:
                        pass
                model_keys = list(models_dict.keys())
                chosen_key = st.selectbox("Pick a regressor to explain", model_keys) if model_keys else None
                if chosen_key:
                    model = models_dict[chosen_key]
                    explainer = shap.Explainer(model)
                    sv = explainer(X)
                    vals = sv.values[0] if hasattr(sv, "values") else sv[0].values
                    shap_df = pd.DataFrame({'feature': feat_names, 'shap_value': vals})
                    st.dataframe(shap_df, use_container_width=True)
                    st.session_state['last_shap_df'] = shap_df
                    st.caption("Bar chart of SHAP values")
                    st.bar_chart(shap_df.set_index('feature'))
            except Exception as e:
                st.error(f"SHAP error: {e}")

# --- Downloads
with tab5:
    st.subheader("Session downloads")
    for key, label in [
        ('last_single_df',  "Single-point results CSV"),
        ('last_compare_df', "Compare results CSV"),
        ('last_batch_df',   "Batch results CSV"),
        ('last_shap_df',    "SHAP values CSV")
    ]:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame):
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button(label=label, data=buf.getvalue(), file_name=f"{key}.csv", mime="text/csv")

st.caption("Tip: Run locally with `pip install -r requirements.txt && streamlit run awg_app.py`. For cloud deploy, include the same requirements.")
