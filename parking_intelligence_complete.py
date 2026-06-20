import json
import warnings
import joblib
import numpy  as np
import pandas as pd
warnings.filterwarnings('ignore')

from pathlib import Path

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH     = Path(__file__).parent / "jan to may police violation_anonymized791b166.csv"
ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"
HOTSPOT_CSV   = ARTIFACTS_DIR / "hotspot_report.csv"

TOP_N_JUNCTIONS   = 30      # junctions that count as "high-traffic"
DENSITY_QUANTILE  = 0.75    # grid cells above this percentile = dense hotspot
CONGESTION_THRESH = 5       # composite score >= this → high-impact label
PEAK_HOURS        = {7, 8, 9, 17, 18, 19, 20}
HEAVY_VEHICLES    = {"LGV", "HGV", "PRIVATE BUS", "SCHOOL BUS",
                     "TOURIST BUS", "MAXI-CAB", "VAN"}

FEATURE_COLS = [
    "latitude", "longitude",
    "hour", "day_of_week", "month", "is_weekend", "is_peak_hour", "time_of_day",
    "grid_violation_count",
    "is_top_junction",
    "has_main_road", "has_footpath", "has_double", "has_bustop",
    "violation_count_per_record",
    "is_heavy_vehicle", "vehicle_type_enc", "police_station_enc",
]


def load_and_engineer(path: str) -> tuple[pd.DataFrame, pd.Series,
                                          pd.DataFrame, LabelEncoder,
                                          LabelEncoder, float, list]:
    """
    Load raw CSV, build all features, return:
        X, y, hotspot_density_df, le_vehicle, le_station, density_threshold, top_junctions
    """
    df = pd.read_csv(path)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="ISO8601")

    # Temporal
    df["hour"]         = df["created_datetime"].dt.hour
    df["day_of_week"]  = df["created_datetime"].dt.dayofweek
    df["month"]        = df["created_datetime"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["time_of_day"]  = pd.cut(
        df["hour"], bins=[-1, 6, 12, 17, 21, 24], labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # ── Spatial grid (≈100 m resolution) ──────────────────────
    df["lat_grid"] = (df["latitude"]  * 100).round() / 100
    df["lon_grid"] = (df["longitude"] * 100).round() / 100

    hotspot_density = (
        df.groupby(["lat_grid", "lon_grid"])
        .size()
        .reset_index(name="grid_violation_count")
    )
    df = df.merge(hotspot_density, on=["lat_grid", "lon_grid"], how="left")
    density_threshold = float(hotspot_density["grid_violation_count"].quantile(DENSITY_QUANTILE))

    # ── Junction flag ──────────────────────────────────────────
    top_junctions = df["junction_name"].value_counts().head(TOP_N_JUNCTIONS).index.tolist()
    df["is_top_junction"] = df["junction_name"].isin(top_junctions).astype(int)

    # ── Violation-type binary flags ────────────────────────────
    df["has_main_road"] = df["violation_type"].str.contains(
        "PARKING IN A MAIN ROAD", case=False, na=False).astype(int)
    df["has_footpath"]  = df["violation_type"].str.contains(
        "PARKING ON FOOTPATH",    case=False, na=False).astype(int)
    df["has_double"]    = df["violation_type"].str.contains(
        "DOUBLE PARKING",         case=False, na=False).astype(int)
    df["has_bustop"]    = df["violation_type"].str.contains(
        "PARKING NEAR BUSTOP",    case=False, na=False).astype(int)
    df["violation_count_per_record"] = df["violation_type"].str.count(",") + 1

    # ── Vehicle severity ───────────────────────────────────────
    df["is_heavy_vehicle"] = df["vehicle_type"].isin(HEAVY_VEHICLES).astype(int)

    # ── Label encoding ─────────────────────────────────────────
    le_vehicle = LabelEncoder()
    le_station = LabelEncoder()
    df["vehicle_type_enc"]  = le_vehicle.fit_transform(df["vehicle_type"].fillna("UNKNOWN"))
    df["police_station_enc"]= le_station.fit_transform(df["police_station"].fillna("UNKNOWN"))

    # ── Target: composite congestion score → binary ────────────
    print("\nSTEP 2 — Building Congestion Impact Target")
    congestion_score = (
        (df["grid_violation_count"] >= density_threshold).astype(int) * 3
        + df["has_main_road"]    * 2
        + df["is_heavy_vehicle"] * 2
        + df["is_top_junction"]  * 2
        + df["has_double"]       * 2
        + df["is_peak_hour"]     * 1
        + df["has_footpath"]     * 1
    )
    df["high_congestion_impact"] = (congestion_score >= CONGESTION_THRESH).astype(int)

    hi = df["high_congestion_impact"].sum()
    lo = len(df) - hi
    print(f"  Density threshold (p{int(DENSITY_QUANTILE*100)}): {density_threshold:.0f} violations/cell")
    print(f"  High-impact : {hi:,}  ({hi/len(df)*100:.1f}%)")
    print(f"  Low-impact  : {lo:,}  ({lo/len(df)*100:.1f}%)")

    X = df[FEATURE_COLS].fillna(0)
    y = df["high_congestion_impact"]

    return X, y, hotspot_density, le_vehicle, le_station, density_threshold, top_junctions


def build_models(y_train: pd.Series) -> dict:
    """Return a dict of untrained model instances."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_w = neg / pos if pos > 0 else 1.0

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight="balanced", C=1.0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1, class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_w,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            is_unbalance=True, random_state=42, verbosity=-1,
        ),
    }


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    X_train_sc, X_test_sc,
    models: dict,
) -> tuple[dict, dict]:
    """Train every model, collect metrics, return results + trained model objects."""
    print("\n" + "=" * 60)
    print("STEP 3 — Training & Evaluating All Models")
    print("=" * 60)

    results       = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n-> {name}")
        Xtr = X_train_sc if name == "Logistic Regression" else X_train
        Xte = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        acc  = accuracy_score (y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score   (y_test, y_pred, zero_division=0)
        f1   = f1_score       (y_test, y_pred, zero_division=0)
        auc  = roc_auc_score  (y_test, y_proba)

        results[name] = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1 Score":  round(f1,   4),
            "ROC-AUC":   round(auc,  4),
        }
        trained_models[name] = model

        print(f"   Accuracy : {acc:.4f}")
        print(f"   Precision: {prec:.4f}  |  Recall: {rec:.4f}")
        print(f"   F1 Score : {f1:.4f}  |  ROC-AUC: {auc:.4f}")

    return results, trained_models


def print_comparison(results: dict) -> str:
    """Pretty-print model comparison table; return name of best model."""
    print("\n" + "=" * 60)
    print("STEP 4 — Model Comparison Summary")
    print("=" * 60)
    df_res = pd.DataFrame(results).T.sort_values("F1 Score", ascending=False)
    print(df_res.to_string())

    best = df_res["F1 Score"].idxmax()
    print(f"\n[SUCCESS] BEST MODEL : {best}")
    print(f"   F1={results[best]['F1 Score']}  ROC-AUC={results[best]['ROC-AUC']}")
    return best


def save_artifacts(
    trained_models: dict,
    best_model_name: str,
    scaler: StandardScaler,
    le_vehicle: LabelEncoder,
    le_station: LabelEncoder,
    results: dict,
    top_junctions: list,
    density_threshold: float,
    X_test: pd.DataFrame,
    X_test_sc: np.ndarray,
    best_model_name_str: str,
) -> None:
    """Persist every artifact needed for inference."""
    print("\n" + "=" * 60)
    print("STEP 5 — Saving Artifacts")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # All individual models
    for name, model in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, ARTIFACTS_DIR / fname)
        print(f"  Saved: {fname}")

    best_model = trained_models[best_model_name]

    # Best model (shortcut for inference)
    joblib.dump(best_model,  ARTIFACTS_DIR / "best_model.pkl")
    joblib.dump(scaler,      ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(le_vehicle,  ARTIFACTS_DIR / "le_vehicle.pkl")
    joblib.dump(le_station,  ARTIFACTS_DIR / "le_station.pkl")
    print("  Saved: best_model.pkl, scaler.pkl, le_vehicle.pkl, le_station.pkl")

    # Metadata for frontend
    metadata = {
        "best_model":             best_model_name,
        "feature_columns":        FEATURE_COLS,
        "target":                 "high_congestion_impact",
        "target_description":     "1 = High congestion impact hotspot, 0 = Low impact",
        "density_threshold":      density_threshold,
        "congestion_score_threshold": CONGESTION_THRESH,
        "model_metrics":          results,
        "top_junctions":          top_junctions,
        "vehicle_type_classes":   le_vehicle.classes_.tolist(),
        "police_station_classes": le_station.classes_.tolist(),
        "congestion_score_weights": {
            "grid_density_above_p75": 3,
            "main_road_parking":      2,
            "heavy_vehicle":          2,
            "top_junction":           2,
            "double_parking":         2,
            "peak_hour":              1,
            "footpath_parking":       1,
        },
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved: metadata.json")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        fi.sort_values(ascending=False).to_csv(ARTIFACTS_DIR / "feature_importance.csv")
        print("  Saved: feature_importance.csv")
        print(f"\n  Top-10 features ({best_model_name}):")
        print(fi.nlargest(10).to_string())

    # Hotspot report
    Xte_use = X_test_sc if best_model_name == "Logistic Regression" else X_test
    X_tmp   = X_test.copy()
    X_tmp["predicted_high_impact"]  = best_model.predict(Xte_use)
    X_tmp["congestion_probability"] = best_model.predict_proba(Xte_use)[:, 1]
    X_tmp["lat_grid"] = X_tmp["latitude"].apply(lambda v: round(v * 100) / 100)
    X_tmp["lon_grid"] = X_tmp["longitude"].apply(lambda v: round(v * 100) / 100)

    hotspot_report = (
        X_tmp[X_tmp["predicted_high_impact"] == 1]
        .groupby(["lat_grid", "lon_grid"])
        .agg(
            violation_density   =("grid_violation_count", "mean"),
            avg_congestion_prob =("congestion_probability", "mean"),
            record_count        =("latitude", "count"),
        )
        .reset_index()
        .sort_values("avg_congestion_prob", ascending=False)
        .head(50)
    )
    hotspot_report.to_csv(ARTIFACTS_DIR / "hotspot_report.csv", index=False)
    print("  Saved: hotspot_report.csv  (top-50 predicted hotspot cells)")

    print("\n" + "=" * 60)
    print("[SUCCESS] PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts in: {ARTIFACTS_DIR.resolve()}/")
    for fname in sorted(ARTIFACTS_DIR.iterdir()):
        size_kb = fname.stat().st_size / 1024
        print(f"  {fname.name:<40s}  {size_kb:>8.1f} KB")


def run_training_pipeline():
    """End-to-end: load → engineer → train → compare → save."""
    (X, y,
     hotspot_density, le_vehicle, le_station,
     density_threshold, top_junctions) = load_and_engineer(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train : {X_train.shape[0]:,} rows")
    print(f"  Test  : {X_test.shape[0]:,} rows")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models                  = build_models(y_train)
    results, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc, models,
    )
    best_model_name = print_comparison(results)

    save_artifacts(
        trained_models, best_model_name,
        scaler, le_vehicle, le_station,
        results, top_junctions, density_threshold,
        X_test, X_test_sc, best_model_name,
    )


_best_model  = None
_scaler      = None
_le_vehicle  = None
_le_station  = None
_META        = None
_TOP_JUNCTIONS    = None
_FEATURE_COLS_INF = None
_DENSITY_THRESH   = None
_BEST_MODEL_NAME  = None


def _load_inference_artifacts():
    """Load artifacts once; reuse on subsequent calls."""
    global _best_model, _scaler, _le_vehicle, _le_station
    global _META, _TOP_JUNCTIONS, _FEATURE_COLS_INF
    global _DENSITY_THRESH, _BEST_MODEL_NAME

    if _best_model is not None:
        return  # already loaded

    _best_model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    _scaler     = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    _le_vehicle = joblib.load(ARTIFACTS_DIR / "le_vehicle.pkl")
    _le_station = joblib.load(ARTIFACTS_DIR / "le_station.pkl")

    with open(ARTIFACTS_DIR / "metadata.json") as f:
        _META = json.load(f)

    _TOP_JUNCTIONS    = set(_META["top_junctions"])
    _FEATURE_COLS_INF = _META["feature_columns"]
    _DENSITY_THRESH   = _META["density_threshold"]
    _BEST_MODEL_NAME  = _META["best_model"]
    print(f"[OK] Loaded model: {_BEST_MODEL_NAME}")


def _safe_encode(encoder: LabelEncoder, value: str, fallback: str = "UNKNOWN") -> int:
    """Encode a label; fall back gracefully for unseen values."""
    classes = list(encoder.classes_)
    if value in classes:
        return classes.index(value)
    if fallback in classes:
        return classes.index(fallback)
    return 0


def predict_congestion_impact(
    latitude:             float,
    longitude:            float,
    created_datetime:     str,
    violation_type:       str,
    vehicle_type:         str,
    police_station:       str,
    junction_name:        str = "",
    grid_violation_count: int = None,
) -> dict:
    """
    Predict whether a parking violation causes high congestion impact.

    Parameters
    ----------
    latitude, longitude       : GPS coordinates of the violation
    created_datetime          : ISO8601 timestamp  e.g. "2024-03-15T08:30:00+00:00"
    violation_type            : Raw string from DB  e.g. '["PARKING IN A MAIN ROAD"]'
    vehicle_type              : e.g. "CAR", "LGV", "SCOOTER"
    police_station            : Station code  e.g. "BTP051"
    junction_name             : Optional junction name from the known list
    grid_violation_count      : Pre-computed cell density; pass None to use median

    Returns
    -------
    dict with keys:
        high_congestion_impact  : int   (0 or 1)
        congestion_probability  : float (0 – 1)
        risk_level              : str   ("LOW" | "MEDIUM" | "HIGH" | "CRITICAL")
        key_risk_factors        : list[str]
    """
    _load_inference_artifacts()

    dt          = pd.to_datetime(created_datetime, utc=True)
    hour        = dt.hour
    day_of_week = dt.dayofweek
    month       = dt.month
    is_weekend  = int(day_of_week >= 5)
    is_peak_hour= int(hour in PEAK_HOURS)
    time_of_day = int(pd.cut([hour], bins=[-1, 6, 12, 17, 21, 24],
                              labels=[0, 1, 2, 3, 4])[0])

    vt = violation_type.upper()
    has_main_road = int("PARKING IN A MAIN ROAD" in vt)
    has_footpath  = int("PARKING ON FOOTPATH"    in vt)
    has_double    = int("DOUBLE PARKING"          in vt)
    has_bustop    = int("PARKING NEAR BUSTOP"     in vt)
    viol_count    = vt.count(",") + 1

    is_heavy       = int(vehicle_type.upper() in HEAVY_VEHICLES)
    is_top_junction= int(junction_name in _TOP_JUNCTIONS)

    gvc = (grid_violation_count
           if grid_violation_count is not None
           else int(_DENSITY_THRESH * 0.5))

    vehicle_enc = _safe_encode(_le_vehicle, vehicle_type.upper())
    station_enc = _safe_encode(_le_station, police_station.upper())

    row = pd.DataFrame([{
        "latitude":                  latitude,
        "longitude":                 longitude,
        "hour":                      hour,
        "day_of_week":               day_of_week,
        "month":                     month,
        "is_weekend":                is_weekend,
        "is_peak_hour":              is_peak_hour,
        "time_of_day":               time_of_day,
        "grid_violation_count":      gvc,
        "is_top_junction":           is_top_junction,
        "has_main_road":             has_main_road,
        "has_footpath":              has_footpath,
        "has_double":                has_double,
        "has_bustop":                has_bustop,
        "violation_count_per_record":viol_count,
        "is_heavy_vehicle":          is_heavy,
        "vehicle_type_enc":          vehicle_enc,
        "police_station_enc":        station_enc,
    }])[ _FEATURE_COLS_INF]

    pred  = int(_best_model.predict(row)[0])
    prob  = float(_best_model.predict_proba(row)[0][1])

    if   prob >= 0.85: risk = "CRITICAL"
    elif prob >= 0.65: risk = "HIGH"
    elif prob >= 0.40: risk = "MEDIUM"
    else:              risk = "LOW"

    factors = []
    if gvc >= _DENSITY_THRESH: factors.append("Dense violation hotspot")
    if has_main_road:           factors.append("Main road obstruction")
    if is_heavy:                factors.append("Heavy vehicle")
    if is_peak_hour:            factors.append("Peak hour violation")
    if is_top_junction:         factors.append("High-traffic junction")
    if has_double:              factors.append("Double parking")
    if has_footpath:            factors.append("Footpath blocked")

    return {
        "high_congestion_impact": pred,
        "congestion_probability": round(prob, 4),
        "risk_level":             risk,
        "key_risk_factors":       factors,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bulk prediction on a DataFrame that has (at minimum) the columns:
        latitude, longitude, created_datetime, violation_type,
        vehicle_type, police_station
    Optional: junction_name, grid_violation_count

    Adds three columns and returns the augmented DataFrame:
        predicted_impact, congestion_probability, risk_level
    """
    _load_inference_artifacts()

    records = [
        predict_congestion_impact(
            latitude             = row.latitude,
            longitude            = row.longitude,
            created_datetime     = row.created_datetime,
            violation_type       = row.violation_type,
            vehicle_type         = row.vehicle_type,
            police_station       = row.police_station,
            junction_name        = getattr(row, "junction_name", ""),
            grid_violation_count = getattr(row, "grid_violation_count", None),
        )
        for _, row in df.iterrows()
    ]
    out = df.copy()
    out["predicted_impact"]       = [r["high_congestion_impact"]  for r in records]
    out["congestion_probability"] = [r["congestion_probability"]   for r in records]
    out["risk_level"]             = [r["risk_level"]               for r in records]
    return out


if __name__ == "__main__":
    # ── Run full training pipeline ─────────────────────────────
    run_training_pipeline()

    # ── Quick inference smoke test ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Inference Smoke Test")
    print("=" * 60)

    sample_result = predict_congestion_impact(
        latitude             = 12.98,
        longitude            = 77.58,
        created_datetime     = "2024-03-15T08:30:00+00:00",
        violation_type       = '["PARKING IN A MAIN ROAD","WRONG PARKING"]',
        vehicle_type         = "LGV",
        police_station       = "BTP051",
        junction_name        = "BTP051 - Safina Plaza Junction",
        grid_violation_count = 24000,
    )

    print("\nSample input  -> Main road + LGV + peak hour + dense hotspot")
    print("Prediction    :")
    for k, v in sample_result.items():
        print(f"  {k:<26s}: {v}")
