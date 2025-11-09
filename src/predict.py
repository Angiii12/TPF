# --- bootstrapping para que 'src' sea importable al ejecutar como script ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import robusto (sirve tanto con -m como sin -m)
try:
    from src.pipeline_preprocesamiento import main as preprocess
except ModuleNotFoundError:
    from pipeline_preprocesamiento import main as preprocess


from pathlib import Path
import sys, json, os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- cargar modelo desde el registry ---
# --- Reemplazo de load_model en src/predict.py ---
def load_model(version="latest"):
    import json, joblib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    reg_path = root / "models" / "model_registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8"))

    def _resolve_artifact(pth):
        p = Path(pth)
        return p if p.is_absolute() else (root / p)

    # --- Formato nuevo: {"latest": "vX.Y.Z", "models": [ {...} ]} ---
    if isinstance(reg, dict) and "models" in reg:
        if version == "latest":
            version = reg.get("latest")
            if not version:
                if not reg["models"]:
                    raise ValueError("Registry vacío.")
                version = reg["models"][-1]["version"]
        entry = next((m for m in reg["models"] if m.get("version") == version), None)
        if not entry:
            raise KeyError(f"Versión '{version}' no encontrada en model_registry.json")
        art = entry.get("artifact_path") or entry.get("path")
        if not art:
            raise KeyError(f"Entrada '{version}' sin 'artifact_path'/'path' en model_registry.json")
        return joblib.load(_resolve_artifact(art))

    # --- Formato viejo: {"vX.Y.Z": {"path": ...}, ...} ---
    candidates = {k: v for k, v in reg.items() if isinstance(v, dict) and ("path" in v or "artifact_path" in v)}
    if version == "latest":
        def to_tuple(vs: str):
            vs = vs.lstrip("vV")
            return tuple(int(x) for x in vs.split("."))
        if not candidates:
            raise ValueError("No hay versiones válidas en el registry.")
        version = max(candidates.keys(), key=to_tuple)

    entry = candidates.get(version)
    if not entry:
        raise KeyError(f"Versión '{version}' no encontrada en registry (formato viejo).")
    art = entry.get("artifact_path") or entry.get("path")
    return joblib.load(_resolve_artifact(art))


# --- aplicar el mismo preprocesamiento ---
def load_and_preprocess(filepath: str):
    X_processed, dias, horas = preprocess(filepath)
    return X_processed, dias, horas

# --- predicción principal ---
def predict_consumption(filepath, version="latest"):
    model = load_model(version)
    X_processed, dias, horas = load_and_preprocess(filepath)

    # --- separar y y armar X numérica ---
    y_true = X_processed["y"].copy() if "y" in X_processed.columns else None
    X = X_processed.drop(columns=["y"], errors="ignore").copy()

    # asegurar numéricos y sin NaN
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    # --- alinear columnas con las del entrenamiento ---
    if hasattr(model, "feature_names_in_"):
        faltantes = [c for c in model.feature_names_in_ if c not in X.columns]
        extras    = [c for c in X.columns if c not in model.feature_names_in_]
        if faltantes:
            print(f"[AVISO] Faltan {len(faltantes)} columnas (se rellenan con 0). Ej: {faltantes[:5]}")
        if extras:
            print(f"[AVISO] Hay {len(extras)} columnas extra (se descartan). Ej: {extras[:5]}")
        X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)

    # --- predecir ---
    y_pred = model.predict(X)

    # --- armar salida de predicciones ---
    df = pd.DataFrame({"prediccion_frio_kw": y_pred})
    if y_true is not None and len(y_true) == len(df):
        df["y_true"] = y_true.values
    if dias is not None and len(df) == len(pd.Series(dias)):
        df["fecha"] = pd.Series(dias).values
    if horas is not None and len(df) == len(pd.Series(horas)):
        df["hora"] = pd.Series(horas).values

    # ordenar columnas si están
    orden = [c for c in ["fecha", "hora", "prediccion_frio_kw", "y_true"] if c in df.columns]
    df = df[orden + [c for c in df.columns if c not in orden]]

    # --- métricas si hay y ---
    metrics = None
    if "y_true" in df.columns:
        mae  = float(mean_absolute_error(df["y_true"], df["prediccion_frio_kw"]))
        mse  = float(mean_squared_error(df["y_true"], df["prediccion_frio_kw"]))
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(df["y_true"], df["prediccion_frio_kw"]))
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return df, metrics


# --- CLI simple ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/predict.py <ruta_excel> [version]")
        sys.exit(1)

    in_path = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "latest"

    os.makedirs("results", exist_ok=True)

    df_pred, metrics = predict_consumption(in_path, version)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path("results") / f"predicciones_{version}_{ts}.csv"
    df_pred.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Predicciones guardadas en: {out_csv}")

    if metrics is not None:
        out_json = Path("results") / f"metricas_prediccion_{version}_{ts}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Métricas: {metrics}\nGuardadas en: {out_json}")
    else:
        print("No se calculan métricas porque el preprocesado no devolvió 'y'.")