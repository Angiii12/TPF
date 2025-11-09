# src/train_model.py
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sklearn

# Importá tu pipeline de prepro
from src.pipeline_preprocesamiento import main as preprocess


# --------------------- Utilidades de rutas / git ---------------------
def get_project_root() -> Path:
    """Detecta la raíz del repo (donde viven src/, models/, results/)."""
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists() or (parent / "src").exists():
            return parent
    return p

def git_commit_hash(root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        return out.decode().strip()
    except Exception:
        return None


# --------------------- Persistencia de registry ---------------------
def actualizar_registry(models_dir: Path, version: str, entry: dict) -> Path:
    registry_path = models_dir / "model_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"latest": None, "models": []}

    # Evitar duplicados por versión
    registry["models"] = [m for m in registry.get("models", []) if m.get("version") != version]
    registry["models"].append(entry)
    registry["latest"] = version

    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    return registry_path

# --------------------- Entrenamiento ---------------------
def train_and_save(
    excel_path: Path,
    version: str,
    params_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    root = get_project_root()
    models_dir = root / "models"
    results_dir = root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocesar datos
    #   Ajustá esta línea si tu pipeline devuelve (X_processed, dias, horas) u otro formato.
    X_processed, *_ = preprocess(str(excel_path))  # ← si tu main devuelve 3 cosas, esta línea ya sirve
    if "y" not in X_processed.columns:
        raise KeyError("No se encontró columna 'y' en X_processed. Ajustá tu pipeline o este script.")

    y = X_processed["y"].copy()
    X = X_processed.drop(columns=["y"]).copy()

    # 2) Split simple (sin shuffle para series temporales)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # 3) Hiperparámetros
    if params_path and params_path.exists():
        best_params_dict = json.loads(params_path.read_text(encoding="utf-8"))
    else:
        best_params_dict = {}  # por defecto: usá los de sklearn y sobreescribí si querés

    # 4) Modelo
    model = RandomForestRegressor(**best_params_dict, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5) Validación
    y_pred = model.predict(X_val)
    mae  = float(mean_absolute_error(y_val, y_pred))
    mse  = float(mean_squared_error(y_val, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_val, y_pred))
    metricas = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # 6) Guardar métricas
    metricas_path = results_dir / "metricas_finales.json"
    metricas_path.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7) Guardar artefacto versionado en models/
    artifact_path = models_dir / f"modelo_{version}.pkl"
    joblib.dump(model, artifact_path)

    # 8) Actualizar registry
    entry = {
        "version": version,
        "model_class": type(model).__name__,
        "hyperparameters": best_params_dict,
        "validation_metrics": metricas,
        "artifact_path": artifact_path.as_posix(),
        "git_commit": git_commit_hash(root),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_source": excel_path.as_posix(),
    }
    registry_path = actualizar_registry(models_dir, version, entry)

    # 9) Prints cortos
    print(f"[OK] Modelo: {artifact_path}")
    print(f"[OK] Métricas: {metricas_path}  ->  {metricas}")
    print(f"[OK] Registry: {registry_path}  (latest = {version})")


# --------------------- CLI ---------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Entrena RandomForest, guarda artefacto versionado y actualiza el registry."
    )
    ap.add_argument("--excel", required=True, help="Ruta al Excel de entrada (datos de entrenamiento).")
    ap.add_argument("--version", default="v1.0.0", help="Versión semántica (vX.Y.Z). Ej: v1.0.0")
    ap.add_argument("--params", default=None, help="Ruta a JSON con hiperparámetros (opcional).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Proporción para validación (default 0.2).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    excel_path = Path(args.excel)
    params_path = Path(args.params) if args.params else None
    train_and_save(
        excel_path=excel_path,
        version=args.version,
        params_path=params_path,
        test_size=args.test_size,
    )