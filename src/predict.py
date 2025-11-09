from pathlib import Path
import sys, json, os
import pandas as pd
import joblib

# --- cargar modelo desde el registry ---
def load_model(version = "latest", registry_path: Path = Path("models/model_registry.json")):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if version == "latest":
        def to_tuple(v: str):
            v = v.lstrip("vV")
            return tuple(int(x) for x in v.split("."))
        version = max(registry.keys(), key=to_tuple)

    model_path = registry[version]["path"]
    return joblib.load(model_path)

# --- aplicar el mismo preprocesamiento ---
def load_and_preprocess(filepath: str):
    from src.pipeline_preprocesamiento import main as preprocess
    X_processed, dias, horas = preprocess(filepath)
    return X_processed, dias, horas

# --- predicción principal ---
def predict_consumption(filepath, version = "latest"):
    model = load_model(version)
    X, dias, horas = load_and_preprocess(filepath)
    y_pred = model.predict(X)
    return pd.DataFrame({
        "fecha": dias,
        "hora": horas,
        "prediccion_frio_kw": y_pred
    })

# --- CLI simple ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/predict.py <ruta_excel> [version]")
        sys.exit(1)

    in_path = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "latest"

    df_pred = predict_consumption(in_path, version)
    os.makedirs("results", exist_ok=True)
    out_path = Path("results/predicciones.csv")
    df_pred.to_csv(out_path, index=False)
    print(f"Predicciones generadas en {out_path}")