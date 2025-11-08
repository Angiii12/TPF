import pandas as pd
import numpy as np
import datetime as _dt
from functools import reduce
import joblib
from joblib import load
from pathlib import Path
import json
from sklearn.impute import KNNImputer
from itertools import combinations
import re
import requests

# ----------------- Config -----------------
DAY_COL, HOUR_COL = "DIA", "HORA"
VAL_COL = "Frio (Kw)"
EPS = 1e-6
LIMITE_CONSECUTIVOS = 5

# ----------------- Helpers ----------------
def safe_div(a, b):
    return np.where(b != 0, a / b, 0)

def _load_vars_from_json(json_path: str | Path) -> list[str]:
    """Acepta JSON como lista ['col1', ...] o dict con claves: variables/cols/columns/features.
       También admite dict {col: bool} (toma las que tengan True)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vars_list = None
    if isinstance(data, list):
        vars_list = data
    elif isinstance(data, dict):
        # dict tipo {col: true/false}
        if all(isinstance(k, str) for k in data.keys()) and any(isinstance(v, (bool, int)) for v in data.values()):
            vars_list = [k for k, v in data.items() if bool(v)]
        else:
            for key in ("variables", "cols", "columns", "features"):
                if key in data and isinstance(data[key], list):
                    vars_list = data[key]
                    break

    if not vars_list:
        raise ValueError("El JSON debe ser una lista de columnas o un dict con 'variables/cols/columns/features' o {col: bool}.")

    # limpiar y deduplicar preservando orden
    seen, cleaned = set(), []
    for v in vars_list:
        v = str(v).strip()
        if v and v not in seen:
            seen.add(v)
            cleaned.append(v)
    return cleaned

def filter_df_by_json(df: pd.DataFrame, json_path: str | Path, *, strict: bool = False):
    """Devuelve df filtrado y (usadas, faltantes). Si strict=True y faltan columnas, levanta error."""
    wanted = _load_vars_from_json(json_path)
    missing = [c for c in wanted if c not in df.columns]
    kept    = [c for c in wanted if c in df.columns]

    if strict and missing:
        raise KeyError(f"Columnas faltantes en df: {missing}")

    df_out = df.loc[:, kept].copy()
    if missing:
        ej = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
    return df_out, kept, missing

def _bounds(v):
    if isinstance(v, dict):
        lo = v.get("low", v.get("lower", v.get("min")))
        hi = v.get("high", v.get("upper", v.get("max")))
        if lo is not None and hi is not None:
            return float(lo), float(hi)
        if "Q1" in v and "Q3" in v:
            iqr = float(v.get("IQR", v["Q3"] - v["Q1"]))
            k = float(v.get("k", 1.5))
            return float(v["Q1"] - k*iqr), float(v["Q3"] + k*iqr)
    return float(v[0]), float(v[1])

def apply_iqr_knn(df_filtrado, thresholds_path="thresholds.joblib", exclude=None):
    exclude = set(exclude or [])
    th = joblib.load(thresholds_path)

    df = df_filtrado.copy()

    cols = [c for c in th if c in df.columns and c not in exclude]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        lo, hi = _bounds(th[c])
        df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    if num_cols:  # por si no hay nada que imputar
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        df[num_cols] = imputer.fit_transform(df[num_cols])

    return df

# -------- util de nombres --------
_NBSP = u"\u00A0"
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g.columns = [re.sub(r"\s+", " ", str(c).replace(_NBSP, " ")).strip() for c in g.columns]
    return g

def _require_day_hour(df: pd.DataFrame, hoja: str):
    missing = [c for c in (DAY_COL, HOUR_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"No puedo encontrar el DIA / HORA en la hoja {hoja}: faltan {missing}")

def _to_minutes(h):
    try:
        # soporte extra: horas como float Excel (fracción del día)
        if isinstance(h, (int, float)) and 0 <= float(h) < 1:
            total_seconds = float(h) * 24 * 60 * 60
            hh = int(total_seconds // 3600)
            mm = int((total_seconds % 3600) // 60)
            ss = int(total_seconds % 60)
            return hh*60 + mm + ss/60
        hh, mm, *rest = str(h).split(":")
        ss = int(rest[0]) if rest else 0
        return int(hh)*60 + int(mm) + ss/60
    except Exception:
        return np.nan

def _interpolar_nans_existentes(df, day_col=DAY_COL, hour_col=HOUR_COL):
    g = df.copy()
    # validación fuerte
    if day_col not in g.columns or hour_col not in g.columns:
        raise KeyError("No puedo encontrar el DIA / HORA")

    # index datetime diario (a medianoche) para 'time' interpolate entre días
    g["_fecha"] = pd.to_datetime(g[day_col], errors="coerce", dayfirst=True)
    g = g.dropna(subset=["_fecha"]).sort_values("_fecha").set_index("_fecha")
    num_cols = g.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")
        g[num_cols] = g[num_cols].ffill().bfill()
    # fallback de HORA a 23:59
    if hour_col in g.columns:
        g[hour_col] = g[hour_col].astype(str).str.strip().replace({"": np.nan})
        g[hour_col] = g[hour_col].fillna("23:59:00")
    return g.reset_index(drop=True)

def _prep_por_hoja(nombre, df):
    # asume columna DIA y HORA; renombra y sufija como antes
    g = _clean_cols(df)
    _require_day_hour(g, nombre)

    # normaliza DIA a date
    g["DIA"] = pd.to_datetime(g["DIA"], errors="coerce", dayfirst=True).dt.normalize()
    # ordena y quita duplicados por día (último gana)
    if g.duplicated("DIA").any():
        g = g.sort_values(["DIA"]).drop_duplicates("DIA", keep="last")

    # reordena con DIA primero, sufija resto
    cols_no_clave = [c for c in g.columns if c != "DIA"]
    g = g[["DIA"] + cols_no_clave].add_suffix(f"__{nombre}")
    g = g.rename(columns={f"DIA__{nombre}": "dia"})
    return g

def _unir_diccionario_por_dia(dic, hojas_incluir=None):
    dfs_norm = []
    for nombre, df in dic.items():
        if hojas_incluir and nombre not in hojas_incluir:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty:
            dfs_norm.append(_prep_por_hoja(nombre, df))
    if not dfs_norm:
        return pd.DataFrame(columns=["dia"])
    df_unif = reduce(lambda l, r: pd.merge(l, r, on="dia", how="outer", sort=False), dfs_norm)
    return df_unif.sort_values("dia", kind="stable").reset_index(drop=True)

def _resolver_hojas_a_usar(xls: pd.ExcelFile, hojas_incluir):
    todas = list(xls.sheet_names)

    if hojas_incluir is None:
        hojas_usar = todas
    else:
        # match case-insensitive y preserva el orden pedido por el usuario
        low_map = {h.lower(): h for h in todas}
        hojas_usar = []
        faltantes = []
        for h in hojas_incluir:
            real = low_map.get(str(h).lower())
            if real is None:
                faltantes.append(h)
            else:
                hojas_usar.append(real)
        if faltantes:
            raise KeyError(f"Hojas solicitadas no encontradas en el Excel: {faltantes}")

    # referencia: si está "Consolidado EE" dentro de hojas_usar, úsala; si no, la primera de hojas_usar
    ref_name = "Consolidado EE" if "Consolidado EE" in hojas_usar else (hojas_usar[0] if hojas_usar else None)
    if ref_name is None:
        raise ValueError("El archivo no contiene hojas utilizables.")

    return hojas_usar, ref_name


def aplicar_scaler(df_in, preproc_path, ycol: str = "y"):
    preproc_path = Path(preproc_path)
    preproc = load(preproc_path)  # carga el Pipeline fit-eado

    df = df_in.copy()

    # Detectar las features entrenadas
    if hasattr(preproc, "transformers_"):  # típico de ColumnTransformer (no es tu caso)
        feats = list(preproc.transformers_[0][2])
    elif hasattr(preproc, "feature_names_in_"):
        feats = list(preproc.feature_names_in_)
    else:
        feats = [c for c in df.columns if c != ycol]

    # Asegurar que existan todas las columnas (nota: NaN romperá PowerTransformer)
    for c in feats:
        if c not in df.columns:
            # Recomendación: llenar con 0 en vez de NaN para evitar error
            df[c] = 0.0

    X = preproc.transform(df[feats])
    X_df = pd.DataFrame(X, columns=feats, index=df.index) if isinstance(X, np.ndarray) else X
    df.loc[:, feats] = X_df.values
    return df

# --------------- Pipeline -----------------
def procesar_y_unificar_excel(xls_io, hojas_incluir=None):
    # 1) Carga solo de hojas necesarias (con limpieza de nombres y validación DIA/HORA)
    xls = pd.ExcelFile(xls_io)
    hojas_usar, ref_name = _resolver_hojas_a_usar(xls, hojas_incluir)

    dfs = {}
    for hoja in hojas_usar:
        df = pd.read_excel(xls, sheet_name=hoja)
        df = _clean_cols(df)           # igual que antes
        _require_day_hour(df, hoja)    # igual que antes (KeyError claro si falta)
        dfs[hoja] = df

    # 2) Referencia (sobre el subconjunto)
    df_ref = dfs[ref_name].copy()
    df_ref["_dia"]  = pd.to_datetime(df_ref[DAY_COL], errors="coerce", dayfirst=True).dt.date
    df_ref["_mins"] = df_ref[HOUR_COL].map(_to_minutes)
    df_ref = df_ref.dropna(subset=["_dia", "_mins"]).sort_values(["_dia", "_mins"], kind="stable")

    if df_ref.empty:
        # sin referencia válida: unir directo (no gastamos en 23:59 ni interpolación)
        dfs_interpolado = {k: v.copy() for k, v in dfs.items()}
        return _unir_diccionario_por_dia(dfs_interpolado, hojas_incluir=hojas_usar)

    # --- lo que sigue es igual a tu flujo actual (calculado una sola vez con df_ref) ---
    primer_dia  = df_ref["_dia"].min()
    ultimo_dia  = df_ref["_dia"].max()
    dias_presentes = set(df_ref["_dia"].unique())
    ideal_range = pd.date_range(start=primer_dia, end=ultimo_dia, freq="D").date
    dias_faltantes_lista = sorted(list(set(ideal_range) - dias_presentes))

    # filtrar rachas largas de faltantes
    dias_faltantes_filtrados = []
    if dias_faltantes_lista:
        one_day = _dt.timedelta(days=1)
        grupo = []
        for d in dias_faltantes_lista:
            if not grupo or (d - grupo[-1]) == one_day:
                grupo.append(d)
            else:
                if len(grupo) < LIMITE_CONSECUTIVOS:
                    dias_faltantes_filtrados.extend(grupo)
                grupo = [d]
        if len(grupo) < LIMITE_CONSECUTIVOS:
            dias_faltantes_filtrados.extend(grupo)

    minuto_23_59 = 23*60 + 59
    dias_con_23_59 = set(df_ref[(df_ref["_mins"] >= minuto_23_59) & (df_ref["_mins"] < minuto_23_59 + 1)]["_dia"].unique())
    dias_sin_23_59_lista = sorted(list(dias_presentes - dias_con_23_59))

    registros_por_dia = df_ref.groupby("_dia").size()
    dias_con_horas_faltantes_con_23_59 = sorted(list(set(registros_por_dia[registros_por_dia < 24].index) - set(dias_sin_23_59_lista)))

    # detección de mala monotonía (solo si VAL_COL está en la hoja de referencia)
    dias_mala_monotonia_lista = []
    if VAL_COL in df_ref.columns:
        for d, g in df_ref.groupby("_dia", sort=True):
            s = pd.to_numeric(g[VAL_COL], errors="coerce").fillna(0.0).to_numpy()
            if s.size and (pd.Series(s).diff().fillna(0) < -EPS).any():
                dias_mala_monotonia_lista.append(d)

    lista_para_nan       = set(dias_faltantes_filtrados) | set(dias_con_horas_faltantes_con_23_59) | set(dias_sin_23_59_lista)
    lista_mala_monotonia = set(dias_mala_monotonia_lista)

    # 3) Filtrado 23:59 + crear filas faltantes + anular datos (solo sobre hojas_usar)
    dfs_23_59 = {}
    for hoja, df in dfs.items():
        tmp = df.copy()
        tmp["_dia"]  = pd.to_datetime(tmp[DAY_COL], errors="coerce", dayfirst=True).dt.date
        tmp["_mins"] = tmp[HOUR_COL].map(_to_minutes)
        tmp["_ord"]  = np.arange(len(tmp))
        tmp = tmp.dropna(subset=["_dia", "_mins"]).sort_values(["_dia", "_mins", "_ord"], kind="stable")

        df_23_59 = tmp[(tmp["_mins"] >= minuto_23_59) & (tmp["_mins"] < minuto_23_59 + 1)].copy()

        if not df_23_59.empty:
            mask_mono = df_23_59["_dia"].isin(lista_mala_monotonia)
            df_monos  = df_23_59[mask_mono].drop_duplicates(subset=["_dia"], keep="first")
            df_buenos = df_23_59[~mask_mono].drop_duplicates(subset=["_dia"], keep="last")
            df_unif   = pd.concat([df_buenos, df_monos], ignore_index=True)
        else:
            df_unif = pd.DataFrame(columns=tmp.columns)

        # crear días faltantes (filtrados por límite) con 23:59
        dias_ya = set(df_unif["_dia"]) if not df_unif.empty else set()
        dias_crear = lista_para_nan - dias_ya
        if dias_crear:
            df_crear = pd.DataFrame(sorted(dias_crear), columns=["_dia"])
            df_unif = pd.concat([df_unif, df_crear], ignore_index=True)
            if DAY_COL not in df_unif.columns:  df_unif[DAY_COL]  = np.nan
            if HOUR_COL not in df_unif.columns: df_unif[HOUR_COL] = np.nan
            df_unif[DAY_COL]  = df_unif[DAY_COL].fillna(df_unif["_dia"])
            df_unif[HOUR_COL] = df_unif[HOUR_COL].fillna("23:59:00")

        # anular columnas no clave en días problemáticos
        cols_para_nan = [c for c in df.columns if c not in (DAY_COL, HOUR_COL)]
        if cols_para_nan:
            df_unif.loc[df_unif["_dia"].isin(lista_para_nan), cols_para_nan] = np.nan

        columnas_finales = df.columns.tolist()
        dfs_23_59[hoja] = (
            df_unif.reindex(columns=columnas_finales)
                  .sort_values(DAY_COL, kind="stable")
                  .reset_index(drop=True)
        )

    # 4) Interpolar y sobrescribir (solo sobre hojas_usar)
    dfs_interpolado = {k: _interpolar_nans_existentes(v) for k, v in dfs_23_59.items() if not v.empty}

    # 5) Unificación por día (restringida a hojas_usar)
    df_unificado = _unir_diccionario_por_dia(dfs_interpolado, hojas_incluir=hojas_usar)

    # 6) HORA final con prioridad original pero filtrada al subconjunto
    preferencia_horas = [
        'HORA__Consolidado KPI',
        'HORA__Consolidado Produccion',
        'HORA__Consolidado EE',
        'HORA__Consolidado Agua',
        'HORA__Consolidado GasVapor',
        'HORA__Consolidado Aire',
        'HORA__Totalizadores Energia',
    ]
    # quedarnos solo con las que pertenezcan a hojas_usar y existan en columnas
    candidatas = [c for c in preferencia_horas
                  if (c in df_unificado.columns) and (c.split("__", 1)[1] in hojas_usar)]
    if not candidatas:
        # fallback: cualquier HORA__* presente (por si la lista anterior no contempla algún nombre)
        candidatas = [c for c in df_unificado.columns if c.startswith("HORA__")]

    if not candidatas:
        raise KeyError("No puedo encontrar el HORA final (ninguna columna HORA__* presente tras unificar).")

    hora_src = candidatas[0]
    df_unificado["HORA"] = pd.to_datetime(
        df_unificado[hora_src].astype(str).str.strip(),
        format="%H:%M:%S", errors="coerce"
    ).dt.time

    # drop de todas las HORA__* que hayan quedado (solo de las hojas usadas)
    drop_horas = [c for c in df_unificado.columns if c.startswith("HORA__")]
    if drop_horas:
        df_unificado = df_unificado.drop(columns=drop_horas)

    return df_unificado



def outliers_detection(df_unificado, json_path, thresholds_path):
    # Nos quedmaos solo con las columnas que no eran nulas o muchos ceros
    df_filtrado, usadas, faltantes = filter_df_by_json(df_unificado, json_path, strict=False)
    df_filtrado = df_filtrado.copy()
    # TAMBIEN DEBEMOS SUBIR LA RUTA DE LOS THRESHOLDS
    # --- Aplicar IQR + KNN Imputer ---
    df_imputado = apply_iqr_knn(df_filtrado, thresholds_path, exclude=["Frio (Kw)__Consolidado EE"])

    df_imputado['y'] = df_imputado['Frio (Kw)__Consolidado EE'].shift(-1)
    df_imputado = df_imputado.iloc[1:-1].reset_index(drop=True)
    df_imputado['Frio'] = df_imputado['Frio (Kw)__Consolidado EE']
    df_imputado = df_imputado.drop(columns=['Frio (Kw)__Consolidado EE'])

    return df_imputado

def feature_engineering(df_imputado):
    df = df_imputado.copy()
    
    # Creacion de Variables de Tiempo
    df['fecha'] = pd.to_datetime(df['dia'], format='%Y-%m-%d', errors='coerce')
    dow = df['fecha'].dt.dayofweek
    df['dow_sin'] = np.sin(2*np.pi * dow / 7)
    df['dow_cos'] = np.cos(2*np.pi * dow / 7)
    m  = df['fecha'].dt.month
    m0 = (m - 1)
    df['mes_sin'] = np.sin(2*np.pi * m0 / 12)
    df['mes_cos'] = np.cos(2*np.pi * m0 / 12)
    df['fin_de_semana'] = df['fecha'].dt.dayofweek.isin([5, 6]).astype(int)

    # Creacion de variables lags y rolling
    ycol = 'Frio'
    df[f'{ycol}_lag7'] = df[ycol].shift(7)
    y_obs = df[ycol].shift(1) # Frio de ayer
    df[f'{ycol}_ma3'] = y_obs.rolling(window=3, min_periods=1).mean()
    df[f'{ycol}_ma7'] = y_obs.rolling(window=7, min_periods=1).mean()

    # Creacion de interacciones
    top5 = ['Sala Maq (Kw)__Consolidado EE', 'Servicios (Kw)__Consolidado EE', 'KW Gral Planta__Consolidado EE', 
            'Planta (Kw)__Consolidado EE', 'Agua Planta (Hl)__Consolidado Agua']
    for a, b in combinations(top5, 2):
        df[f'{a}x{b}'] = df[a] * df[b]
    for cyc in ['dow_sin', 'dow_cos', 'mes_sin', 'mes_cos']:
        if cyc in df.columns:
            for c in top5:
                df[f'{c}_x_{cyc}'] = df[c] * df[cyc]
    if 'fin_de_semana' in df.columns:
        for c in top5:
            df[f'{c}_x_finde'] = df[c] * df['fin_de_semana']

    # Creacion de ratios de areas
    AREAS = {
    "Elaboración": r"Elab|Elabor|Coci|Cocina|Mosto|Lauter|Macer|Paste",
    "Envasado":    r"Envas|Llen|Linea|L[2345]\b",
    "Bodega":      r"Bodega|Bodeg",
    "Servicios":   r"Servicios|Vapor|Gas|Agua|Aire|Caldera|Compres|Chiller|Sala",
    "Sala_Maq":    r"Sala.*Maq",}

    num_cols = df.select_dtypes(include=[np.number]).columns
    area_cols = {}
    for area, pat in AREAS.items():
        regex = re.compile(pat, flags=re.IGNORECASE)
        cols = [c for c in num_cols if regex.search(c)]
        area_cols[area] = cols

    area_sum = {}
    for area, cols in area_cols.items():
        if cols:
            df[f'{area}_sum'] = df[cols].sum(axis=1, skipna=True)
            area_sum[area] = df[f'{area}_sum']
        else:
            df[f'{area}_sum'] = 0.0
            area_sum[area] = df[f'{area}_sum']

    areas_presentes = list(area_sum.keys())
    total_sel = sum(area_sum[a] for a in areas_presentes)
    df['Consumo_Total_Areas'] = total_sel

    for a in areas_presentes:
        df[f'{a}_share'] = safe_div(df[f'{a}_sum'], df['Consumo_Total_Areas'])
    for a, b in combinations(areas_presentes, 2):
        df[f'ratio_{a}_sobre_{b}'] = safe_div(df[f'{a}_sum'], df[f'{b}_sum'])
        df[f'ratio_{b}_sobre_{a}'] = safe_div(df[f'{b}_sum'], df[f'{a}_sum'])

    df = df.replace([np.inf, -np.inf], np.nan)


    # Creacion de estacionalidad y clima
    est_idx = np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
        [0, 1, 2, 3], default=np.nan)
    df['estacion_sin'] = np.sin(2*np.pi * est_idx / 4)
    df['estacion_cos'] = np.cos(2*np.pi * est_idx / 4)

    start_date = df["fecha"].min().date().isoformat()
    end_date   = df["fecha"].max().date().isoformat()
    lat, lon = 32.56717, -116.62509
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_mean"
        "&timezone=auto"
    )
    data = requests.get(url, timeout=60).json()
    wx_d = pd.DataFrame({
        "fecha": pd.to_datetime(data["daily"]["time"]),
        "t2m_mean_C": data["daily"]["temperature_2m_mean"],
    })
    df = df.merge(wx_d, on="fecha", how="left")

    # Limpiar
    df = df.iloc[7:].copy()
    df = df.fillna(0)

    df_imputado = df.copy()
    return df_imputado, df_imputado['dia'], df_imputado['HORA']


def seleccionar_variables(df_seleccion, scaler_path):
    
    variables_30 = ['Frio_ma7', 'Frio', 'ratio_Bodega_sobre_Sala_Maq', 'ratio_Sala_Maq_sobre_Bodega', 'Sala Maq (Kw)__Consolidado EExServicios (Kw)__Consolidado EE', 'Sala Maq (Kw)__Consolidado EE',
                    'Bodega_share', 'Sala Maq (Kw)__Consolidado EE_x_mes_cos', 'ratio_Servicios_sobre_Bodega', 'Servicios (Kw)__Consolidado EE_x_mes_cos',
                    'ratio_Bodega_sobre_Servicios', 'Sala_Maq_sum', 'Sala Maq (Kw)__Consolidado EExPlanta (Kw)__Consolidado EE', 'Frio_lag7',
                    'Frio_ma3', 'Agua Planta (Hl)__Consolidado Agua_x_mes_cos', 'Sala Maq (Kw)__Consolidado EE_x_mes_sin', 'Sala Maq (Kw)__Consolidado EExKW Gral Planta__Consolidado EE',
                    'Aire L4 / Hl__Consolidado KPI', 'KW Gral Planta__Consolidado EE_x_mes_cos', 'Linea 3 (Kw)__Consolidado EE', 'Servicios (Kw)__Consolidado EE',
                    'Red L1 y L2__Consolidado Agua', 'Sala Maq (Kw)__Consolidado EExAgua Planta (Hl)__Consolidado Agua', 'Agua Linea 3/Hl__Consolidado KPI', 'Sala Maq (Kw)__Consolidado EE_x_dow_cos',
                    'CO 2 Linea 4 / Hl__Consolidado KPI', 'Red Paste L4__Consolidado Agua', 'ET Linea 3/Hl__Consolidado KPI', 'Totalizador_Aire_Cocina__Consolidado Aire']

    df_final = df_seleccion.copy()

    # Si falta alguna de las 30, crearla como NaN para que el preprocesador pueda imputar/transformar
    for c in variables_30:
        if c not in df_final.columns:
            df_final[c] = np.nan

    # Quedarme solo con las 30 + 'y'
    df_final = df_final[variables_30 + ['y']]

    # Aplicar el preprocesador único (sin tocar 'y')
    df_final_tx = aplicar_scaler(df_final, scaler_path, ycol="y")

    # Devolver solo las 30 + 'y' (por si el preproc añadió algo más)
    return df_final_tx[variables_30 + ['y']]


def main(xls_io):
    # --- rutas robustas, relativas al repo ---
    THIS_FILE = Path(__file__).resolve()    # .../TPF/src/pipeline_preprocesamiento.py
    REPO_DIR  = THIS_FILE.parents[1]        # .../TPF
    DATA_DIR  = REPO_DIR / "data"           # .../TPF/data

    json_path      = DATA_DIR / "variables.json"
    scaler_path    = DATA_DIR / "preproc_minmax_power.joblib"
    thresholds_path = max(DATA_DIR.glob("thresholds_*.joblib"), key=lambda p: p.stat().st_mtime)     # último thresholds_*.joblib (por fecha)

    #  Validación rápida
    for p in [json_path, scaler_path, thresholds_path]:
        if not p.exists():
            raise FileNotFoundError(f"No se encontró: {p}")

    hojas_incluir = ["Consolidado KPI","Consolidado Produccion","Consolidado EE", "Consolidado Agua",
                     "Consolidado GasVapor","Consolidado Aire", "Totalizadores Energia"]
    df_unificado = procesar_y_unificar_excel(xls_io, hojas_incluir=hojas_incluir)
    df_impuratado = outliers_detection(df_unificado, json_path, thresholds_path)
    df_final, dias, horas = feature_engineering(df_impuratado)
    df_seleccion = seleccionar_variables(df_final, scaler_path)

    return df_seleccion, dias, horas