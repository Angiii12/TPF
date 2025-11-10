# 🧊 Predicción de Consumo Energético Industrial - Planta Cervecera

> Proyecto final del curso **Laboratorio de Datos II**  
> Predicción del consumo energético del sistema de refrigeración (Frio kW) en una planta cervecera mexicana.

---

## 🧠 Descripción General

Este proyecto crea un **pipeline completo de Machine Learning** para predecir el consumo de **Frio (kW)** del día siguiente.  
Incluye:
- Preprocesamiento y análisis exploratorio (EDA)
- Entrenamiento y comparación de modelos
- Registro de versiones y métricas
- Pipeline automático de predicción
- Buenas prácticas de MLOps (trazabilidad, versionado y reproducibilidad)

---

## 📦 Estructura del Proyecto

```

TPF/
├── data/
│   ├── raw/                         # Archivos Excel originales
│   ├── processed/
│   │   ├── dataset_final.csv        # Dataset limpio y procesado
│   │   └── data_lineage.json        # Registro de transformaciones
│   └── checksums.json               # Hash de integridad de datos
│
├── models/
│   ├── modelo_v1.0.0.pkl            # Modelo final entrenado
│   └── model_registry.json          # Registro con versiones y métricas
│
├── notebooks/
│   ├── eda.ipynb                    # Análisis exploratorio
│   ├── preprocesamiento.ipynb       # Limpieza y creación de variables
│   └── modelado.ipynb               # Entrenamiento y evaluación
│
├── src/
│   ├── preprocessing_pipeline.py    # Pipeline reproducible de preprocesamiento
│   ├── train_model.py               # Entrenamiento y registro de modelos
│   ├── predict.py                   # Script para generar predicciones
│   └── auxiliar_functions.py        # Funciones de apoyo
│
├── results/
│   ├── experiment_logs.csv          # Log de experimentos (métricas)
│   └── predicciones.csv             # Salida de predicciones
│
├── requirements.txt
├── .gitignore
└── README.md

````

---

## 🧩 Requisitos Previos

- Tener **Python 3.12 o superior**
- Tener instalado **Git**
- Contar con conexión a internet para descargar librerías
- (Opcional pero recomendado) Tener **Conda** o **Miniconda**

---

## ⚙️ Instalación Paso a Paso

### 🪟 **Para Windows**

1️⃣ **Clonar el repositorio**
```bash
git clone https://github.com/Angiii12/TPF.git
cd TPF
````

2️⃣ **Crear un entorno virtual**

```bash
conda create -n cervecera_env python=3.12
conda activate cervecera_env
```

> 💡 Si no usás Conda:
>
> ```bash
> python -m venv cervecera_env
> cervecera_env\Scripts\activate
> ```

3️⃣ **Instalar las dependencias**

```bash
pip install uv
uv pip install -r requirements.txt
```

4️⃣ **Abrir los notebooks o correr scripts**

* Para abrir notebooks:

  ```bash
  jupyter notebook
  ```
* Para correr los scripts desde consola:

  ```bash
  python src/preprocessing_pipeline.py
  python src/train_model.py
  python src/predict.py data/raw/nuevo_archivo.xlsx
  ```

---

### 🍎 **Para macOS o Linux**

1️⃣ **Clonar el repositorio**

```bash
git clone https://github.com/Angiii12/TPF.git
cd TPF
```

2️⃣ **Crear el entorno**

```bash
conda create -n cervecera_env python=3.12
conda activate cervecera_env
```

> 💡 Alternativa sin conda:
>
> ```bash
> python3 -m venv cervecera_env
> source cervecera_env/bin/activate
> ```

3️⃣ **Instalar dependencias**

```bash
pip install uv
uv pip install -r requirements.txt
```

4️⃣ **Ejecutar notebooks o scripts**

```bash
python3 src/preprocessing_pipeline.py
python3 src/train_model.py
python3 src/predict.py data/raw/nuevo_archivo.xlsx
```

---

## 📊 Flujo de Trabajo del Proyecto

### 🧾 **Fase 1: EDA (Exploración de Datos)**

* Unificar todos los Excel en un solo dataset
* Detectar valores faltantes, outliers y errores
* Analizar correlaciones y patrones temporales
* Visualizar el comportamiento de `Frio (kW)`

---

### 🧹 **Fase 2: Preprocesamiento**

* Limpieza de errores y datos incompletos
* Creación de variables nuevas:
  día, mes, fin de semana, lags, ratios, etc.
* Normalización de variables
* Guardado del dataset final y su checksum

---

### ⚙️ **Fase 3: Modelado**

* Entrenamiento de 4 modelos:

  * XGBoost
  * Random Forest
  * LightGBM
  * Ridge/Lasso
* Comparación de métricas (MAE, RMSE, R²)
* Registro del mejor modelo con versión y hash

---

### 🔮 **Fase 4: Predicción**

El script `src/predict.py` genera predicciones automáticas con el modelo más reciente.

#### Ejemplo:

```bash
python src/predict.py data/raw/nuevo_archivo.xlsx
```

📁 **Salida esperada:** `results/predicciones.csv`

| fecha      | hora  | prediccion_frio_kw |
| ---------- | ----- | ------------------ |
| 2024-11-01 | 23:59 | 14235.6            |
| 2024-11-02 | 23:59 | 14520.1            |
| ...        | ...   | ...                |

---

## 💡 Ejemplo de Uso Completo

```bash
# 1. Clonar proyecto
git clone https://github.com/Angiii12/TPF.git
cd TPF

# 2. Crear entorno
conda create -n cervecera_env python=3.12
conda activate cervecera_env

# 3. Instalar dependencias
pip install uv
uv pip install -r requirements.txt

# 4. Procesar datos y entrenar
python src/preprocessing_pipeline.py
python src/train_model.py

# 5. Predecir con un nuevo archivo Excel
python src/predict.py data/raw/Totalizadores_Planta_Cerveza_2024_2025.xlsx
```

---

## 👩‍💻 Autor

**Angelina, Marcos y Federico**
Estudiante de *Ingeniería en IA* \
📍 Proyecto académico - Predicción de consumo energético industrial


---

## 📄 Licencia

Este proyecto se distribuye con fines educativos bajo la licencia **MIT**.
Podés usarlo, modificarlo o adaptarlo libremente citando la fuente.

---

## 🌟 Consejos Finales

* Si usás VS Code, podés abrir la vista previa del README con `Ctrl + Shift + V`
* Si estás en GitHub, podés editar y ver el resultado con [github.dev](https://github.dev)
* Si todo se instaló bien, deberías poder correr los scripts sin errores desde consola 💪

> 🧊 *"Un pipeline reproducible hoy, es una predicción estable mañana."* 😄


