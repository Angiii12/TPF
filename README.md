# 🚀 TPF Predicción de Consumo Energético

Este repositorio contiene el proyecto final para [Laboratorio de Datos II], enfocado en la predicción de consumo energético utilizando técnicas de Machine Learning.

---

## 🛠️ Instalación y Configuración

Sigue estos pasos para configurar el entorno de trabajo y poder ejecutar el proyecto.

### 1. Clonar el Repositorio

Abre una terminal y clona este repositorio en tu máquina local:

```bash
git clone https://github.com/Angiii12/TPF
```
### 2. Configurar el entorno virtual (recomendado Conda)
```
# 1. Crea un nuevo entorno de Conda (puedes cambiar 'tpf_env' por el nombre que prefieras)
conda create --name tpf_env python=3.9

# 2. Activa el entorno recién creado
conda activate tpf_env

# 3. Instala todas las librerías necesarias desde el archivo requirements.txt
pip install -r requirements.txt
``````

## 🏃 Cómo Ejecutar los Scripts Principales

Una vez que el entorno esté activado (`conda activate tpf_env`), puedes ejecutar los scripts y notebooks del proyecto.

* **Para el Análisis Exploratorio (EDA):**
    * Abrir y ejecutar el notebook: `notebooks/01_EDA.ipynb`

* **Para el Preprocesamiento de Datos:**
    * Abrir y ejecutar el notebook: `notebooks/02_Preprocessing.ipynb`

* **Para entrenar el modelo:**
    ```bash
    python src/train_model.py
    ```

* **Para ejecutar el pipeline de predicción:**
    ```bash
    python src/prediction_pipeline.py
    ``````
