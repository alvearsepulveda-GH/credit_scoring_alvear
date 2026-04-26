# 🏦 Credit Scoring Model - Proyecto Alejandro Alvear

Este repositorio contiene el desarrollo de un sistema de **Credit Scoring** profesional para la predicción de riesgo de impago (Default). El proyecto aplica estándares de la industria bancaria y una arquitectura de software modular para asegurar la robustez y reproducibilidad de los resultados.

---

## 🛠️ 1. Metodología: Ingeniería de Variables (WoE & IV)
El núcleo del modelo se basa en la transformación de datos financieros mediante:
- **Information Value (IV):** Utilizado para la selección de las variables con mayor poder predictivo, eliminando aquellas que no aportan señal al modelo.
- **Weight of Evidence (WoE):** Aplicado para linealizar la relación entre las variables independientes y el riesgo, manejando de forma robusta los valores atípicos y nulos.

## 🏗️ 2. Arquitectura del Proyecto (Pipeline Modular)
A diferencia de un análisis en un único archivo, este proyecto está organizado en módulos especializados dentro de la carpeta `src/`:

* `data.py`: Pipeline de carga de datos y validación de esquemas.
* `features.py`: Lógica de cálculo de tablas WoE y transformación de datasets.
* `models.py`: Funciones para el entrenamiento simultáneo, evaluación y serialización.
* `metrics.py`: Implementación de métricas de desempeño financiero (AUC-ROC y Costo Total).

## 📊 3. Comparación y Selección de Modelos
Se evaluaron múltiples algoritmos bajo las mismas condiciones de entrenamiento (`stratified split`) para garantizar una competencia justa. Los resultados obtenidos en el set de prueba son:

| Modelo | AUC-ROC (Test) | Decisión |
| :--- | :--- | :--- |
| **Random Forest** | **0.7838** | 🏆 Seleccionado |
| XGBoost | 0.7766 | Candidato |
| Logistic Regression | 0.7658 | Baseline |

El modelo **Random Forest** fue seleccionado por presentar la mejor capacidad de discriminación. El modelo final y sus metadatos están guardados en la carpeta `models/baseline_v1/`.

## 🚀 4. Reproducción del Ambiente (uv)
Para garantizar que el código funcione exactamente igual en cualquier máquina, se utiliza el gestor de paquetes **uv**.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/alvearsepulveda-GH/credit_scoring_alvear.git](https://github.com/alvearsepulveda-GH/credit_scoring_alvear.git)
   cd credit_scoring_alvear