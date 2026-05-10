# Resumen de trabajo - Proyecto Credit Scoring Alvear

## Contexto

Este documento resume el trabajo realizado para preparar el proyecto `credit_scoring_alvear` como parte de las actividades del Magister en Finanzas de la Universidad de Santiago de Chile.

El objetivo general fue ordenar el repositorio, preparar una rama de trabajo e integrar una actividad de Machine Learning relacionada con validacion temporal, leakage y walk-forward validation.

## Bitacora de la conversacion

Durante el trabajo se conversaron y abordaron los siguientes temas:

1. Creacion de un agente propio

Se converso sobre como programar un agente propio usando una estructura conceptual basada en:

- Modelo.
- Instrucciones.
- Herramientas.
- Memoria o estado.
- Ciclo de decision.

Tambien se explico que un chatbot solo responde, mientras que un agente puede usar herramientas, leer datos, llamar funciones, ejecutar acciones y apoyar flujos mas complejos.

2. Alternativas sin costo

Se reviso si era posible construir el ejemplo sin costo. La conclusion fue que si es posible avanzar mucho con herramientas gratuitas:

- Visual Studio Code.
- GitHub.
- Python.
- Git.
- `pandas`.
- `scikit-learn`.
- `matplotlib`.
- `jupyter`.
- Un agente local basado en reglas.

Tambien se aclaro que el uso de APIs comerciales, como la API de OpenAI, puede tener costo. Por eso se recomendo partir con una version local, gratuita y simple.

3. Contexto personal y academico

Se considero el contexto de Alejandro:

- Estudiante del MAFI de la Universidad de Santiago de Chile.
- Profesional de Finning.
- Proyecto relacionado con credit scoring y Machine Learning financiero.

Con ese contexto, se propuso orientar el proyecto hacia un agente o sistema de apoyo para analisis de riesgo crediticio.

4. Estructura estandar del proyecto

Se reviso una funcion Python entregada en la tarea para crear una estructura estandar de proyecto ML Finance. La estructura incluia:

```text
data/raw
data/processed
data/external
notebooks
src
models
reports/figures
tests
README.md
.gitignore
environment.yml
```

Se recomendo adaptar el nombre del proyecto a algo como:

```python
crear_estructura_proyecto("agente_credit_scoring", base_dir=".")
```

5. Ruta local del proyecto

Se explico como obtener la ruta de una carpeta en macOS usando Finder, VS Code o el comando:

```bash
pwd
```

Luego se trabajo con la ruta entregada:

```bash
/Users/alejandroalvear/Documents/Alejandro/Desarrollo/Finanzas/MachineLearning/credit_scoring_alvear
```

6. Revision del repositorio existente

Se inspecciono el repositorio y se encontro que ya tenia una estructura avanzada:

```text
src/data.py
src/features.py
src/models.py
src/metrics.py
notebooks/01_credit_scoring.ipynb
notebooks/Pandas_Profiling_Report_Bankloan.html
data/raw/.gitkeep
data/processed/.gitkeep
README.md
environment.yml
pyproject.toml
uv.lock
```

Se detectaron cambios locales previos en:

```text
notebooks/01_credit_scoring.ipynb
notebooks/Pandas_Profiling_Report_Bankloan.html
```

Estos cambios se trataron como trabajo existente del usuario y no se modificaron intencionalmente.

7. Uso basico de Git

Se explico que el mensaje de ayuda de Git aparece cuando se ejecuta `git` sin un subcomando completo.

Tambien se indicaron comandos utiles:

```bash
git status
git add .
git commit -m "mensaje"
git push
```

8. Creacion de rama de trabajo

Para integrar la nueva actividad sin afectar directamente `main`, se creo la rama:

```bash
codex/integra-validation-ts
```

9. Integracion del notebook de clase

El notebook original estaba en:

```bash
/Users/alejandroalvear/Downloads/clase_03_validation_ts.ipynb
```

Se integro como:

```text
notebooks/02_validation_ts.ipynb
```

Primero se copio limpio, sin outputs pesados. Luego se reorganizo para que quedara como notebook propio del proyecto y no como una copia directa de la clase del profesor.

10. Explicacion simple del notebook

Se explico que el notebook trata sobre validacion temporal en finanzas. La idea principal fue:

```text
Un modelo financiero debe entrenar con el pasado y probarse en el futuro.
```

Se explicaron conceptos como:

- Train y test.
- Split aleatorio.
- Split temporal.
- Out-of-time validation.
- Leakage temporal.
- Walk-forward validation.
- Gap temporal.
- Nested cross-validation temporal.
- AUC ROC.

Tambien se conecto esto con credit scoring: no se debe usar informacion posterior a la fecha de evaluacion del cliente para predecir default o mora futura.

11. Modularizacion del notebook

Se transformo el notebook en una version mas profesional. La logica principal fue movida a:

```text
src/validation_ts.py
```

El notebook quedo como una narrativa limpia que llama funciones reutilizables.

12. Errores de ambiente resueltos

Se resolvieron problemas relacionados con Jupyter y dependencias:

- Falta de `ipykernel_launcher`.
- Falta de JupyterLab.
- Falta de `yfinance`.

Se actualizaron:

```text
pyproject.toml
uv.lock
```

13. Ejecucion del notebook

Se dejo indicado que el notebook puede ejecutarse con:

```bash
cd /Users/alejandroalvear/Documents/Alejandro/Desarrollo/Finanzas/MachineLearning/credit_scoring_alvear
uv run jupyter-lab
```

Luego se debe abrir:

```text
notebooks/02_validation_ts.ipynb
```

14. Guardado de la conversacion en Markdown

Finalmente, se creo este archivo Markdown para dejar documentado el trabajo realizado y los temas conversados:

```text
docs/resumen_trabajo_agente_credit_scoring.md
```

## Proyecto

Ruta local del proyecto:

```bash
/Users/alejandroalvear/Documents/Alejandro/Desarrollo/Finanzas/MachineLearning/credit_scoring_alvear
```

Repositorio trabajado:

```text
credit_scoring_alvear
```

Rama creada para esta actividad:

```bash
codex/integra-validation-ts
```

## Estructura revisada

El proyecto ya contaba con una estructura base orientada a Machine Learning financiero:

```text
data/
notebooks/
src/
models/
reports/
README.md
environment.yml
pyproject.toml
uv.lock
```

Tambien se observo que existian cambios locales previos en:

```text
notebooks/01_credit_scoring.ipynb
notebooks/Pandas_Profiling_Report_Bankloan.html
```

Estos archivos no fueron modificados intencionalmente durante la integracion del nuevo notebook.

## Notebook integrado

El notebook original de la clase estaba ubicado en:

```bash
/Users/alejandroalvear/Downloads/clase_03_validation_ts.ipynb
```

Se integro al proyecto como:

```text
notebooks/02_validation_ts.ipynb
```

La primera version fue limpiada para eliminar salidas pesadas y conteos de ejecucion. Luego se reorganizo para que no quedara como una copia directa de la clase, sino como una actividad propia del proyecto.

## Modularizacion realizada

Para dejar el notebook mas limpio y profesional, se traslado la logica principal a:

```text
src/validation_ts.py
```

Este modulo contiene funciones para:

- Descargar precios y calcular retornos con `yfinance`.
- Resumir autocorrelacion temporal.
- Construir datasets con targets futuros.
- Comparar split aleatorio versus split temporal.
- Mostrar ejemplos de leakage.
- Construir features temporales sin mirar el futuro.
- Comparar `KFold` versus `TimeSeriesSplit`.
- Ejecutar walk-forward validation.
- Ejecutar nested walk-forward validation.
- Graficar resultados de validacion.

Con este cambio, el notebook queda mas ordenado y las celdas llaman funciones reutilizables del proyecto.

## Conceptos trabajados

La actividad integrada refuerza los siguientes conceptos de Machine Learning aplicado a finanzas:

- Validacion temporal.
- Out-of-time validation.
- Leakage temporal.
- Split aleatorio versus split cronologico.
- Features con rezagos.
- Variables rolling correctamente desplazadas.
- Walk-forward validation.
- Nested cross-validation temporal.
- Evaluacion mediante AUC ROC.

La idea central es que un modelo financiero debe entrenarse con informacion pasada y evaluarse con informacion futura, simulando el uso real del modelo.

## Relacion con credit scoring

Aunque el notebook utiliza datos de mercado como SPY, AAPL, JPM, XOM y GLD, la logica es aplicable al credit scoring.

En un problema de riesgo crediticio, el modelo no debe usar informacion posterior a la fecha de evaluacion del cliente. Por ejemplo, si se quiere estimar la probabilidad de default de un cliente al momento de solicitar credito, las variables deben estar disponibles antes de observar la mora o el default.

Ejemplo incorrecto:

```text
Predecir mora futura usando informacion posterior al evento de mora.
```

Ejemplo correcto:

```text
Predecir mora futura usando solo informacion disponible hasta la fecha de evaluacion.
```

## Ambiente de ejecucion

El proyecto utiliza `uv` como gestor de dependencias.

Se agregaron y verificaron dependencias necesarias para ejecutar el notebook:

- `ipykernel`
- `jupyterlab`
- `matplotlib`
- `numpy`
- `seaborn`
- `yfinance`

Archivos actualizados:

```text
pyproject.toml
uv.lock
```

## Problemas resueltos

Durante la preparacion se resolvieron los siguientes problemas:

1. Error de kernel por modulo faltante:

```text
ipykernel_launcher
```

Solucion: instalar y fijar una version compatible de `ipykernel`.

2. Falta de JupyterLab en el ambiente:

```text
jupyter
```

Solucion: agregar `jupyterlab` al proyecto.

3. Falta de `yfinance`:

```python
import yfinance as yf
```

Solucion: agregar `yfinance` a `pyproject.toml`, actualizar `uv.lock` y sincronizar el ambiente.

## Comandos utiles

Entrar al proyecto:

```bash
cd /Users/alejandroalvear/Documents/Alejandro/Desarrollo/Finanzas/MachineLearning/credit_scoring_alvear
```

Sincronizar dependencias:

```bash
uv sync
```

Abrir JupyterLab:

```bash
uv run jupyter-lab
```

Abrir el notebook:

```text
notebooks/02_validation_ts.ipynb
```

Ver estado de Git:

```bash
git status
```

Agregar solo los archivos de esta actividad:

```bash
git add README.md pyproject.toml uv.lock notebooks/02_validation_ts.ipynb src/validation_ts.py docs/resumen_trabajo_agente_credit_scoring.md
```

Crear commit:

```bash
git commit -m "Integra validacion temporal para machine learning financiero"
```

Subir rama:

```bash
git push -u origin codex/integra-validation-ts
```

## Verificaciones realizadas

Se verifico que:

- El modulo `src/validation_ts.py` compila correctamente.
- El notebook `notebooks/02_validation_ts.ipynb` es JSON valido.
- El notebook no contiene outputs pesados embebidos.
- Las funciones principales corren con datos sinteticos.
- `yfinance` importa correctamente en el ambiente.
- JupyterLab esta disponible en el ambiente del proyecto.

## Estado final

El proyecto queda con una actividad nueva de Machine Learning financiero, organizada en una estructura mas profesional:

```text
src/validation_ts.py
notebooks/02_validation_ts.ipynb
docs/resumen_trabajo_agente_credit_scoring.md
```

El notebook ahora se ve mas limpio, reutilizable y alineado con una entrega de proyecto, no como una copia directa de una clase.
