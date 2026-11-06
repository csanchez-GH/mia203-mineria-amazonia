# Detección de minería artesanal y otras clases raras en imágenes satelitales de la Amazonía

**Proyecto final · MIA-203 Redes Neuronales y Aprendizaje Profundo · Maestría en Inteligencia Artificial, FIIS – UNI**
Autor: Carlos Sánchez - correo: c.sanchez.cornejo@uni.pe

Comparamos funciones de pérdida para el desbalance (BCE, BCE ponderada, focal loss y Asymmetric Loss) y dos
arquitecturas preentrenadas (ResNet-50 y el Vision Transformer DeiT-Small) para detectar **etiquetas raras**,
entre ellas la **minería artesanal**, en 40 479 imágenes satelitales de la cuenca amazónica con 17 etiquetas.
Con Grad-CAM analizamos si el modelo aprendió el atajo "agua = mina".

## Resultados principales

Conjunto de prueba, media ± desviación estándar en 3 semillas (versión oficial del notebook):

|Modelo|Pérdida|mAP raras|AP minería|Recall minería|F2 por imagen|
|-|-|-|-|-|-|
|Línea base (LogReg sobre ResNet-50 congelada)|—|0.347|0.748|0.765|0.895|
|ResNet-50|Focal|0.311 ± 0.018|0.854 ± 0.019|0.850 ± 0.049|0.919 ± 0.001|
|ResNet-50|BCE|0.339 ± 0.012|0.867 ± 0.024|0.869 ± 0.030|0.921 ± 0.001|
|ResNet-50|ASL|0.371 ± 0.017|0.863 ± 0.017|0.882 ± 0.020|0.921 ± 0.001|
|ResNet-50|BCE ponderada|0.427 ± 0.009|0.885 ± 0.013|0.922 ± 0.020|0.921 ± 0.000|
|**DeiT-Small (ViT)**|**BCE ponderada**|**0.520 ± 0.013**|**0.915 ± 0.018**|**0.974 ± 0.030**|**0.927 ± 0.001**|

* El ajuste fino con BCE **no supera** a la línea base en las etiquetas raras; la **BCE ponderada** sí (+8.8 puntos sobre BCE).
* Con la misma pérdida, **DeiT-Small supera a ResNet-50** en las seis etiquetas raras.
* **Grad-CAM:** las falsas alarmas de minería se concentran en **bancos de sedimento de los ríos**, que se parecen a los relaves mineros.
* **Costo:** 1.74 horas de GPU Tesla T4 para los 15 experimentos (unas 2 horas para el notebook completo).

## Estructura del repositorio

```
├── README.md
├── requirements.txt        # versiones exactas del entorno de Kaggle
├── LICENSE                 # MIT (solo el código)
├── data/README.md          # cómo obtener los datos y sus licencias
├── src/                    # el código del notebook organizado por módulos
│   ├── datos.py            #   etiquetas, partición estratificada, caché y DataLoaders
│   ├── perdidas.py         #   BCE, BCE ponderada, focal loss y ASL
│   ├── modelos.py          #   ResNet-50 y DeiT-Small (timm)
│   ├── entrenamiento.py    #   receta de entrenamiento de un experimento
│   ├── metricas.py         #   umbrales por etiqueta, F2 y average precision
│   ├── interpretabilidad.py#   Grad-CAM para CNN y ViT
│   └── regenerar\_tablas.py #   regenera las tablas del artículo sin GPU
├── notebooks/
│   ├── mia203-mineria-amazonia.ipynb   # notebook completo (ejecución oficial, versión 4)
│   └── demo\_colab.ipynb                # demo: carga los pesos y predice en Colab
├── configs/                # configuración del entrenamiento y lista de etiquetas
├── splits/                 # etiquetas binarias y partición train/val/test
├── results/                # tablas, figuras, registros, predicciones y métricas de cada experimento
└── ejemplos/               # imágenes de ejemplo para la demo
```

## Cómo reproducir

### 1\. Instalación

```bash
git clone https://github.com/csanchez-GH/mia203-mineria-amazonia.git
cd mia203-mineria-amazonia
pip install -r requirements.txt
```

### 2\. Datos

Ver [`data/README.md`](data/README.md). Las imágenes se obtienen de Kaggle (competencia *Planet* y dataset
*nikitarom/planets-dataset*).

### 3\. Entrenar y evaluar todo (Kaggle, GPU T4, unas 2 horas)

1. Sube `notebooks/mia203-mineria-amazonia.ipynb` a Kaggle (*Create > New Notebook > File > Import Notebook*).
2. Agrega como entrada la competencia *Planet: Understanding the Amazon from Space* y el dataset *nikitarom/planets-dataset*.
3. Configura **GPU T4** e **Internet: On** (para descargar los pesos preentrenados de timm).
4. Ejecuta *Save Version > Save \& Run All*. Todo lo generado queda en `/kaggle/working`
con la misma estructura de `results/`, `splits/`, `configs/` y `weights/`.

El notebook está dividido en bloques que generan cada parte del artículo:

|Bloque|Qué genera|
|-|-|
|1|Análisis exploratorio y Fig. 1 (distribución de etiquetas)|
|2|Partición train/val/test y Tabla I|
|3 y 3b|Línea base (regresión logística sobre ResNet-50 congelada)|
|4a–4d|Pérdidas, entrenamiento de los 15 experimentos, Tabla IV (ablación de la pérdida)|
|5a|Tablas II y III, Figs. 3 y 4, costo computacional|
|5b|Tabla V (atajo "agua = mina") y Fig. 5 (Grad-CAM)|

### 4\. Regenerar las tablas del artículo (sin GPU, en segundos)

Desde la raíz del repositorio:

```bash
python src/regenerar\_tablas.py
```

Lee los resultados guardados de la ejecución oficial y regenera las Tablas II, III y V
(se guardan en `results/tables/regenerado\_\*.csv`).

### Nota sobre la reproducibilidad

Usamos semillas fijas (partición: 42; experimentos: 0, 1 y 2). Activamos `cudnn.benchmark` para acelerar las
convoluciones, por lo que al volver a entrenar, **los resultados de ResNet-50 pueden variar ligeramente** (en
nuestras pruebas, la media de la mAP de raras cambió entre 0.004 y 0.025 según la pérdida, sin alterar el orden
de las pérdidas). Los resultados de DeiT-Small y de la línea base se reprodujeron exactamente. Todos los números
del artículo y de este README corresponden a la **versión 4** del notebook, cuyos resultados están en `results/`.

## Pesos de los modelos

Los pesos (semilla 0) están en Google Drive: **https://drive.google.com/drive/folders/1IX-yQlUP\_EXZa6-QEunwSYsSOQKttBHp?usp=sharing**



|Archivo|Modelo|
|-|-|
|`vit\_bce\_pond\_s0.pt`|DeiT-Small + BCE ponderada (mejor modelo)|
|`resnet50\_bce\_pond\_s0.pt`|ResNet-50 + BCE ponderada|
|`resnet50\_bce\_s0.pt`, `resnet50\_focal\_s0.pt`, `resnet50\_asl\_s0.pt`|ResNet-50 con las otras pérdidas|

## Demo en Colab

Abre [`notebooks/demo\_colab.ipynb`](notebooks/demo_colab.ipynb) en Colab: https://colab.research.google.com/github/csanchez-GH/mia203-mineria-amazonia/blob/main/notebooks/demo\_colab.ipynb



El notebook clona este repositorio, descarga los pesos de DeiT-Small, predice las 17 etiquetas de las imágenes de `ejemplos/`
(o de las que subas) y muestra el Grad-CAM de "minería artesanal".

## Uso responsable

Una imagen satelital no permite distinguir la minería formal de la ilegal, y el modelo confunde algunos
sedimentos de río con relaves. Por eso debe usarse como **herramienta de priorización con revisión humana**,
nunca como prueba de una actividad ilegal.

## Uso de herramientas de IA

Usamos Claude (Anthropic) como asistente para proponer y discutir el diseño experimental, escribir y depurar
el código, y redactar y revisar los textos. Todos los resultados provienen de nuestras propias ejecuciones en
Kaggle; revisamos y validamos cada salida.

## Referencias principales

* Planet Labs y Kaggle, *Planet: Understanding the Amazon from Space*, 2017.
* K. He et al., "Deep residual learning for image recognition", CVPR 2016.
* H. Touvron et al., "Training data-efficient image transformers \& distillation through attention", ICML 2021.
* T.-Y. Lin et al., "Focal loss for dense object detection", ICCV 2017.
* T. Ridnik et al., "Asymmetric loss for multi-label classification", ICCV 2021.
* R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization", ICCV 2017.

