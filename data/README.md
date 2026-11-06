# Datos

Los datos **no se suben al repositorio**. Se obtienen de Kaggle:

| Qué | Fuente | Licencia |
|---|---|---|
| Etiquetas oficiales (`train_v2.csv`) | Competencia [*Planet: Understanding the Amazon from Space*](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) (Planet Labs y Kaggle, 2017) | Reglas de la competencia (hay que aceptarlas en Kaggle) |
| Imágenes (`train-jpg/`, 40 479 JPG de 256 × 256) | Copia pública [*nikitarom/planets-dataset*](https://www.kaggle.com/datasets/nikitarom/planets-dataset), porque la competencia ya no aloja las imágenes | ODbL |

Las imágenes son de la cuenca amazónica (incluido el Perú), tomadas entre enero de 2016 y febrero
de 2017, con unos 3.7 m por píxel.

## Cómo usarlos en Kaggle (recomendado)

1. Acepta las reglas de la competencia (pestaña *Data* o *Rules*).
2. En el notebook, con **"+ Add Input"**, agrega la competencia y el dataset *nikitarom/planets-dataset*.
3. El notebook encuentra solo las rutas: busca la carpeta `train-jpg` y el archivo `train_v2.csv`
   dentro de `/kaggle/input`.

## Archivos derivados incluidos en el repositorio

- `splits/etiquetas_binarias.csv`: una fila por imagen y una columna 0/1 por etiqueta.
- `splits/particion.csv`: partición train/val/test (70/15/15, semilla 42) usada en todos los experimentos.
