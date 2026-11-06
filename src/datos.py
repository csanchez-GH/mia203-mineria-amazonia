"""Carga de etiquetas, partición estratificada, caché de imágenes y DataLoaders."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

# Normalización de ImageNet, porque usamos modelos preentrenados en ImageNet
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def cargar_etiquetas(csv_etiquetas):
    """Convierte el CSV oficial (image_name, tags) en una tabla binaria con una columna por etiqueta."""
    df = pd.read_csv(csv_etiquetas)
    listas = df.tags.str.split()
    etiquetas = sorted({t for lista in listas for t in lista})
    Y = np.array([[int(t in set(lista)) for t in etiquetas] for lista in listas], dtype=np.uint8)
    df_bin = pd.concat([df[['image_name']], pd.DataFrame(Y, columns=etiquetas)], axis=1)
    return df_bin, etiquetas


def etiquetas_raras(df_bin, etiquetas, umbral_pct=1.5):
    """Etiquetas presentes en menos del umbral_pct % de las imágenes."""
    pct = 100 * df_bin[etiquetas].sum() / len(df_bin)
    return pct[pct < umbral_pct].sort_values(ascending=False).index.tolist()


def particion_estratificada(df_bin, etiquetas, semilla=42):
    """Partición 70/15/15 estratificada por la etiqueta más rara de cada imagen."""
    frecuencia = df_bin[etiquetas].sum()
    orden_rareza = frecuencia.sort_values().index.tolist()   # de la más rara a la más común

    def etiqueta_mas_rara(fila):
        for e in orden_rareza:
            if fila[e] == 1:
                return e

    estrato = df_bin.apply(etiqueta_mas_rara, axis=1)
    tamanos = estrato.value_counts()
    estrato[estrato.isin(tamanos[tamanos < 10].index)] = 'otros'

    train_idx, temp_idx = train_test_split(df_bin.index, test_size=0.30, stratify=estrato, random_state=semilla)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=estrato[temp_idx], random_state=semilla)
    particion = pd.DataFrame({'image_name': df_bin.image_name, 'split': ''})
    particion.loc[train_idx, 'split'] = 'train'
    particion.loc[val_idx, 'split'] = 'val'
    particion.loc[test_idx, 'split'] = 'test'
    return particion


def construir_cache(nombres, dir_img, tam=160, ruta='/tmp/cache_160.npy', hilos=8):
    """Decodifica todos los JPG una vez, los reduce a tam x tam y los guarda como matriz uint8."""
    ruta = Path(ruta)
    if not ruta.exists():
        t0 = time.time()
        tmp = ruta.with_name(ruta.stem + '_incompleta.npy')
        cache = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(nombres), tam, tam, 3))

        def leer(nombre):
            with Image.open(Path(dir_img) / f'{nombre}.jpg') as im:
                return np.asarray(im.convert('RGB').resize((tam, tam), Image.BILINEAR))

        with ThreadPoolExecutor(max_workers=hilos) as ex:
            for i, arr in enumerate(ex.map(leer, nombres)):
                cache[i] = arr
        cache.flush()
        del cache
        os.rename(tmp, ruta)   # solo renombramos al terminar: nunca usamos una caché a medias
        print(f'Caché creada en {(time.time() - t0) / 60:.1f} min')
    return np.load(ruta, mmap_mode='r')


def preparar_imagen(arr_uint8):
    """Convierte una imagen HWC uint8 (ya en tam x tam) en un tensor normalizado CHW."""
    x = torch.from_numpy(np.array(arr_uint8)).permute(2, 0, 1).float().div(255)
    return (x - MEDIA) / DESV


class DatasetSatelital(torch.utils.data.Dataset):
    """Lee imágenes de la caché y aplica volteos y rotaciones de 90° si aumentar=True."""

    def __init__(self, cache, Y, indices, aumentar):
        self.cache, self.Y, self.indices, self.aumentar = cache, Y, indices, aumentar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x = torch.from_numpy(np.array(self.cache[i])).permute(2, 0, 1).float().div(255)
        if self.aumentar:
            # Desde el espacio no hay "arriba" ni "abajo": volteos y rotaciones no cambian el significado
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            if torch.rand(1) < 0.5:
                x = x.flip(1)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))
        return (x - MEDIA) / DESV, torch.from_numpy(self.Y[i])


def crear_loader(cache, Y, indices, aumentar, batch=64, mezclar=False, semilla=0, workers=4):
    g = torch.Generator()
    g.manual_seed(semilla)
    return torch.utils.data.DataLoader(DatasetSatelital(cache, Y, indices, aumentar), batch_size=batch,
                                       shuffle=mezclar, num_workers=workers, pin_memory=True,
                                       drop_last=mezclar, generator=g)
