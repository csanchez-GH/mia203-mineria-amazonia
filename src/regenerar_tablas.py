"""Regenera las Tablas II, III y V del artículo a partir de los resultados guardados (sin GPU ni datos).

Uso, desde la raíz del repositorio:
    python src/regenerar_tablas.py

Lee los archivos de results/, splits/ y configs/ que generó la ejecución oficial (versión 4) del notebook,
imprime las tablas y las guarda en results/tables/regenerado_*.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDEN = [('resnet50', 'focal'), ('resnet50', 'bce'), ('resnet50', 'asl'),
         ('resnet50', 'bce_pond'), ('vit', 'bce_pond')]
NOMBRE_ARQ = {'resnet50': 'ResNet-50', 'vit': 'ViT (DeiT-S)'}
NOMBRE_PERD = {'bce': 'BCE', 'bce_pond': 'BCE ponderada', 'focal': 'Focal', 'asl': 'ASL'}
METRICAS = {'map_raras': 'mAP raras', 'f2_macro_raras': 'F2 raras', 'map_macro': 'mAP 17',
            'f2_muestras': 'F2 muestras', 'ap_mineria': 'AP minería',
            'recall_mineria': 'Recall minería', 'precision_mineria': 'Precisión minería'}
SEMILLAS = [0, 1, 2]


def tabla_principal(raiz):
    """Tabla II: media ± desviación estándar (3 semillas) y la línea base."""
    runs = pd.DataFrame([json.load(open(p)) for p in sorted((raiz / 'results/runs').glob('*.json'))])
    runs = runs[runs.semilla.isin(SEMILLAS)]
    base = json.load(open(raiz / 'results/baseline.json'))
    filas = [{'Modelo': 'Línea base (LogReg)', 'Pérdida': '—', **{n: f'{base[k]:.3f}' for k, n in METRICAS.items()}}]
    for arq, perd in ORDEN:
        d = runs[(runs.arquitectura == arq) & (runs.perdida == perd)]
        assert len(d) == len(SEMILLAS), f'Faltan semillas de {arq} + {perd}'
        filas.append({'Modelo': NOMBRE_ARQ[arq], 'Pérdida': NOMBRE_PERD[perd],
                      **{n: f'{d[k].mean():.3f} ± {d[k].std():.3f}' for k, n in METRICAS.items()}})
    return pd.DataFrame(filas), runs


def tabla_por_etiqueta(raiz, raras):
    """Tabla III: average precision por etiqueta rara (promedio de 3 semillas)."""
    tablas = raiz / 'results/tables'
    columnas = {'Línea base': pd.read_csv(tablas / 'baseline_por_etiqueta.csv', index_col=0).loc[raras, 'ap']}
    for arq, perd in ORDEN:
        aps = [pd.read_csv(tablas / f'por_etiqueta_{arq}_{perd}_s{s}.csv', index_col=0).loc[raras, 'ap'] for s in SEMILLAS]
        columnas[f'{NOMBRE_ARQ[arq]} + {NOMBRE_PERD[perd]}'] = pd.concat(aps, axis=1).mean(axis=1)
    t = pd.DataFrame(columnas).round(3)
    t.index.name = 'Etiqueta rara'
    return t


def tabla_atajo(raiz, etiquetas):
    """Tabla V: análisis del atajo "agua = mina" en test, con los modelos de la semilla 0 y BCE ponderada."""
    df_bin = pd.read_csv(raiz / 'splits/etiquetas_binarias.csv')
    particion = pd.read_csv(raiz / 'splits/particion.csv')
    assert (df_bin.image_name.values == particion.image_name.values).all(), 'Etiquetas y partición desalineadas'
    Y_test = df_bin.loc[particion.split.values == 'test', etiquetas].values   # mismo orden que las predicciones
    j_mina, j_agua = etiquetas.index('artisinal_mine'), etiquetas.index('water')
    mina, agua = Y_test[:, j_mina] == 1, Y_test[:, j_agua] == 1
    filas = {}
    for arq in ['resnet50', 'vit']:
        id_exp = f'{arq}_bce_pond_s0'
        P = np.load(raiz / f'results/preds/{id_exp}.npz')['P_test'][:, j_mina]
        umbral = pd.read_csv(raiz / f'results/tables/por_etiqueta_{id_exp}.csv', index_col=0).loc['artisinal_mine', 'umbral']
        pred = P >= umbral
        falsas = pred & ~mina
        filas[NOMBRE_ARQ[arq]] = {
            'Falsas alarmas (n)': int(falsas.sum()),
            '% con agua en falsas alarmas': 100 * agua[falsas].mean() if falsas.any() else np.nan,
            '% con agua en imágenes sin mina': 100 * agua[~mina].mean(),
            'p(mina) media: sin mina y con agua': P[~mina & agua].mean(),
            'p(mina) media: sin mina y sin agua': P[~mina & ~agua].mean(),
            f'Recall en minas con agua (n={int((mina & agua).sum())})': pred[mina & agua].mean(),
            f'Recall en minas sin agua (n={int((mina & ~agua).sum())})': pred[mina & ~agua].mean(),
        }
    return pd.DataFrame(filas).round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raiz', default='.', help='Raíz del repositorio (por defecto, la carpeta actual)')
    raiz = Path(parser.parse_args().raiz)
    config = json.load(open(raiz / 'configs/config_global.json'))
    etiquetas, raras = config['ETIQUETAS'], config['RARAS']

    t2, runs = tabla_principal(raiz)
    t3 = tabla_por_etiqueta(raiz, raras)
    t5 = tabla_atajo(raiz, etiquetas)
    salida = raiz / 'results/tables'
    t2.to_csv(salida / 'regenerado_tabla_II.csv', index=False)
    t3.to_csv(salida / 'regenerado_tabla_III.csv')
    t5.to_csv(salida / 'regenerado_tabla_V.csv')

    pd.set_option('display.width', 250)
    print('===== TABLA II: resultados en prueba (media ± d. e., 3 semillas) =====')
    print(t2.to_string(index=False))
    print('\n===== TABLA III: AP por etiqueta rara (prueba, promedio de 3 semillas) =====')
    print(t3.to_string())
    print('\n===== TABLA V: análisis del atajo "agua = mina" (prueba, semilla 0) =====')
    print(t5.to_string())
    print(f'\nHoras de GPU de entrenamiento (15 experimentos): {runs.minutos_entrenamiento.sum() / 60:.2f}')
    print(f'Tablas guardadas en {salida}/regenerado_*.csv')


if __name__ == '__main__':
    main()
