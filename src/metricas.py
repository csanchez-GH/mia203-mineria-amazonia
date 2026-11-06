"""Métricas multietiqueta: umbrales por etiqueta elegidos en validación, F2 y average precision."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score

GRILLA_UMBRALES = np.round(np.arange(0.02, 0.99, 0.02), 2)


def ajustar_umbrales(P_val, Y_val):
    """Para cada etiqueta, el umbral de la grilla que maximiza F2 en validación (nunca en test)."""
    umbrales = np.zeros(P_val.shape[1])
    for j in range(P_val.shape[1]):
        f2 = [fbeta_score(Y_val[:, j], P_val[:, j] >= t, beta=2, zero_division=0) for t in GRILLA_UMBRALES]
        umbrales[j] = GRILLA_UMBRALES[int(np.argmax(f2))]
    return umbrales


def evaluar(P_val, Y_val, P_test, Y_test, etiquetas, raras, etiqueta_mineria='artisinal_mine'):
    """Ajusta umbrales en validación y devuelve (métricas globales, tabla por etiqueta) en test."""
    umbrales = ajustar_umbrales(P_val, Y_val)
    pred = (P_test >= umbrales).astype(int)
    por_etiqueta = pd.DataFrame({
        'umbral': umbrales,
        'f2': fbeta_score(Y_test, pred, beta=2, average=None, zero_division=0),
        'precision': precision_score(Y_test, pred, average=None, zero_division=0),
        'recall': recall_score(Y_test, pred, average=None, zero_division=0),
        'ap': average_precision_score(Y_test, P_test, average=None),
    }, index=etiquetas)
    m = {
        'f2_muestras': fbeta_score(Y_test, pred, beta=2, average='samples', zero_division=0),
        'f2_macro': por_etiqueta.f2.mean(),
        'f2_macro_raras': por_etiqueta.loc[raras, 'f2'].mean(),
        'map_macro': por_etiqueta.ap.mean(),
        'map_raras': por_etiqueta.loc[raras, 'ap'].mean(),
        'f2_mineria': por_etiqueta.loc[etiqueta_mineria, 'f2'],
        'recall_mineria': por_etiqueta.loc[etiqueta_mineria, 'recall'],
        'precision_mineria': por_etiqueta.loc[etiqueta_mineria, 'precision'],
        'ap_mineria': por_etiqueta.loc[etiqueta_mineria, 'ap'],
    }
    return {k: float(v) for k, v in m.items()}, por_etiqueta


def map_raras(P, Y_real, idx_raras):
    """mAP solo sobre las etiquetas raras: métrica principal del proyecto."""
    return float(average_precision_score(Y_real[:, idx_raras], P[:, idx_raras], average='macro'))
