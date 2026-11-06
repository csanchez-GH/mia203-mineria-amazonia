"""Entrenamiento de un experimento: fine-tuning completo, selección de época en validación y evaluación en test.

Misma receta que el bloque 4b del notebook (configs/config_entrenamiento.json):
AdamW, 6 épocas, 1 época de calentamiento + coseno, lotes de 64, precisión mixta float16.
"""
import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .datos import crear_loader
from .metricas import evaluar, map_raras
from .modelos import crear_modelo
from .perdidas import crear_perdida

CONFIG_ENT = {
    'epocas': 6, 'batch': 64, 'lr': {'resnet50': 2e-4, 'vit': 1e-4}, 'weight_decay': 0.05,
    'epocas_calentamiento': 1, 'tam_imagen': 160,
}


def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    # Acelera las convoluciones; a cambio, ResNet-50 puede variar ligeramente entre ejecuciones
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def predecir(modelo, loader, dispositivo='cuda'):
    """Probabilidades (sigmoide) de todas las imágenes del loader."""
    modelo.eval()
    salidas = []
    for x, _ in loader:
        with torch.autocast('cuda', dtype=torch.float16):
            salidas.append(torch.sigmoid(modelo(x.to(dispositivo, non_blocking=True)).float()).cpu())
    return torch.cat(salidas).numpy()


def entrenar(arquitectura, perdida, semilla, cache, Y, idx, etiquetas, raras,
             config=CONFIG_ENT, dispositivo='cuda'):
    """Entrena un modelo y devuelve (resultado, registro por época, mejor estado, P_val, P_test, tabla por etiqueta).

    idx es un diccionario con los índices de 'train', 'val' y 'test'.
    """
    epocas = config['epocas']
    fijar_semilla(semilla)
    idx_raras = [etiquetas.index(e) for e in raras]

    dl_train = crear_loader(cache, Y, idx['train'], aumentar=True, batch=config['batch'], mezclar=True, semilla=semilla)
    dl_val = crear_loader(cache, Y, idx['val'], aumentar=False, batch=256)
    dl_test = crear_loader(cache, Y, idx['test'], aumentar=False, batch=256)
    Y_val, Y_test = Y[idx['val']], Y[idx['test']]

    modelo = crear_modelo(arquitectura, len(etiquetas), config['tam_imagen']).to(dispositivo)
    criterio = crear_perdida(perdida, Y[idx['train']]).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=config['lr'][arquitectura],
                                    weight_decay=config['weight_decay'])

    pasos_totales = epocas * len(dl_train)
    pasos_calent = config['epocas_calentamiento'] * len(dl_train)

    def factor_lr(paso):
        if paso < pasos_calent:
            return (paso + 1) / pasos_calent
        progreso = (paso - pasos_calent) / max(1, pasos_totales - pasos_calent)
        return 0.5 * (1 + math.cos(math.pi * progreso))

    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, factor_lr)
    escalador = torch.amp.GradScaler('cuda')

    mejor_map, mejor_estado, mejor_epoca, registro = -1.0, None, 0, []
    t_inicio = time.time()
    for epoca in range(1, epocas + 1):
        t_ep = time.time()
        modelo.train()
        suma_perdida, n = 0.0, 0
        for x, y in dl_train:
            x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = modelo(x)
            loss = criterio(logits, y)
            optimizador.zero_grad(set_to_none=True)
            escalador.scale(loss).backward()
            escalador.step(optimizador)
            escalador.update()
            planificador.step()
            suma_perdida += loss.item() * len(y)
            n += len(y)

        # Al final de cada época medimos en validación (nunca en test)
        P_val = predecir(modelo, dl_val, dispositivo)
        m_val_raras = map_raras(P_val, Y_val, idx_raras)
        registro.append({'epoca': epoca, 'perdida_train': suma_perdida / n,
                         'map_val': float(average_precision_score(Y_val, P_val, average='macro')),
                         'map_val_raras': m_val_raras, 'minutos': (time.time() - t_ep) / 60})
        if m_val_raras > mejor_map:
            mejor_map, mejor_epoca = m_val_raras, epoca
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

    minutos = (time.time() - t_inicio) / 60
    modelo.load_state_dict(mejor_estado)
    P_val, P_test = predecir(modelo, dl_val, dispositivo), predecir(modelo, dl_test, dispositivo)
    metricas, por_etiqueta = evaluar(P_val, Y_val, P_test, Y_test, etiquetas, raras)
    resultado = {'id': f'{arquitectura}_{perdida}_s{semilla}', 'arquitectura': arquitectura, 'perdida': perdida,
                 'semilla': semilla, 'epocas': epocas, 'mejor_epoca': mejor_epoca, 'map_val_raras': mejor_map,
                 'minutos_entrenamiento': minutos,
                 'parametros_M': sum(p.numel() for p in modelo.parameters()) / 1e6, **metricas}
    return resultado, registro, mejor_estado, P_val, P_test, por_etiqueta
