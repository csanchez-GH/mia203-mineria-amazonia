"""Las cuatro funciones de pérdida comparadas: BCE, BCE ponderada, focal loss y Asymmetric Loss.

Todas reciben logits (salida de la red antes de la sigmoide) y etiquetas 0/1, y promedian sobre todas
las etiquetas de todas las imágenes del lote. Código idéntico al del bloque 4a del notebook.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1) BCE: entropía cruzada binaria (nuestro punto de partida) ---
# Fórmula, para cada etiqueta: L = -[ y·log(p) + (1-y)·log(1-p) ],  con p = sigmoide(logit)
# Trata a todas las etiquetas por igual. Como la mayoría de los ejemplos son negativos
# (por ejemplo, el 99,2 % de las imágenes NO tienen mina), la red puede aprender a decir
# casi siempre "no" y aun así tener una pérdida baja. Ese es el problema que queremos atacar.
class PerdidaBCE(nn.Module):
    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits.float(), y)

# --- 2) BCE ponderada: le damos más peso a los positivos de las etiquetas raras ---
# Fórmula: L = -[ w·y·log(p) + (1-y)·log(1-p) ],  con un peso w distinto para cada etiqueta.
# El peso "clásico" sería w = negativos / positivos, pero para blow_down eso da más de 400 y el
# entrenamiento se vuelve inestable. Por eso usamos la raíz cuadrada: w = sqrt(negativos / positivos).
# Así las raras pesan más, pero sin exagerar.
class PerdidaBCEPonderada(nn.Module):
    def __init__(self, Y_train):
        super().__init__()
        positivos = Y_train.sum(0)                          # cuántos positivos tiene cada etiqueta en train
        negativos = len(Y_train) - positivos
        peso = np.sqrt(negativos / np.maximum(positivos, 1))
        self.register_buffer('peso', torch.tensor(peso, dtype=torch.float32))  # viaja a la GPU con el modelo
    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=self.peso)

# --- 3) Focal loss (Lin et al., 2017): bajamos el peso de los ejemplos "fáciles" ---
# Fórmula: L = -α_t · (1 - p_t)^γ · log(p_t)
#   donde p_t = p si y = 1, y p_t = 1 - p si y = 0 (es la probabilidad de la respuesta correcta).
# La idea: si la red ya acierta con mucha seguridad (p_t cerca de 1), el factor (1 - p_t)^γ
# se vuelve casi 0 y ese ejemplo casi no aporta. Así la red se concentra en los ejemplos
# difíciles. Usamos los valores del paper original: γ = 2 y α = 0,25.
class PerdidaFocal(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, y):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')  # = -log(p_t)
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)
        perdida = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
            perdida = alpha_t * perdida
        return perdida.mean()

# --- 4) Asymmetric Loss, ASL (Ridnik et al., 2021): trata distinto a positivos y negativos ---
# Fórmulas:
#   para positivos (y = 1): L+ = -(1 - p)^γ+ · log(p)
#   para negativos (y = 0): L- = -(p_m)^γ- · log(1 - p_m),  con p_m = max(p - m, 0)
# Es la técnica más reciente que probamos y fue pensada justo para clasificación multietiqueta,
# donde los negativos son muchísimos más que los positivos. Hace dos cosas:
#   - Con γ- > γ+, "apaga" con más fuerza a los negativos fáciles que a los positivos.
#   - Con el margen m, si la red ya da una probabilidad muy baja a un negativo (menor que m),
#     ese negativo directamente deja de aportar a la pérdida.
# Usamos los valores recomendados por los autores: γ+ = 0, γ- = 4 y m = 0,05.
class PerdidaASL(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, margen=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos, self.gamma_neg, self.margen, self.eps = gamma_pos, gamma_neg, margen, eps
    def forward(self, logits, y):
        p = torch.sigmoid(logits.float())
        p_m = (p - self.margen).clamp(min=0)                # probabilidad "desplazada" para los negativos
        perdida_pos = y * (1 - p) ** self.gamma_pos * torch.log(p.clamp(min=self.eps))
        perdida_neg = (1 - y) * p_m ** self.gamma_neg * torch.log((1 - p_m).clamp(min=self.eps))
        return -(perdida_pos + perdida_neg).mean()


PERDIDAS = ['bce', 'bce_pond', 'focal', 'asl']


def crear_perdida(nombre, Y_train=None):
    """Crea la pérdida por su nombre. La BCE ponderada necesita las etiquetas de entrenamiento."""
    if nombre == 'bce':
        return PerdidaBCE()
    if nombre == 'bce_pond':
        if Y_train is None:
            raise ValueError('La BCE ponderada necesita Y_train para calcular sus pesos')
        return PerdidaBCEPonderada(Y_train)   # pesos calculados SOLO con train
    if nombre == 'focal':
        return PerdidaFocal()
    if nombre == 'asl':
        return PerdidaASL()
    raise ValueError(f'Pérdida desconocida: {nombre}')
