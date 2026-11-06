"""Grad-CAM (Selvaraju et al., 2017) para ResNet-50 y para el ViT (DeiT-Small)."""
import torch
import torch.nn.functional as F


class GradCAM:
    """Mapa de las zonas que más hacen subir una salida del modelo.

    - ResNet-50: usar capa=modelo.layer4[-1] (grilla de 5 x 5 con imágenes de 160 px).
    - ViT: usar capa=modelo.blocks[-1].norm1 y es_vit=True (los 100 parches se reordenan en 10 x 10).
    """

    def __init__(self, modelo, capa, es_vit=False, tam=160):
        self.modelo, self.es_vit, self.tam = modelo, es_vit, tam
        self.act, self.grad = None, None
        capa.register_forward_hook(self._guardar)

    def _guardar(self, modulo, entrada, salida):
        self.act = salida
        # Solo registramos el gradiente si la pasada hacia adelante lo calcula (no dentro de torch.no_grad)
        if salida.requires_grad:
            salida.register_hook(lambda g: setattr(self, 'grad', g))

    def __call__(self, x, clase):
        self.modelo.zero_grad()
        logits = self.modelo(x)
        logits[0, clase].backward()
        A, G = self.act, self.grad
        if self.es_vit:
            n_pref = self.modelo.num_prefix_tokens          # token de clase
            A, G = A[:, n_pref:], G[:, n_pref:]
            lado = int(A.shape[1] ** 0.5)
            A = A.reshape(1, lado, lado, -1).permute(0, 3, 1, 2)
            G = G.reshape(1, lado, lado, -1).permute(0, 3, 1, 2)
        pesos = G.mean(dim=(2, 3), keepdim=True)            # importancia de cada mapa
        cam = F.relu((pesos * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(self.tam, self.tam), mode='bilinear', align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), torch.sigmoid(logits[0, clase]).item()
