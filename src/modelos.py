"""Modelos preentrenados (timm) adaptados a 17 etiquetas e imágenes de 160 píxeles."""
import timm

# ResNet-50 (CNN, ~23.5 M parámetros) y DeiT-Small (Vision Transformer, ~21.6 M parámetros),
# ambos preentrenados en ImageNet-1k
MODELOS = {'resnet50': 'resnet50.a1_in1k', 'vit': 'deit_small_patch16_224.fb_in1k'}


def crear_modelo(arquitectura, n_etiquetas=17, tam=160, preentrenado=True):
    """Carga el backbone y reemplaza la capa final por una de n_etiquetas salidas.

    En el ViT indicamos el tamaño de imagen: timm interpola los embeddings de posición
    (preparados para 224 píxeles) a la grilla de 10 x 10 parches de 160 píxeles.
    """
    extra = {'img_size': tam} if arquitectura == 'vit' else {}
    return timm.create_model(MODELOS[arquitectura], pretrained=preentrenado, num_classes=n_etiquetas, **extra)
