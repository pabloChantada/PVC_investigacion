# %%
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import cv2
import numpy as np

def apply_gabor_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=8.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        kernels.append(kernel)
    features = []
    for kernel in kernels:
        fimg = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.append(fimg.flatten())
    # Apilar características a lo largo de las columnas
    features = np.column_stack(features)
    return features  # Forma: (65536, 4)


def extract_edge_features(img):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convertir a uint8
    gray_uint8 = (gray * 255).astype(np.uint8)
    
    # Sobel en la imagen original en float32
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2).flatten()
    
    # Laplacian en la imagen uint8
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F).flatten()
    
    return np.column_stack((sobel_combined, laplacian))


def calculate_ndvi(img):
    # Convertir a float para evitar errores de división
    img = img.astype(float) + 1e-6
    # Asumiendo que el canal NIR es el canal R (ajustar según tus datos)
    nir = img[:, :, 2]  # Canal R
    red = img[:, :, 0]  # Canal B
    ndvi = (nir - red) / (nir + red)
    return ndvi.flatten()

def extract_features(img):
    # Intensidad RGB
    r = img[:, :, 2].flatten()  # Canal R
    g = img[:, :, 1].flatten()  # Canal G
    b = img[:, :, 0].flatten()  # Canal B
    
    # Filtros de Gabor
    gabor_feats = apply_gabor_filter(img)
    
    # Bordes
    edge_feats = extract_edge_features(img)
    
    # NDVI
    ndvi = calculate_ndvi(img).flatten()
    
    # Concatenar todas las características
    features = np.column_stack((
        r, g, b,                # (65536, 3)
        edge_feats,             # (65536, 2)
        gabor_feats,            # (65536, 4)
        ndvi                    # (65536,)
    ))
    
    return features  # Forma final: (65536, 10)


# Asegurarse de que las dimensiones sean correctas
assert X.shape[0] == y.shape[0], "El número de imágenes y máscaras no coincide."

# Inicializar listas
features_list = []
labels_list = []

for img, mask in zip(X, y):
    # Extraer características
    features = extract_features(img)
    features_list.append(features)
    # Aplanar la máscara
    labels = mask.flatten()
    labels_list.append(labels)

# Convertir listas a arrays
X_features = np.vstack(features_list)
y_labels = np.hstack(labels_list)

print(f"Características extraídas: {X_features.shape}")
print(f"Etiquetas extraídas: {y_labels.shape}")

def preprocess_image(img_path, img_size=(256, 256)):
    """
    Preprocesa la imagen con técnicas adicionales
    """
    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    
    # Redimensionar
    img_resized = cv2.resize(img, img_size)
    
    # Normalización de contraste
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Reducción de ruido
    img_denoised = cv2.fastNlMeansDenoisingColored(img_enhanced)
    
    # Normalización
    img_normalized = img_denoised.astype('float32') / 255.0
    
    return img_normalized

def postprocess_mask(mask, threshold=0.5, min_area=100):
    """
    Postprocesa la máscara predicha para mejorar la calidad
    """
    # Binarización
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Operaciones morfológicas
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Eliminar componentes pequeños
    num_labels, labels = cv2.connectedComponents(mask_cleaned)
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        if area < min_area:
            mask_cleaned[labels == label] = 0
    
    return mask_cleaned