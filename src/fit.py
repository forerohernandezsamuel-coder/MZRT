import cv2
import numpy as np
from box import Box
from train import *
import os
import pickle

# ==============================================================================
# CARGA GLOBAL ÚNICA DEL MODELO (Se ejecuta una sola vez al encender el servidor)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'trained_models', 'nn_trained_model_hog.sav')

if not os.path.exists(MODEL_PATH):
    print('Please wait while training the NN-HOG model....')
    # Nota: Asegúrate de tener importada la función train si se requiere aquí
    train('NN', 'hog', 'nn_trained_model_hog')

print("-> Loading ML Model into RAM memory once...")
GLOBAL_MODEL = pickle.load(open(MODEL_PATH, 'rb'))
# ==============================================================================


def predict(img):
    """
    Función optimizada: Realiza la predicción usando el modelo que ya reside 
    en la memoria RAM, eliminando las lecturas repetitivas de disco duro.
    """
    # Extraemos características de la imagen que entra
    features = extract_features(img, 'hog')
    
    # Predecimos usando la referencia global en memoria
    labels = GLOBAL_MODEL.predict([features])

    return labels


# if __name__ == "__main__":
#     img = cv2.imread('testresult/0_6.png')
#     labels = predict(img)
#     print(labels)
