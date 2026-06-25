import cv2
import numpy as np
from box import Box
from train import *
import os
import pickle

def predict(img):
    # 1. Obtenemos la ruta absoluta de la carpeta donde vive este archivo fit.py (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construimos la ruta absoluta apuntando directamente al modelo .sav
    model_path = os.path.join(current_dir, 'trained_models', 'nn_trained_model_hog.sav')

    # 3. Validamos la existencia usando la ruta absoluta real
    if not os.path.exists(model_path):
        print('Please wait while training the NN-HOG model....')
        train('NN', 'hog', 'nn_trained_model_hog')

    # 4. Cargamos el modelo usando la ruta absoluta garantizada
    model = pickle.load(open(model_path, 'rb'))
    features = extract_features(img, 'hog')
    labels = model.predict([features])

    return labels


# if __name__ == "__main__":
#     img = cv2.imread('testresult/0_6.png')
#     labels = predict(img)
#     print(labels)
