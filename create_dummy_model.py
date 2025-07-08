#!/usr/bin/env python3
"""
Script pour créer un modèle factice si le vrai modèle n'est pas disponible
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def create_dummy_model():
    """Crée un modèle factice pour les tests"""
    
    # Créer un modèle basé sur MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Geler les couches de base
    base_model.trainable = False
    
    # Ajouter les couches de classification
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(6, activation='softmax')  # 6 classes pour nos plats
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Créer des données factices pour l'entraînement
    x_dummy = np.random.random((10, 224, 224, 3))
    y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, 6, 10), 6)
    
    # Entraîner brièvement le modèle
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
    
    return model

if __name__ == "__main__":
    import os
    
    model_path = "best_food_model.h5"
    
    if not os.path.exists(model_path):
        print("Création d'un modèle factice...")
        model = create_dummy_model()
        model.save(model_path)
        print(f"Modèle factice sauvegardé dans {model_path}")
    else:
        print(f"Le modèle {model_path} existe déjà.")