import tensorflow as tf
from tensorflow import keras

# Chargement du modèle
model = keras.models.load_model("model_finetuned.h5")

# Résumé du modèle
model.summary()
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# Prédiction
##tetst de surete une image
import os
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

#  Dossier contenant les images à tester
test_images_dir = "test_surete"

class_names = ['ekwang', 'eru', 'jollof-ghana', 'ndole','non-food', 'palm-nut-soup', 'waakye'] # exemple

#  Récupère toutes les images dans le dossier
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Parcours et prédiction
for image_name in image_files:
    image_path = os.path.join(test_images_dir, image_name)

    # Chargement et prétraitement
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisation si ton modèle le nécessite
    # Prédiction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index]
    predicted_label = class_names[predicted_index]

    # Affichage
    plt.imshow(img)
    plt.title(f"{image_name}\nPrédiction : {predicted_label} ({confidence * 100:.1f}%)")
    plt.axis("off")
    plt.show()
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])  # si modèle de classification
#noms des classes
print("Classe prédite :",predicted_class)
class_names = ['ekwang', 'eru', 'jollof-ghana', 'ndole','non-food', 'palm-nut-soup', 'waakye'] # exemple
print("Classe prédite :", class_names[predicted_class])


