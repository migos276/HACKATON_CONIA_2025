import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Chemins vers les dossiers
data_dir = Path("dataset_norm")
train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

# Paramètres
batch_size = 32
img_height = 224
img_width = 224
IMG_SHAPE = (img_height, img_width, 3)

# Chargement des datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
num_classes = len(class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Normalisation des images
def preprocess(ds):
    return ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = preprocess(train_ds)
val_ds = preprocess(val_ds)
test_ds = preprocess(test_ds)

# Optimisation des performances
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

### AJOUT : Couche d'augmentation de données ###
# Crée de nouvelles images à la volée pour enrichir le jeu de données
# et limiter le sur-apprentissage.
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# Modèle pré-entraîné
base_model = MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Étape 1 : on gèle le modèle de base
base_model.trainable = False

# Construction du modèle complet
model = Sequential([
    # On ajoute une couche d'input explicite pour que le modèle connaisse la taille d'entrée
    layers.Input(shape=IMG_SHAPE),
    ### AJOUT : Intégration de la couche d'augmentation de données ###
    # Cette couche ne sera active que pendant l'entraînement
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Afficher la structure du modèle pour vérifier
model.summary()

### AJOUT : Callbacks pour sauvegarder le meilleur modèle et arrêter l'entraînement si besoin ###
# Sauvegarde le modèle uniquement si la 'val_accuracy' s'améliore
checkpoint_cb = ModelCheckpoint(
    "best_model.h5",              # Nom du fichier pour le meilleur modèle
    save_best_only=True,          # Ne sauvegarde que le meilleur
    monitor='val_accuracy',       # Métrique à surveiller
    mode='max'                    # On veut maximiser cette métrique
)

# Arrête l'entraînement si la perte de validation ne s'améliore pas pendant 'patience' époques
early_stopping_cb = EarlyStopping(
    patience=5,                   # Nombre d'époques à attendre sans amélioration
    monitor='val_loss',           # Métrique à surveiller
    restore_best_weights=True     # Restaure les poids du meilleur modèle à la fin
)

# Entraînement initial (feature extraction)
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb, early_stopping_cb] # Utilisation des callbacks
)

# Phase fine-tuning
print("\n Dégel de la dernière partie du backbone pour fine-tuning…")

base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompiler avec un learning rate plus faible
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # learning rate réduit
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement supplémentaire
fine_tune_epochs = 10 # On peut se permettre un peu plus d'époques pour le fine-tuning
total_epochs = epochs + fine_tune_epochs

print(f"\nContinuer l'entraînement pour {fine_tune_epochs} époques de fine-tuning...")

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    # L'entraînement reprendra là où il s'est arrêté, donc on entraîne pour 'fine_tune_epochs' de plus
    epochs=fine_tune_epochs,
    callbacks=[checkpoint_cb, early_stopping_cb] # On réutilise les mêmes callbacks
)

### AJOUT : Chargement du meilleur modèle sauvegardé avant l'évaluation finale ###
print("\nChargement du meilleur modèle sauvegardé ('best_model.h5') pour l'évaluation finale...")
model = tf.keras.models.load_model("best_model.h5")

# Évaluation sur le jeu de test
loss, accuracy = model.evaluate(test_ds)
print(f"\nPrécision du MEILLEUR modèle sur le jeu de test : {accuracy:.2f}")

# Sauvegarde du modèle FINAL (le dernier état, pas forcément le meilleur)
model.save("model_finetuned_final_state.h5")
print("Modèle final sauvegardé sous model_finetuned_final_state.h5")

# Courbes
# On combine les historiques des deux phases d'entraînement
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# L'axe X s'adapte au nombre d'époques réellement effectuées
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Précision Entraînement')
plt.plot(epochs_range, val_acc, label='Précision Validation')
plt.legend()
plt.title('Précision')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perte Entraînement')
plt.plot(epochs_range, val_loss, label='Perte Validation')
plt.legend()
plt.title('Perte')
plt.show()
