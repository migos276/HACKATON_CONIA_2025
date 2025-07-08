#!/bin/bash

# Script de démarrage pour l'application AfriFood AI

echo "🍽️ Démarrage d'AfriFood AI..."

# Créer un modèle factice si nécessaire
if [ ! -f "best_food_model.h5" ]; then
    echo "📦 Création d'un modèle factice..."
    python create_dummy_model.py
fi

# Créer la base de données si elle n'existe pas
if [ ! -f "afrifood.db" ]; then
    echo "🗄️ Initialisation de la base de données..."
    python -c "from app import init_database; init_database()"
fi

# Démarrer l'application
echo "🚀 Démarrage de l'application..."
python app.py