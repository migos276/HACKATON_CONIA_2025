# ===== SUPPRESSION DES WARNINGS =====
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# ===== IMPORTS FLASK =====
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename

# ===== IMPORTS TENSORFLOW =====
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== IMPORTS STANDARD =====
import os
import base64
from io import BytesIO
from PIL import Image
import hashlib
import json
from datetime import datetime
import logging
import sqlite3
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION BASE DE DONNÉES =====
DATABASE_PATH = 'afrifood.db'

def init_database():
    """Initialise la base de données SQLite"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Table des utilisateurs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table des prédictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_predictions TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db_connection():
    """Context manager pour les connexions à la base de données"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ===== CLASSE IMAGEPREDICTOR =====
class ImagePredictor:
    def __init__(self, model_path, class_names):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier du modèle est introuvable : {model_path}")
        print(f"[INFO] Chargement du modèle depuis : {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = (224, 224)  # Taille standard pour MobileNetV2
        print(f"[INFO] Modèle chargé. Taille d'image attendue : {self.img_size}")

    def preprocess_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=self.img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return img_array_expanded
    
    def preprocess_image_from_pil(self, pil_image):
        """Prétraite une image PIL pour la prédiction"""
        img_resized = pil_image.resize(self.img_size)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = preprocess_input(img_array)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return img_array_expanded
    
    def predict_single_image(self, image_path):
        """Fait une prédiction sur une seule image."""
        preprocessed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(preprocessed_img, verbose=0)
        score = predictions[0] 
        
        predicted_class_index = np.argmax(score)
        predicted_class_name = self.class_names[predicted_class_index]
        confidence_score = float(np.max(score))
        
        return predicted_class_name, confidence_score, predictions[0]
    
    def predict_from_pil_image(self, pil_image):
        """Fait une prédiction sur une image PIL."""
        preprocessed_img = self.preprocess_image_from_pil(pil_image)
        predictions = self.model.predict(preprocessed_img, verbose=0)
        score = predictions[0] 
        
        predicted_class_index = np.argmax(score)
        predicted_class_name = self.class_names[predicted_class_index]
        confidence_score = float(np.max(score))
        
        return predicted_class_name, confidence_score, predictions[0]

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'best_food_model.h5'

# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Classes des plats
CLASS_NAMES = ['ekwang', 'eru', 'jollof-ghana', 'ndole','non-food', 'palm-nut-soup', 'waakye']

# Initialiser la base de données
init_database()

# Initialiser le prédicteur
predictor = None
if os.path.exists(MODEL_PATH):
    try:
        predictor = ImagePredictor(model_path=MODEL_PATH, class_names=CLASS_NAMES)
        print("ImagePredictor initialisé avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'initialisation du prédicteur: {e}")
else:
    print(f"Modèle non trouvé à {MODEL_PATH}. L'application fonctionnera sans prédiction.")

# Descriptions enrichies des plats traditionnels avec liens vidéo
plats_traditionnels = {
    'ekwang': {
        'nom': 'Ekwang',
        'nom_local': 'Ekwang (Cameroun)',
        'description': 'Plat traditionnel camerounais à base de tubercules de taro râpés et enveloppés dans des feuilles de taro, cuit avec de la viande et du poisson. Un délice ancestral qui unit les familles.',
        'origine': 'Cameroun',
        'region': 'Région du Sud-Ouest',
        'ingredients': ['Taro râpé', 'Feuilles de taro', 'Viande de bœuf', 'Poisson fumé', 'Crevettes séchées', 'Huile de palme', 'Épices locales'],
        'temps_preparation': '2-3 heures',
        'difficulte': 'Intermédiaire',
        'valeur_nutritive': 'Riche en fibres, vitamines et protéines',
        'histoire': 'Plat cérémoniel souvent préparé lors des grandes occasions familiales',
        'emoji': '🌿',
        'couleur': '#2D5016',
        'videos_preparation': [
            {
                'titre': 'Recette traditionnelle d\'Ekwang',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '15 min',
                'langue': 'Français'
            },
            {
                'titre': 'Ekwang authentique du Sud-Ouest',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '20 min',
                'langue': 'Anglais'
            },
            {
                'titre': 'Préparation moderne d\'Ekwang',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '12 min',
                'langue': 'Français'
            }
        ]
    },
    'eru': {
        'nom': 'Eru',
        'nom_local': 'Eru (Okok)',
        'description': 'Plat traditionnel camerounais préparé avec des feuilles d\'eru (légume vert sauvage), accompagné de viande et de poisson fumé. Un trésor nutritionnel de la forêt équatoriale.',
        'origine': 'Cameroun',
        'region': 'Région du Sud et Centre',
        'ingredients': ['Feuilles d\'eru fraîches', 'Viande de bœuf', 'Poisson fumé', 'Crevettes', 'Huile de palme', 'Stockfish', 'Épices traditionnelles'],
        'temps_preparation': '1-2 heures',
        'difficulte': 'Facile',
        'valeur_nutritive': 'Très riche en fer, calcium et vitamines',
        'histoire': 'Légume sauvage récolté dans les forêts, symbole de connexion avec la nature',
        'emoji': '🥬',
        'couleur': '#1B4332',
        'videos_preparation': [
            {
                'titre': 'Comment préparer l\'Eru traditionnel',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '18 min',
                'langue': 'Français'
            },
            {
                'titre': 'Eru avec stockfish - Recette complète',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '25 min',
                'langue': 'Anglais'
            },
            {
                'titre': 'Secrets de grand-mère pour l\'Eru',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '22 min',
                'langue': 'Français'
            }
        ]
    },
    'jollof-ghana': {
        'nom': 'Jollof Rice',
        'nom_local': 'Jollof Rice (Ghana)',
        'description': 'Riz parfumé cuit dans une sauce tomate épicée avec des légumes et de la viande, version ghanéenne du célèbre plat ouest-africain. Le roi des plats de fête!',
        'origine': 'Ghana',
        'region': 'Tout le Ghana',
        'ingredients': ['Riz jasmin', 'Tomates fraîches', 'Oignons', 'Viande de poulet', 'Épices ghanéennes', 'Légumes colorés', 'Bouillon de viande'],
        'temps_preparation': '45 minutes',
        'difficulte': 'Facile',
        'valeur_nutritive': 'Équilibré en glucides, protéines et légumes',
        'histoire': 'Plat de célébration, fierté culinaire ghanéenne dans la "guerre" du Jollof',
        'emoji': '🍚',
        'couleur': '#D2691E',
        'videos_preparation': [
            {
                'titre': 'Jollof Rice ghanéen authentique',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '30 min',
                'langue': 'Anglais'
            },
            {
                'titre': 'Le secret du parfait Jollof ghanéen',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '35 min',
                'langue': 'Français'
            },
            {
                'titre': 'Jollof Rice pour débutants',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '20 min',
                'langue': 'Anglais'
            }
        ]
    },
    'ndole': {
        'nom': 'Ndolé',
        'nom_local': 'Ndolé (Plat National)',
        'description': 'Plat national du Cameroun préparé avec des feuilles de ndolé amères, des arachides grillées, de la viande et du poisson. L\'âme culinaire du Cameroun.',
        'origine': 'Cameroun',
        'region': 'National (origine Douala)',
        'ingredients': ['Feuilles de ndolé', 'Arachides grillées', 'Viande de bœuf', 'Poisson fumé', 'Crevettes fraîches', 'Stockfish', 'Huile de palme'],
        'temps_preparation': '2-3 heures',
        'difficulte': 'Difficile',
        'valeur_nutritive': 'Très riche en protéines, lipides sains et minéraux',
        'histoire': 'Plat des grandes occasions, symbole de l\'hospitalité camerounaise',
        'emoji': '🥜',
        'couleur': '#8B4513',
        'videos_preparation': [
            {
                'titre': 'Ndolé traditionnel - Recette complète',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '45 min',
                'langue': 'Français'
            },
            {
                'titre': 'Maîtriser le Ndolé comme un chef',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '50 min',
                'langue': 'Anglais'
            },
            {
                'titre': 'Ndolé végétarien moderne',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '30 min',
                'langue': 'Français'
            }
        ]
    },
    'non-food': {
        'nom': 'non-trouve',
        'nom_local': 'non-trouve',
        'description': 'element non connu',
        'origine': 'RAS',
        'region': 'RAS',
        'ingredients': [],
        'temps_preparation': 'RAS',
        'difficulte': 'RAS',
        'valeur_nutritive': 'RAS',
        'histoire': 'RAS',
        'emoji': 'RAS',
        'couleur': '#FFFF',
        'videos_preparation': []
    },
    
    'palm-nut-soup': {
        'nom': 'Soupe de Noix de Palme',
        'nom_local': 'Palm Nut Soup',
        'description': 'Soupe traditionnelle ouest-africaine préparée avec l\'huile de palme rouge extraite des noix fraîches, de la viande et des légumes. Un concentré de saveurs ancestrales.',
        'origine': 'Afrique de l\'Ouest',
        'region': 'Ghana, Côte d\'Ivoire, Liberia',
        'ingredients': ['Noix de palme fraîches', 'Viande de chèvre', 'Poisson fumé', 'Légumes verts', 'Épices locales', 'Piment africain'],
        'temps_preparation': '3-4 heures',
        'difficulte': 'Difficile',
        'valeur_nutritive': 'Riche en vitamine A, antioxydants et acides gras essentiels',
        'histoire': 'Soupe sacrée dans certaines cultures, liée aux rituels de purification',
        'emoji': '🌴',
        'couleur': '#FF6B35',
        'videos_preparation': [
            {
                'titre': 'Palm Nut Soup authentique',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '40 min',
                'langue': 'Anglais'
            },
            {
                'titre': 'Extraction traditionnelle de l\'huile de palme',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '25 min',
                'langue': 'Français'
            },
            {
                'titre': 'Soupe de palme moderne et rapide',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '30 min',
                'langue': 'Anglais'
            }
        ]
    },
    'waakye': {
        'nom': 'Waakye',
        'nom_local': 'Waakye (Riz aux haricots)',
        'description': 'Plat ghanéen emblématique composé de riz et de haricots cuits ensemble avec des feuilles de millet, souvent servi avec diverses garnitures colorées. Le petit-déjeuner des champions!',
        'origine': 'Ghana',
        'region': 'Nord du Ghana (origine Hausa)',
        'ingredients': ['Riz local', 'Haricots noirs', 'Feuilles de millet séchées', 'Garnitures variées', 'Sauce tomate épicée', 'Œufs durs', 'Avocat'],
        'temps_preparation': '1 heure',
        'difficulte': 'Facile',
        'valeur_nutritive': 'Protéines complètes, fibres et glucides complexes',
        'histoire': 'Plat du petit-déjeuner devenu symbole de l\'identité ghanéenne urbaine',
        'emoji': '🍛',
        'couleur': '#8B0000',
        'videos_preparation': [
            {
                'titre': 'Waakye traditionnel ghanéen',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '35 min',
                'langue': 'Anglais'
            },
            {
                'titre': 'Waakye avec toutes les garnitures',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '28 min',
                'langue': 'Français'
            },
            {
                'titre': 'Secrets du Waakye parfait',
                'url': 'https://www.youtube.com/watch?v=wzYgtKvqXoM',
                'duree': '32 min',
                'langue': 'Anglais'
            }
        ]
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_by_email(email):
    """Récupère un utilisateur par son email"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        return cursor.fetchone()

def create_user(email, password):
    """Crée un nouvel utilisateur"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (email, password_hash) VALUES (?, ?)',
            (email, password_hash)
        )
        conn.commit()
        return cursor.lastrowid

def save_prediction(user_id, predicted_class, confidence, all_predictions):
    """Sauvegarde une prédiction en base de données"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO predictions (user_id, predicted_class, confidence, all_predictions) VALUES (?, ?, ?, ?)',
            (user_id, predicted_class, confidence, json.dumps(all_predictions.tolist()))
        )
        conn.commit()

def get_user_predictions(user_id):
    """Récupère les prédictions d'un utilisateur"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        )
        predictions = cursor.fetchall()
        
        result = []
        for pred in predictions:
            result.append({
                'timestamp': pred['created_at'],
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence'],
                'all_predictions': json.loads(pred['all_predictions'])
            })
        return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email et mot de passe requis'}), 400
        
        user = get_user_by_email(email)
        if user and user['password_hash'] == hashlib.sha256(password.encode()).hexdigest():
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            return redirect(url_for('dashboard'))
        else:
            return jsonify({'error': 'Identifiants incorrects'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email et mot de passe requis'}), 400
        
        if get_user_by_email(email):
            return jsonify({'error': 'Utilisateur déjà existant'}), 400
        
        user_id = create_user(email, password)
        session['user_id'] = user_id
        session['user_email'] = email
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', user_email=session['user_email'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Utilisateur non connecté'}), 401
    
    if predictor is None:
        return jsonify({'error': 'Modèle non disponible'}), 500
    
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Aucun fichier sélectionné'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                predicted_class, confidence, all_predictions = predictor.predict_single_image(filepath)
                os.remove(filepath)
                if(confidence<=0.70):
                    predicted_class='non-food'
        
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
            image_bytes = base64.b64decode(image_data.split(',')[1])
            pil_image = Image.open(BytesIO(image_bytes))
            pil_image = pil_image.convert('RGB')
            
            predicted_class, confidence, all_predictions = predictor.predict_from_pil_image(pil_image)
        
        else:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        # Sauvegarder la prédiction
        save_prediction(session['user_id'], predicted_class, confidence, all_predictions)
        
        # Obtenir les informations du plat
        plat_info = plats_traditionnels.get(predicted_class, {})
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'plat_info': plat_info,
            'all_predictions': {CLASS_NAMES[i]: float(all_predictions[i]) for i in range(len(CLASS_NAMES))}
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

@app.route('/plats-traditionnels')
def plats_traditionnels_route():
    return render_template('plats_traditionnels.html', plats=plats_traditionnels)

@app.route('/plat/<string:nom_plat>')
def detail_plat(nom_plat):
    if nom_plat not in plats_traditionnels:
        return jsonify({'error': 'Plat non trouvé'}), 404
    
    plat = plats_traditionnels[nom_plat]
    return render_template('detail_plat.html', plat=plat, nom_plat=nom_plat)

@app.route('/historique')
def historique():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    predictions = get_user_predictions(session['user_id'])
    return render_template('historique.html', predictions=predictions)

@app.route('/api/user-stats')
def user_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Utilisateur non connecté'}), 401
    
    predictions = get_user_predictions(session['user_id'])
    
    total_predictions = len(predictions)
    plats_predits = {}
    
    for pred in predictions:
        plat = pred['predicted_class']
        plats_predits[plat] = plats_predits.get(plat, 0) + 1
    
    # Récupérer la date de création du compte
    user = get_user_by_email(session['user_email'])
    
    return jsonify({
        'total_predictions': total_predictions,
        'plats_predits': plats_predits,
        'membre_depuis': user['created_at'] if user else 'Inconnue'
    })

@app.route('/health')
def health_check():
    """Endpoint de santé pour Docker"""
    return jsonify({'status': 'healthy', 'model_loaded': predictor is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)