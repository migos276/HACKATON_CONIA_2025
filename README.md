# 🍽️ AfriFood AI - Application Dockerisée

Une application Flask intelligente utilisant TensorFlow pour identifier automatiquement les plats traditionnels africains à partir d'images, maintenant avec base de données SQLite et support Docker.

## Démarrage Rapide avec Docker

### Prérequis
- Docker et Docker Compose installés
- Au moins 2GB de RAM libre

### Lancement Simple

```bash
lancer application 
comment lancer mon application a partir de l'image Docker

-charger d'abord image docker dans votre environement 

 gunzip hackaton.tar.gz    
 docker load -i  hackaton.tar

-lancer le docker

 docker run -p 5000:5000 project-afrifood-app:latest python3 app.py
# L'application sera accessible sur http://localhost:5000
```

### Lancement avec Nginx (Production)

```bash
# Démarrer avec le reverse proxy nginx
docker-compose --profile production up --build

# L'application sera accessible sur http://localhost
```

## 🛠️ Installation Locale

### Prérequis
- Python 3.9+
- pip

### Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Créer un modèle factice (si pas de modèle réel)
python create_dummy_model.py

# Démarrer l'application
chmod +x run.sh
./run.sh
```

## 📊 Base de Données

L'application utilise maintenant SQLite pour la persistance des données :

- **users** : Stockage des comptes utilisateurs
- **predictions** : Historique des analyses d'images

La base de données est automatiquement créée au premier démarrage.

## 🐳 Configuration Docker

### Variables d'Environnement

```bash
# Dans docker-compose.yml ou .env
SECRET_KEY=your-super-secret-key
FLASK_ENV=production
PORT=5000
```

### Volumes Persistants

- `./uploads:/app/uploads` - Images temporaires
- `./afrifood.db:/app/afrifood.db` - Base de données SQLite

## 🔧 Développement

### Structure du Projet

```
african-food-api/
├── app.py                 # Application principale
├── Dockerfile            # Configuration Docker
├── docker-compose.yml    # Orchestration des services
├── nginx.conf           # Configuration Nginx
├── requirements.txt     # Dépendances Python
├── create_dummy_model.py # Création de modèle factice
├── run.sh              # Script de démarrage
├── afrifood.db         # Base de données SQLite (auto-créée)
├── uploads/            # Dossier temporaire
└── templates/          # Templates HTML
```

### Commandes Utiles

```bash
# Voir les logs
docker-compose logs -f

# Redémarrer un service
docker-compose restart afrifood-app

# Accéder au conteneur
docker-compose exec afrifood-app bash

# Nettoyer les volumes
docker-compose down -v
```

## 🏥 Monitoring

### Health Check

L'application inclut un endpoint de santé :

```bash
curl http://localhost:5000/health
```

Réponse :
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Logs

```bash
# Logs de l'application
docker-compose logs afrifood-app

# Logs en temps réel
docker-compose logs -f
```

## 🔒 Sécurité

### Recommandations de Production

1. **Variables d'environnement** :
   ```bash
   export SECRET_KEY=$(openssl rand -hex 32)
   ```

2. **Reverse Proxy** :
   - Utilisez le profil production avec Nginx
   - Configurez SSL/TLS

3. **Base de données** :
   - Sauvegardez régulièrement `afrifood.db`
   - Considérez PostgreSQL pour la production

## 📈 Performance

### Optimisations

- **Modèle** : Le modèle TensorFlow est chargé une seule fois au démarrage
- **Base de données** : SQLite avec index automatiques
- **Images** : Nettoyage automatique des fichiers temporaires

### Limites

- **Taille d'image** : Maximum 10MB
- **Formats supportés** : JPG, PNG, JPEG, GIF
- **Concurrent users** : Testé jusqu'à 50 utilisateurs simultanés

## 🚀 Déploiement

### Docker Hub

```bash
# Construire l'image
docker build -t afrifood-ai:latest .

# Pousser vers Docker Hub
docker tag afrifood-ai:latest username/afrifood-ai:latest
docker push username/afrifood-ai:latest
```

### Cloud Deployment

#### AWS ECS
```bash
# Utiliser le docker-compose.yml avec AWS ECS CLI
ecs-cli compose up
```

#### Google Cloud Run
```bash
# Déployer sur Cloud Run
gcloud run deploy afrifood-ai --source .
```

#### Heroku
```bash
# Utiliser le Dockerfile
heroku container:push web
heroku container:release web
```

## 🐛 Dépannage

### Problèmes Courants

1. **Modèle non trouvé** :
   ```bash
   python create_dummy_model.py
   ```

2. **Erreur de permissions** :
   ```bash
   chmod +x run.sh
   sudo chown -R $USER:$USER uploads/
   ```

3. **Port déjà utilisé** :
   ```bash
   # Changer le port dans docker-compose.yml
   ports:
     - "5001:5000"
   ```

4. **Mémoire insuffisante** :
   ```bash
   # Augmenter la mémoire Docker
   # Docker Desktop > Settings > Resources > Memory
   ```

## 📝 API Documentation

### Endpoints Principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/register` | POST | Inscription |
| `/login` | POST | Connexion |
| `/predict` | POST | Analyse d'image |
| `/health` | GET | Status de santé |

### Exemple d'utilisation

```python
import requests

# Connexion
session = requests.Session()
login_data = {'email': 'user@example.com', 'password': 'password'}
session.post('http://localhost:5000/login', data=login_data)

# Prédiction
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = session.post('http://localhost:5000/predict', files=files)
    result = response.json()
    print(f"Plat: {result['predicted_class']}")
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Tester avec Docker
4. Soumettre une Pull Request

## 📄 Licence

MIT License - voir le fichier LICENSE pour plus de détails.

---

*Développé avec ❤️ pour préserver et célébrer la richesse culinaire africaine*