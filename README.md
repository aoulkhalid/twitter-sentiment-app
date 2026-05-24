# 🚀 Application de Analyse de Sentiments Twitter

Ce projet est une application de Machine Learning qui analyse les sentiments des tweets et les classe en trois catégories :
**Positif**, **Négatif** ou **Neutre**.

Le projet inclut un pipeline complet : prétraitement des données, entraînement du modèle et interface web avec Flask.

---

## 📌 Fonctionnalités

- Nettoyage et prétraitement des données
- Analyse exploratoire des données (EDA)
- Entraînement d’un modèle de Machine Learning
- Classification des sentiments (Positif / Négatif / Neutre)
- Interface web avec Flask pour les prédictions en temps réel

---

## 🧠 Pipeline de Machine Learning

1. Chargement des datasets (Twitter Training & Validation)
2. Prétraitement du texte (nettoyage, tokenisation, etc.)
3. Extraction des caractéristiques (TF-IDF / vectorisation)
4. Entraînement du modèle
5. Évaluation du modèle
6. Sauvegarde du modèle entraîné
7. Déploiement avec Flask

---

## 🛠️ Technologies utilisées

- Python 🐍
- Flask 🌐
- Pandas
- NumPy
- Scikit-learn
- NLP (TF-IDF, traitement de texte)

---

## 📂 Structure du projet

twitter-sentiment-app/
│
├── app/
│ ├── ml/
│ │ └── pipeline.py
│ ├── init.py
│ └── routes.py
│
├── data/
│ ├── twitter_training.csv
│ └── twitter_validation.csv
│
├── artifacts/
│ └── (modèles entraînés)
│
├── run.py
├── config.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation et exécution

### 1. Cloner le projet

```bash
git clone https://github.com/aoulkhalid/twitter-sentiment-app.git
cd twitter-sentiment-app
2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate
3. Installer les dépendances
pip install -r requirements.txt
4. Lancer l’entraînement du modèle
python -m app.ml.pipeline
5. Lancer l’application Flask
python run.py

Puis ouvrir dans le navigateur :

http://localhost:5000
📊 Dataset

Le dataset contient des tweets annotés pour l’analyse de sentiments :

Twitter Training Dataset
Twitter Validation Dataset

(Source : Kaggle – Twitter Sentiment Analysis)

🚀 Améliorations futures
Amélioration de la précision avec Deep Learning (LSTM / Transformers)
Déploiement sur le cloud (AWS / Render / Heroku)
Ajout d’une API REST
Scraping Twitter en temps réel
👨‍💻 Auteur
Khalid Ela
