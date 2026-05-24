# 🚀 Twitter Sentiment Analysis App

This project is a Machine Learning web application that analyzes the sentiment of tweets and classifies them into **Positive**, **Negative**, or **Neutral**.

It includes a full pipeline: data preprocessing, model training, and a Flask web interface for predictions.

---

## 📌 Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Sentiment classification (Positive / Negative / Neutral)
- Flask web application for real-time predictions

---

## 🧠 Machine Learning Pipeline

1. Load dataset (Twitter Training & Validation data)
2. Text preprocessing (cleaning, tokenization, etc.)
3. Feature extraction (TF-IDF / vectorization)
4. Model training
5. Model evaluation
6. Save trained model
7. Deploy with Flask

---

## 🛠️ Tech Stack

- Python 🐍
- Flask 🌐
- Pandas
- NumPy
- Scikit-learn
- NLP techniques (TF-IDF, text cleaning)

---


## 📂 Project Structure

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
