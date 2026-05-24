"""
pipeline.py — Script d'entraînement autonome.
Lance ce fichier UNE FOIS pour créer artifacts/model.pkl.
Après ça, l'app Flask charge le modèle sans réentraîner.

Usage :
    cd twitter-sentiment-app/
    python -m app.ml.pipeline
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

from config import Config
from app.ml.preprocess import clean_tweet

# ── Constantes ────────────────────────────────────────────────────────
SEED        = 42
SCORING     = 'f1_macro'
LABEL_ORDER = Config.LABEL_ORDER
COLS        = ['tweet_id', 'entity', 'sentiment', 'tweet_content']

np.random.seed(SEED)


def load_data():
    """Charge et nettoie les CSV (Section 1 + 4 du notebook)."""
    print("[1/5] Chargement des données...")
    train = pd.read_csv(Config.TRAIN_CSV, header=None, names=COLS,
                        encoding='utf-8', on_bad_lines='skip')
    test  = pd.read_csv(Config.TEST_CSV,  header=None, names=COLS,
                        encoding='utf-8', on_bad_lines='skip')

    # Suppression des valeurs manquantes
    train.dropna(subset=['tweet_content', 'sentiment'], inplace=True)
    test.dropna(subset=['tweet_content', 'sentiment'],  inplace=True)

    # Filtrage des labels valides
    train = train[train['sentiment'].isin(LABEL_ORDER)]
    test  = test[test['sentiment'].isin(LABEL_ORDER)]

    print(f"   Train : {len(train):,} tweets | Test : {len(test):,} tweets")
    return train, test


def preprocess(train, test):
    """Nettoyage texte + encodage labels (Section 4 du notebook)."""
    print("[2/5] Prétraitement...")
    train['clean'] = train['tweet_content'].apply(clean_tweet)
    test['clean']  = test['tweet_content'].apply(clean_tweet)

    # Suppression des tweets vides après nettoyage
    train = train[train['clean'].str.strip() != '']
    test  = test[test['clean'].str.strip()  != '']

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    train['label'] = le.transform(train['sentiment'])
    test['label']  = le.transform(test['sentiment'])

    return train, test, le


def compare_models(X_train, y_train):
    """Validation croisée 5 plis sur 5 classifieurs (Section 5)."""
    print("[3/5] Comparaison des modèles (CV 5 plis)...")

    base_pipe = lambda clf: Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2)),
        ('clf', clf)
    ])

    candidates = {
        'Logistic Regression': base_pipe(LogisticRegression(max_iter=1000, random_state=SEED)),
        'Multinomial NB':      base_pipe(MultinomialNB()),
        'Complement NB':       base_pipe(ComplementNB()),
        'Linear SVC':          base_pipe(LinearSVC(random_state=SEED)),
        'SGD Classifier':      base_pipe(SGDClassifier(loss='modified_huber', random_state=SEED)),
    }

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, pipe in candidates.items():
        s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        scores[name] = s.mean()
        print(f"   {name:<25} F1-Macro={s.mean():.4f} ± {s.std():.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\n   ✔ Meilleur modèle : {best_name} (F1={scores[best_name]:.4f})")
    return best_name, candidates[best_name]


def tune(best_name, best_pipe, X_train, y_train):
    """HalvingGridSearchCV sur le meilleur modèle (Section 6)."""
    print("[4/5] Réglage des hyperparamètres...")

    param_grids = {
        'Logistic Regression': {
            'tfidf__max_features': [20_000, 50_000],
            'tfidf__ngram_range':  [(1, 1), (1, 2)],
            'clf__C':              [0.1, 1.0, 10.0],
            'clf__penalty':        ['l2'],
            'clf__solver':         ['saga'],
        },
        'Multinomial NB': {
            'tfidf__max_features': [20_000, 50_000],
            'tfidf__ngram_range':  [(1, 1), (1, 2)],
            'clf__alpha':          [0.1, 0.5, 1.0],
        },
        'Complement NB': {
            'tfidf__max_features': [20_000, 50_000],
            'tfidf__ngram_range':  [(1, 1), (1, 2)],
            'clf__alpha':          [0.1, 0.5, 1.0],
        },
        'Linear SVC': {
            'tfidf__max_features': [20_000, 50_000],
            'tfidf__ngram_range':  [(1, 1), (1, 2)],
            'clf__C':              [0.1, 1.0, 10.0],
        },
        'SGD Classifier': {
            'tfidf__max_features': [20_000, 50_000],
            'tfidf__ngram_range':  [(1, 1), (1, 2)],
            'clf__alpha':          [1e-4, 1e-3],
        },
    }

    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    gs = HalvingGridSearchCV(
        best_pipe, param_grids[best_name],
        cv=cv_tune, scoring=SCORING,
        factor=3, min_resources='exhaust',
        n_jobs=-1, refit=True,
        random_state=SEED, verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"   Best params : {gs.best_params_}")
    print(f"   Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_


def save_artifacts(model, le):
    """Sérialise model.pkl et label_encoder.pkl dans artifacts/."""
    print("[5/5] Sauvegarde des artefacts...")
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, Config.MODEL_PATH)
    joblib.dump(le,    Config.LABEL_ENCODER_PATH)
    print(f"   ✔ {Config.MODEL_PATH}")
    print(f"   ✔ {Config.LABEL_ENCODER_PATH}")


def main():
    print("=" * 55)
    print("  Pipeline d'entraînement — Twitter Sentiment")
    print("=" * 55)

    train, test     = load_data()
    train, test, le = preprocess(train, test)

    X_train, y_train = train['clean'].values, train['label'].values

    best_name, best_pipe = compare_models(X_train, y_train)
    final_model          = tune(best_name, best_pipe, X_train, y_train)

    save_artifacts(final_model, le)
    print("\nEntraînement terminé. Lance run.py pour démarrer l'app.")


if __name__ == '__main__':
    main()
