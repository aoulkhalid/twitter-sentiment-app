import os
import joblib
import numpy as np
from typing import Optional
from .preprocess import clean_tweet


class ModelPredictor:
    """
    Charge le pipeline sklearn sérialisé (TF-IDF + classifieur)
    et expose une méthode predict() pour l'inférence en temps réel.

    Le modèle est chargé UNE SEULE FOIS au démarrage de l'app
    (via create_app) et réutilisé pour toutes les requêtes.
    """

    def __init__(self, model_path: str, label_encoder_path: str):
        self.model         = None
        self.label_encoder = None
        self.is_loaded     = False
        self._load(model_path, label_encoder_path)

    def _load(self, model_path: str, label_encoder_path: str) -> None:
        """Charge model.pkl et label_encoder.pkl depuis artifacts/."""
        if os.path.exists(model_path) and os.path.exists(label_encoder_path):
            self.model         = joblib.load(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.is_loaded     = True
            print(f"[ML] Modèle chargé : {model_path}")
        else:
            print("[ML] ATTENTION : model.pkl introuvable. Lance d'abord app/ml/pipeline.py")

    def predict(self, tweet_text: str) -> dict:
        """
        Prédit le sentiment d'un tweet.

        Retourne :
        {
            'label':        'Positive',          # classe prédite
            'confidence':   0.87,                # probabilité max
            'probabilities': {                   # toutes les classes
                'Positive': 0.87,
                'Negative': 0.05,
                'Neutral':  0.06,
                'Irrelevant': 0.02
            },
            'cleaned_text': 'nettoyé ...'
        }
        """
        if not self.is_loaded:
            return {'error': 'Modèle non chargé. Lance pipeline.py d\'abord.'}

        cleaned = clean_tweet(tweet_text)

        if not cleaned:
            return {'error': 'Tweet vide après nettoyage.'}

        # Prédiction
        label_idx   = self.model.predict([cleaned])[0]
        label_name  = self.label_encoder.inverse_transform([label_idx])[0]

        # Probabilités (si le classifieur le supporte)
        proba_dict = {}
        if hasattr(self.model, 'predict_proba'):
            probas     = self.model.predict_proba([cleaned])[0]
            classes    = self.label_encoder.inverse_transform(
                np.arange(len(probas))
            )
            proba_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probas)}
            confidence = float(probas.max())
        else:
            confidence = 1.0  # LinearSVC sans calibration

        return {
            'label':         label_name,
            'confidence':    round(confidence, 4),
            'probabilities': proba_dict,
            'cleaned_text':  cleaned,
        }
