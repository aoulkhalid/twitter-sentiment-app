import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')

    # Chemins vers les artefacts ML
    MODEL_PATH         = os.path.join(BASE_DIR, 'artifacts', 'model.pkl')
    LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'artifacts', 'label_encoder.pkl')
    PLOTS_DIR          = os.path.join(BASE_DIR, 'artifacts', 'plots')

    # Chemins vers les données
    TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'twitter_training.csv')
    TEST_CSV  = os.path.join(BASE_DIR, 'data', 'twitter_validation.csv')

    # Classes du modèle (mêmes que le notebook)
    LABEL_ORDER = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    PALETTE = {
        'Positive':   '#1976D2',
        'Negative':   '#D32F2F',
        'Neutral':    '#F57C00',
        'Irrelevant': '#616161',
    }

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'default':     DevelopmentConfig,
}
