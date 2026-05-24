from flask import Flask
from config import config
from .ml.predict import ModelPredictor

# Instance globale du prédicteur (chargé une seule fois au démarrage)
predictor: ModelPredictor = None


def create_app(config_name: str = 'default') -> Flask:
    """
    Application Factory Pattern.
    Crée l'instance Flask, charge la config, enregistre les Blueprints
    et charge le modèle ML en mémoire une seule fois.
    """
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(config[config_name])

    # ── Chargement du modèle au démarrage ─────────────────────────────
    global predictor
    predictor = ModelPredictor(
        model_path=app.config['MODEL_PATH'],
        label_encoder_path=app.config['LABEL_ENCODER_PATH'],
    )

    # ── Enregistrement des Blueprints ──────────────────────────────────
    from .routes.main    import main_bp
    from .routes.predict import predict_bp
    from .routes.eda     import eda_bp
    from .routes.metrics import metrics_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(metrics_bp)

    return app
