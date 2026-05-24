from flask import Blueprint, render_template, current_app
import os, json

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """
    Dashboard principal.
    Affiche : état du modèle, métriques résumées, liens vers les pages.
    """
    from app import predictor

    # Vérifie si le modèle est chargé
    model_ready = predictor.is_loaded

    # Charge les métriques pré-calculées si elles existent
    metrics_file = os.path.join(current_app.config['PLOTS_DIR'], 'metrics.json')
    metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)

    return render_template('index.html',
                           model_ready=model_ready,
                           metrics=metrics)


@main_bp.route('/about')
def about():
    return render_template('about.html')
