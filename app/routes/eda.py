from flask import Blueprint, render_template, current_app
import os, base64

eda_bp = Blueprint('eda', __name__)


def load_plot_b64(filename: str) -> str:
    """Charge un PNG pré-généré et le convertit en base64 pour l'embed HTML."""
    path = os.path.join(current_app.config['PLOTS_DIR'], filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ''


@eda_bp.route('/eda')
def eda():
    """
    Page EDA — charge les figures pré-générées depuis artifacts/plots/.
    Les plots ont été créés par app/ml/pipeline.py (ou le notebook).
    """
    plots = {
        'missing_bar':           load_plot_b64('01_missing_bar.png'),
        'null_heatmap':          load_plot_b64('02_null_heatmap.png'),
        'sentiment_distribution':load_plot_b64('03_sentiment_distribution.png'),
        'balance_heatmap':       load_plot_b64('04_balance_heatmap.png'),
        'tweet_length':          load_plot_b64('07_tweet_length.png'),
        'bigrams':               load_plot_b64('09_bigrams.png'),
        'unigrams':              load_plot_b64('10_unigrams.png'),
        'wordclouds':            load_plot_b64('11_wordclouds.png'),
    }

    return render_template('eda.html', plots=plots)
