from flask import Blueprint, render_template, current_app
import os, base64

metrics_bp = Blueprint('metrics', __name__)


def load_plot_b64(filename: str) -> str:
    path = os.path.join(current_app.config['PLOTS_DIR'], filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ''


@metrics_bp.route('/metrics')
def metrics():
    """Page métriques du modèle final (Section 8 du notebook)."""
    plots = {
        'model_comparison':  load_plot_b64('13_model_comparison.png'),
        'cv_heatmap':        load_plot_b64('14_cv_heatmap.png'),
        'gridsearch_heatmap':load_plot_b64('12_gridsearch_heatmap.png'),
        'learning_curve':    load_plot_b64('13_learning_curve.png'),
        'confusion_matrix':  load_plot_b64('14_confusion_matrix.png'),
        'per_class_metrics': load_plot_b64('15_per_class_metrics.png'),
        'roc_curves':        load_plot_b64('16_roc_curves.png'),
        'pr_curves':         load_plot_b64('17_pr_curves.png'),
        'feature_importance':load_plot_b64('18_feature_importance.png'),
    }

    return render_template('metrics.html', plots=plots)
