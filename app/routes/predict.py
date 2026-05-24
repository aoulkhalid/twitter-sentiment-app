from flask import Blueprint, render_template, request, jsonify

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    GET  → affiche le formulaire
    POST → reçoit le tweet, retourne la prédiction
    """
    from app import predictor

    result = None

    if request.method == 'POST':
        # Supporte JSON (API) et formulaire HTML
        if request.is_json:
            data       = request.get_json()
            tweet_text = data.get('tweet', '')
        else:
            tweet_text = request.form.get('tweet', '')

        result = predictor.predict(tweet_text)

        # Retourne JSON si appelé depuis l'API
        if request.is_json:
            return jsonify(result)

    return render_template('predict.html', result=result)


@predict_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Endpoint REST pur.
    Body JSON : { "tweet": "I love this product!" }
    """
    from app import predictor

    data       = request.get_json(force=True)
    tweet_text = data.get('tweet', '')

    if not tweet_text:
        return jsonify({'error': 'Champ "tweet" manquant'}), 400

    result = predictor.predict(tweet_text)
    return jsonify(result)
