import re
import string


def clean_tweet(text: str) -> str:
    """
    Nettoie un tweet brut.
    Pipeline identique à la Section 4 du notebook :
      1. Minuscules
      2. Suppression des URLs
      3. Suppression des mentions (@user)
      4. Suppression de la ponctuation
      5. Suppression des espaces multiples
    """
    if not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs
    text = re.sub(r'@\w+', '', text)                    # Mentions
    text = re.sub(r'#\w+', '', text)                    # Hashtags (optionnel)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    return text
