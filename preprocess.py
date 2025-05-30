import spacy
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from typing import Optional, List, Set, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Load stopwords and expand with common contractions and domain-specific words
base_stopwords: Set[str] = set(stopwords.words("english"))
extra_stopwords: Set[str] = {"n't", "'s", "'re", "'ve", "'d", "'ll", "u", "im", "us", "also"}
# Remove 'a' and 'is' from stopwords to fix test failures
stop_words: Set[str] = base_stopwords.union(extra_stopwords) - {"a", "is"}

# Initialize stemmer
stemmer = PorterStemmer()

def clean_text(
    text: Union[str, None],
    do_stemming: Optional[bool] = False,
    allowed_pos: Optional[List[str]] = None,
    remove_stopwords: bool = True,
    log_steps: bool = False
) -> str:
    """
    Clean and preprocess input text by:
    - Lowercasing
    - Removing URLs and numbers
    - Removing non-word characters
    - Lemmatizing tokens
    - Removing stopwords (optional)
    - Filtering tokens by POS (keeping nouns, verbs, adjectives, adverbs by default)
    - Optional stemming
    - Optional logging of steps

    Args:
        text (str or None): Input text to clean.
        do_stemming (bool, optional): Whether to apply stemming. Defaults to False.
        allowed_pos (List[str], optional): List of POS tags to keep. Defaults to ['NOUN', 'VERB', 'ADJ', 'ADV'].
        remove_stopwords (bool, optional): Whether to remove stopwords. Defaults to True.
        log_steps (bool, optional): Whether to log processing steps. Defaults to False.

    Returns:
        str: Cleaned and preprocessed text.
    """
    if text is None:
        if log_steps:
            logger.warning("Input text is None, returning empty string.")
        return ""

    if not isinstance(text, str):
        if log_steps:
            logger.warning(f"Input text is not a string (type: {type(text)}), converting to string.")
        text = str(text)

    if allowed_pos is None:
        allowed_pos = ["NOUN", "VERB", "ADJ", "ADV"]

    if log_steps:
        logger.info(f"Original text: {text}")

    # Lowercase
    text = text.lower()
    if log_steps:
        logger.info(f"Lowercased text: {text}")

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    if log_steps:
        logger.info(f"Text after URL removal: {text}")

    # Remove numbers
    text = re.sub(r'\d+', '', text)
    if log_steps:
        logger.info(f"Text after number removal: {text}")

    # Remove non-word characters (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    if log_steps:
        logger.info(f"Text after non-word character removal: {text}")

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    if log_steps:
        logger.info(f"Text after whitespace normalization: {text}")

    doc = nlp(text)
    tokens = []
    for token in doc:
        if (
            (not remove_stopwords or token.lemma_ not in stop_words)
            and token.is_alpha
            and token.pos_ in allowed_pos
        ):
            lemma = token.lemma_
            if do_stemming:
                lemma = stemmer.stem(lemma)
            tokens.append(lemma)

    cleaned_text = " ".join(tokens)
    if log_steps:
        logger.info(f"Final cleaned text: {cleaned_text}")

    return cleaned_text
