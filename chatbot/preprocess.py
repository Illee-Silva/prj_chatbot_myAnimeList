"""Pré-processamento textual usando NLTK e spaCy."""

from __future__ import annotations

import re
from typing import List

import nltk
import spacy
from nltk.corpus import stopwords


class TextPreprocessor:
    """Normaliza texto para etapa de vetorização com TF-IDF."""

    def __init__(self) -> None:
        # Baixa recursos essenciais do NLTK em runtime local.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

        self.stop_words = set(stopwords.words("english"))
        self.nlp = self._load_spacy_model()

    def _load_spacy_model(self):
        """Tenta carregar modelo do spaCy; usa fallback se não existir."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Fallback para manter aplicação executável sem bloquear.
            return spacy.blank("en")

    def normalize_text(self, text: str) -> str:
        """Limpa, tokeniza, remove stopwords e lematiza quando possível."""
        # Remove URLs, caracteres especiais e excesso de espaços.
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()

        # Tokenização inicial com NLTK (com fallback para ambientes restritos).
        try:
            nltk_tokens = nltk.word_tokenize(text)
        except LookupError:
            nltk_tokens = text.split()
        nltk_tokens = [tk for tk in nltk_tokens if tk not in self.stop_words and len(tk) > 2]

        # Lematização e filtragem linguística com spaCy.
        doc = self.nlp(" ".join(nltk_tokens))
        processed_tokens: List[str] = []

        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue

            lemma = token.lemma_.strip().lower() if token.lemma_ else token.text.lower()
            if lemma and lemma not in self.stop_words and len(lemma) > 2:
                processed_tokens.append(lemma)

        # Caso o modelo fallback não tenha lematização robusta.
        if not processed_tokens:
            processed_tokens = nltk_tokens

        return " ".join(processed_tokens)
