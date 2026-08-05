import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab') # required for word tokenization
nltk.download('stopwords') # required for stop word list
nltk.download('wordnet') # required for word lemmatizer

class DataPreprocessor:
    """
        Clean and preprocess selected text columns in a DataFrame.

        The pipeline normalizes whitespace, removes unwanted characters,
        tokenizes text, removes stopwords, and lemmatizes words.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
        cols : list
            Text columns to preprocess.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with additional `<column>_clean` columns.
    """
    def __init__(self, df, cols: list):
        self.df = df
        self.cols = cols
        self.stop_words = set(stopwords.words("english"))
        self.df_clean = pd.DataFrame()
        self.word_lemmatizer = WordNetLemmatizer()

    def _normalize_whitespace(self, text):
        if pd.isna(text):
            return ""

        return re.sub(r"\s+", " ", str(text)).strip()

    def _clean_text(self, text):
        text = re.sub(r"<[^>]*>", " ", text)       # HTML
        text = re.sub(r"https?://\S+", " ", text)  # URLs
        text = re.sub(r"\S+@\S+", " ", text)       # Emails
        text = re.sub(r"[^a-zA-Z\s]", " ", text)   # Special characters

        return text.lower()

    def _tokenization(self, col):
        self.df_clean[f"{col}_clean"] = (
            self.df_clean[col].apply(word_tokenize)
        )

    def _remove_stopwords(self, col):
        self.df_clean[f"{col}_clean"] = (
            self.df_clean[f"{col}_clean"].apply(
                lambda words: [
                    word
                    for word in words
                    if word not in self.stop_words
                ]
            )
        )

    def _lemmatizer(self, col):
        self.df_clean[f"{col}_clean"] = (
            self.df_clean[f"{col}_clean"].apply(
                lambda words: [
                    self.word_lemmatizer.lemmatize(word)
                    for word in words
                ]
            )
        )

    def run_preprocessor(self):
        for col in self.cols:
            self.df_clean[col] = (
                self.df[col]
                .apply(self._normalize_whitespace)
                .apply(self._clean_text)
                .apply(self._normalize_whitespace)
            )

            self._tokenization(col)
            self._remove_stopwords(col)
            self._lemmatizer(col)

        return self.df_clean

    def combine_text(self):
        clean_cols = [f"{col}_clean" for col in self.cols]
    
        return self.df_clean[clean_cols].apply(
            lambda row: " ".join(
                word
                for tokens in row
                for word in tokens
            ),
            axis=1
        )
