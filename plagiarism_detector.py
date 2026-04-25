"""
=============================================================================
Plagiarism Detection System
=============================================================================
Uses TF-IDF (Term Frequency–Inverse Document Frequency) vectorization and
Cosine Similarity to identify potentially plagiarized document pairs.

Author  : Plagiarism Detection System
Libraries: scikit-learn, NLTK, pandas, matplotlib, seaborn
=============================================================================

Theory
------
TF-IDF (Term Frequency–Inverse Document Frequency)
    TF-IDF is a numerical statistic that reflects how important a word is to
    a document within a collection (corpus).

    - Term Frequency (TF): Measures how frequently a term appears in a
      document. TF(t, d) = (Number of times term t appears in d) / (Total
      number of terms in d)

    - Inverse Document Frequency (IDF): Measures how important a term is
      across the entire corpus. Common words get a lower IDF score.
      IDF(t) = log(N / (1 + df(t)))  where N = total documents,
                                           df(t) = docs containing t

    - TF-IDF(t, d) = TF(t, d) × IDF(t)

    The resulting value is high when a term is frequent in one document but
    rare across others — making it a strong discriminator.

Cosine Similarity
    Cosine similarity measures the cosine of the angle between two non-zero
    vectors in an inner product space.

    similarity = (A · B) / (||A|| × ||B||)

    A score of 1.0 means identical documents; 0.0 means no shared vocabulary.
    It is ideal for text comparison because it is length-independent — a
    short and long document with the same topic can still score highly.
"""

import re
import string
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available
# ---------------------------------------------------------------------------
def _download_nltk_data() -> None:
    """Download required NLTK corpora silently if not already present."""
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_download_nltk_data()


# ===========================================================================
# PlagiarismDetector
# ===========================================================================
class PlagiarismDetector:
    """
    A modular plagiarism detection system based on TF-IDF and Cosine
    Similarity.

    Parameters
    ----------
    threshold : float, optional (default=0.70)
        Similarity score (0–1) above which a document pair is flagged as
        potentially plagiarised.

    Example
    -------
    >>> detector = PlagiarismDetector(threshold=0.70)
    >>> sim_matrix, sim_df = detector.calculate_similarity(documents)
    >>> flagged = detector.detect_plagiarism(document_names, sim_matrix)
    """

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold
        self._vectorizer = TfidfVectorizer()
        self._stop_words = set(stopwords.words("english"))

    # ------------------------------------------------------------------
    # 1. Text Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """
        Clean and normalise a raw text string.

        Steps
        -----
        1. Lowercase the entire text.
        2. Remove punctuation characters.
        3. Tokenize into individual words.
        4. Remove English stopwords.
        5. Rejoin tokens into a single string.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Preprocessed text ready for vectorization.
        """
        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Step 3: Tokenize
        tokens = word_tokenize(text)

        # Step 4: Remove stopwords and single-character tokens
        tokens = [
            t for t in tokens
            if t not in self._stop_words and len(t) > 1
        ]

        # Step 5: Rejoin
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # 2. TF-IDF Vectorization & Cosine Similarity
    # ------------------------------------------------------------------
    def calculate_similarity(
        self, documents: list[str]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Vectorize preprocessed documents with TF-IDF and compute the
        pairwise cosine similarity matrix.

        Parameters
        ----------
        documents : list[str]
            List of raw (un-preprocessed) document strings.

        Returns
        -------
        similarity_matrix : np.ndarray
            N×N array of pairwise cosine similarity scores (0–1).
        similarity_df : pd.DataFrame
            Same matrix as a labelled DataFrame with percentage formatting
            for display purposes.
        """
        preprocessed = [self.preprocess(doc) for doc in documents]

        # Build TF-IDF matrix (rows = documents, columns = vocab terms)
        tfidf_matrix = self._vectorizer.fit_transform(preprocessed)

        # Pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return similarity_matrix

    def get_similarity_dataframe(
        self,
        similarity_matrix: np.ndarray,
        document_names: list[str],
    ) -> pd.DataFrame:
        """
        Wrap a raw similarity matrix in a labelled pandas DataFrame.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            N×N cosine similarity array.
        document_names : list[str]
            Names/labels for each document (used as row/column headers).

        Returns
        -------
        pd.DataFrame
            Labelled similarity matrix with values as percentages (0–100).
        """
        df = pd.DataFrame(
            similarity_matrix * 100,
            index=document_names,
            columns=document_names,
        )
        return df

    # ------------------------------------------------------------------
    # 3. Plagiarism Detection (Document Level)
    # ------------------------------------------------------------------
    def detect_plagiarism(
        self,
        document_names: list[str],
        similarity_matrix: np.ndarray,
    ) -> list[dict]:
        """
        Identify document pairs whose similarity exceeds the threshold.

        Parameters
        ----------
        document_names : list[str]
            Labels for each document.
        similarity_matrix : np.ndarray
            N×N cosine similarity array.

        Returns
        -------
        list[dict]
            Each dict contains keys:
                - 'doc1'       : Name of the first document.
                - 'doc2'       : Name of the second document.
                - 'similarity' : Similarity score as a percentage (float).
                - 'flagged'    : Always True (only flagged pairs returned).
        """
        flagged = []
        n = len(document_names)

        for i in range(n):
            for j in range(i + 1, n):
                score = similarity_matrix[i][j]
                if score >= self.threshold:
                    flagged.append(
                        {
                            "doc1":       document_names[i],
                            "doc2":       document_names[j],
                            "similarity": round(score * 100, 2),
                            "flagged":    True,
                        }
                    )

        # Sort descending by similarity
        flagged.sort(key=lambda x: x["similarity"], reverse=True)
        return flagged

    # ------------------------------------------------------------------
    # 4. Sentence-Level Comparison
    # ------------------------------------------------------------------
    def sentence_level_comparison(
        self,
        doc1: str,
        doc2: str,
        sentence_threshold: float = 0.60,
    ) -> pd.DataFrame:
        """
        Compare every sentence in doc1 against every sentence in doc2 and
        return pairs that exceed *sentence_threshold*.

        Parameters
        ----------
        doc1 : str
            Full text of the first document.
        doc2 : str
            Full text of the second document.
        sentence_threshold : float, optional (default=0.60)
            Cosine similarity cutoff for sentence-level flagging.

        Returns
        -------
        pd.DataFrame
            Columns: ['Sentence (Doc A)', 'Sentence (Doc B)',
                      'Similarity (%)']
            Sorted descending by similarity.
        """
        sents1 = sent_tokenize(doc1)
        sents2 = sent_tokenize(doc2)

        if not sents1 or not sents2:
            return pd.DataFrame()

        preprocessed1 = [self.preprocess(s) for s in sents1]
        preprocessed2 = [self.preprocess(s) for s in sents2]

        # Fit on all sentences combined so the vocabulary is shared
        all_sents = preprocessed1 + preprocessed2
        vectorizer = TfidfVectorizer()
        tfidf_all = vectorizer.fit_transform(all_sents)

        tfidf1 = tfidf_all[: len(sents1)]
        tfidf2 = tfidf_all[len(sents1):]

        sim_matrix = cosine_similarity(tfidf1, tfidf2)

        results = []
        for i, row in enumerate(sim_matrix):
            for j, score in enumerate(row):
                if score >= sentence_threshold:
                    results.append(
                        {
                            "Sentence (Doc A)": sents1[i].strip(),
                            "Sentence (Doc B)": sents2[j].strip(),
                            "Similarity (%)":   round(score * 100, 2),
                        }
                    )

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("Similarity (%)", ascending=False).reset_index(
                drop=True
            )
        return df

    # ------------------------------------------------------------------
    # 5. Heatmap Visualization
    # ------------------------------------------------------------------
    def plot_heatmap(
        self,
        similarity_matrix: np.ndarray,
        document_names: list[str],
        output_path: str = "similarity_heatmap.png",
        figsize: tuple[int, int] = (10, 8),
    ) -> None:
        """
        Generate and save a colour-coded heatmap of the similarity matrix.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            N×N cosine similarity array (values in [0, 1]).
        document_names : list[str]
            Labels used as tick labels.
        output_path : str, optional (default='similarity_heatmap.png')
            File path where the PNG will be saved.
        figsize : tuple[int, int], optional (default=(10, 8))
            Matplotlib figure size in inches.
        """
        sim_pct = similarity_matrix * 100  # convert to percentage

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        sns.heatmap(
            sim_pct,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            xticklabels=document_names,
            yticklabels=document_names,
            linewidths=0.5,
            linecolor="#2a2a4a",
            cbar_kws={"label": "Similarity (%)", "shrink": 0.8},
            ax=ax,
            vmin=0,
            vmax=100,
        )

        # Style title and labels
        ax.set_title(
            "📄 Document Similarity Matrix (TF-IDF + Cosine Similarity)",
            fontsize=14,
            fontweight="bold",
            color="white",
            pad=15,
        )
        ax.tick_params(colors="white", labelsize=9)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=30, ha="right", color="white"
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="white")

        # Colour bar styling
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

        # Threshold line annotation
        ax.annotate(
            f"[!] Threshold: {self.threshold * 100:.0f}%",
            xy=(0.01, 0.01),
            xycoords="axes fraction",
            fontsize=9,
            color="#ffd700",
            fontstyle="italic",
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"\n✅ Heatmap saved to: {output_path}")

    # ------------------------------------------------------------------
    # 6. Pretty-Print Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def print_similarity_table(
        similarity_df: pd.DataFrame,
    ) -> None:
        """Print the similarity matrix as a formatted percentage table."""
        print("\n" + "=" * 70)
        print("  DOCUMENT SIMILARITY MATRIX  (values in %)")
        print("=" * 70)
        formatted = similarity_df.applymap(lambda x: f"{x:.1f}%")
        print(formatted.to_string())
        print("=" * 70)

    @staticmethod
    def print_flagged_pairs(
        flagged: list[dict],
        threshold: float,
    ) -> None:
        """Print all flagged document pairs in a readable format."""
        print(f"\n{'=' * 70}")
        print(f"  [!] FLAGGED PAIRS  (threshold: {threshold * 100:.0f}%)")
        print(f"{'=' * 70}")

        if not flagged:
            print("  [OK] No document pairs exceed the plagiarism threshold.")
        else:
            for idx, pair in enumerate(flagged, start=1):
                bar_len = int(pair["similarity"] / 5)
                bar = "#" * bar_len + "." * (20 - bar_len)
                print(
                    f"  {idx}. {pair['doc1']}  <->  {pair['doc2']}\n"
                    f"     Similarity : {pair['similarity']:.2f}%  "
                    f"[{bar}]\n"
                    f"     Status     : *** POTENTIAL PLAGIARISM DETECTED ***\n"
                )
        print("=" * 70)

    @staticmethod
    def print_sentence_comparison(
        sentence_df: pd.DataFrame,
        doc1_name: str,
        doc2_name: str,
    ) -> None:
        """Print sentence-level comparison results."""
        print(f"\n{'=' * 70}")
        print(f"  SENTENCE-LEVEL COMPARISON: {doc1_name} vs {doc2_name}")
        print(f"{'=' * 70}")

        if sentence_df.empty:
            print("  No similar sentences found above the threshold.")
            return

        for _, row in sentence_df.iterrows():
            print(f"\n  Similarity: {row['Similarity (%)']:.1f}%")
            print(f"  >> Doc A: \"{row['Sentence (Doc A)'][:120]}\"")
            print(f"  >> Doc B: \"{row['Sentence (Doc B)'][:120]}\"")
            print(f"  {'~' * 65}")

        print("=" * 70)
