# PlagiaScope — Concepts & Theory

> A comprehensive reference covering every theoretical idea behind the
> TF-IDF + Cosine Similarity plagiarism detection system.

---

## Table of Contents

1. [What is Plagiarism Detection?](#1-what-is-plagiarism-detection)
2. [Natural Language Processing (NLP) Pipeline](#2-natural-language-processing-nlp-pipeline)
3. [Text Preprocessing](#3-text-preprocessing)
4. [Bag of Words (BoW) Model](#4-bag-of-words-bow-model)
5. [TF-IDF — Term Frequency–Inverse Document Frequency](#5-tf-idf--term-frequencyinverse-document-frequency)
6. [Vector Space Model](#6-vector-space-model)
7. [Cosine Similarity](#7-cosine-similarity)
8. [Pairwise Document Comparison](#8-pairwise-document-comparison)
9. [Sentence-Level Analysis](#9-sentence-level-analysis)
10. [Similarity Threshold & Flagging](#10-similarity-threshold--flagging)
11. [Heatmap Visualization](#11-heatmap-visualization)
12. [System Architecture](#12-system-architecture)
13. [Limitations & Future Improvements](#13-limitations--future-improvements)
14. [Glossary](#14-glossary)

---

## 1. What is Plagiarism Detection?

**Plagiarism** is the act of using someone else's words, ideas, or work without
proper attribution. In academic and professional contexts it is a serious
ethical and legal violation.

**Automated plagiarism detection** is the process of algorithmically
identifying textual similarity between documents. Rather than reading every
pair of documents manually, a system converts text into mathematical
representations and measures their distance or similarity in a high-dimensional
space.

### Types of Plagiarism

| Type | Description | Detectable by TF-IDF? |
|---|---|---|
| **Verbatim / Copy-paste** | Exact copying of text | ✅ Very easily |
| **Near-verbatim** | Minor word substitutions (synonyms) | ✅ High score |
| **Paraphrase** | Ideas rewritten in different words | ⚠ Partially |
| **Idea / Structural** | Same structure, completely different words | ❌ Needs semantic models |
| **Cross-language** | Translated from another language | ❌ Needs translation layer |

> **This system targets verbatim and near-verbatim plagiarism**, which
> constitute the majority of academic dishonesty cases.

---

## 2. Natural Language Processing (NLP) Pipeline

The system follows a standard NLP pipeline:

```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│  1. Text Preprocessing          │  (lowercase, punctuation, stopwords)
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  2. Tokenization                │  (split into individual words)
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  3. Vocabulary Construction     │  (unique terms across all documents)
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  4. TF-IDF Vectorization        │  (convert each doc to a numeric vector)
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  5. Cosine Similarity           │  (measure angle between vectors)
└─────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────┐
│  6. Threshold Flagging          │  (flag pairs above similarity cutoff)
└─────────────────────────────────┘
   │
   ▼
Results + Heatmap
```

---

## 3. Text Preprocessing

Raw text must be cleaned before meaningful comparison. Without preprocessing,
trivial differences (capitalisation, punctuation) would affect the similarity
score unfairly.

### 3.1 Lowercasing

All text is converted to lowercase so that `"Machine"`, `"machine"`, and
`"MACHINE"` are treated as the same token.

```
Input : "Machine Learning is GREAT!"
Output: "machine learning is great!"
```

### 3.2 Punctuation Removal

Punctuation marks (`.,!?;:'"()[]{}`) carry no semantic meaning for similarity
and are stripped.

```
Input : "machine learning is great!"
Output: "machine learning is great"
```

### 3.3 Tokenization

The cleaned string is split into individual **tokens** (words).

```
Input : "machine learning is great"
Tokens: ["machine", "learning", "is", "great"]
```

> **Library used:** `nltk.tokenize.word_tokenize`

### 3.4 Stopword Removal

**Stopwords** are extremely common words that appear in almost every document
and therefore carry no discriminative power.

Examples: `a, an, the, is, are, was, were, of, in, to, for, with, on, at ...`

Removing them reduces noise and focuses the model on meaningful content words.

```
Tokens before: ["machine", "learning", "is", "great"]
Tokens after : ["machine", "learning", "great"]
```

> **Library used:** `nltk.corpus.stopwords` (English language list of ~179 words)

### 3.5 Preprocessing Example

| Step | Text |
|---|---|
| **Original** | "Machine learning is a subset of Artificial Intelligence." |
| After lowercase | "machine learning is a subset of artificial intelligence." |
| After punctuation removal | "machine learning is a subset of artificial intelligence" |
| After tokenization | `["machine","learning","is","a","subset","of","artificial","intelligence"]` |
| After stopword removal | `["machine","learning","subset","artificial","intelligence"]` |
| **Final (rejoined)** | `"machine learning subset artificial intelligence"` |

---

## 4. Bag of Words (BoW) Model

Before TF-IDF, the simpler **Bag of Words** model is worth understanding as
the foundation.

In BoW, each document is represented as a vector where each dimension
corresponds to a word in the shared vocabulary, and the value is the **raw
count** of that word in the document.

### Example

Vocabulary: `[cat, dog, fish, runs, swims]`

| Document | cat | dog | fish | runs | swims |
|---|---|---|---|---|---|
| "The cat runs" | 1 | 0 | 0 | 1 | 0 |
| "The dog swims" | 0 | 1 | 0 | 0 | 1 |
| "The fish swims" | 0 | 0 | 1 | 0 | 1 |

**Problem with BoW:** Common words like "the" would dominate the counts.
TF-IDF solves this by weighting terms by their rarity across the corpus.

---

## 5. TF-IDF — Term Frequency–Inverse Document Frequency

TF-IDF is the core feature-engineering technique used in this system.
It assigns a numerical weight to each word in each document such that:

- **High weight** → word appears frequently in *this* document but rarely
  across *other* documents (unique, discriminative word)
- **Low weight** → word is common across all documents (not useful for
  differentiation)

### 5.1 Term Frequency (TF)

Measures how often a term `t` appears in document `d`, normalised by the
document length to avoid bias towards longer documents.

```
           Number of times t appears in d
TF(t, d) = ─────────────────────────────────
              Total number of terms in d
```

**Example:** Document = "the cat sat on the cat mat" (7 words)

| Term | Count | TF |
|---|---|---|
| the | 2 | 2/7 = 0.286 |
| cat | 2 | 2/7 = 0.286 |
| sat | 1 | 1/7 = 0.143 |
| mat | 1 | 1/7 = 0.143 |

### 5.2 Inverse Document Frequency (IDF)

Measures how **rare** a term is across the entire corpus of N documents.
Terms that appear in many documents get a low IDF; rare terms get a high IDF.

```
                  N
IDF(t) = log ─────────────
              1 + df(t)

where:
  N     = total number of documents in the corpus
  df(t) = number of documents that contain term t
  +1    = smoothing to prevent division by zero
```

> **Note:** scikit-learn uses a slightly modified formula with `log((1+N)/(1+df(t))) + 1`
> to ensure all IDF values are ≥ 1 and to handle edge cases.

**Example:** Corpus of 5 documents

| Term | df(t) | IDF = log(5 / (1+df)) |
|---|---|---|
| "the" (appears in all 5) | 5 | log(5/6) ≈ **-0.18** (very low) |
| "learning" (appears in 3) | 3 | log(5/4) ≈ **0.22** |
| "eigenvalue" (appears in 1) | 1 | log(5/2) ≈ **0.92** (high) |

### 5.3 TF-IDF Score

The final TF-IDF weight for term `t` in document `d`:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

A word that is **frequent in this document** AND **rare across all documents**
gets the highest TF-IDF score — it is the most distinctive word for that document.

### 5.4 Complete TF-IDF Example

**Corpus (2 documents after preprocessing):**
- Doc 1: `"machine learning algorithms data"`
- Doc 2: `"data science statistics algorithms"`

**Vocabulary:** `{machine, learning, algorithms, data, science, statistics}`

**TF matrix:**

| Term | TF(Doc 1) | TF(Doc 2) |
|---|---|---|
| machine | 1/4 = 0.25 | 0/4 = 0.00 |
| learning | 1/4 = 0.25 | 0/4 = 0.00 |
| algorithms | 1/4 = 0.25 | 1/4 = 0.25 |
| data | 1/4 = 0.25 | 1/4 = 0.25 |
| science | 0/4 = 0.00 | 1/4 = 0.25 |
| statistics | 0/4 = 0.00 | 1/4 = 0.25 |

**IDF (N=2):**

| Term | df | IDF = log(2/(1+df)) |
|---|---|---|
| machine | 1 | log(2/2) = 0.00 |
| learning | 1 | log(2/2) = 0.00 |
| algorithms | 2 | log(2/3) = -0.18 |
| data | 2 | log(2/3) = -0.18 |
| science | 1 | log(2/2) = 0.00 |
| statistics | 1 | log(2/2) = 0.00 |

> With scikit-learn's smoothed variant, unique terms get positive IDF.

**Insight:** `"algorithms"` and `"data"` (shared terms) get lower IDF scores.
`"machine"`, `"learning"`, `"science"`, `"statistics"` (unique to one doc)
get higher scores, making them the true fingerprints of each document.

---

## 6. Vector Space Model

After TF-IDF, each document becomes a **point (vector) in an N-dimensional
space** where N = size of the vocabulary.

```
                  ↑  "machine"
                  │
        Doc 1  ●  │    (high on "machine", low on "science")
                  │
                  │
         ─────────┼──────────────────────→  "science"
                  │
                  │         ● Doc 2
                  │    (low on "machine", high on "science")
```

Documents with similar vocabulary will point in **similar directions** in this
space. The **angle** between them captures their semantic closeness — regardless
of document length.

---

## 7. Cosine Similarity

### 7.1 Definition

Cosine similarity measures the **cosine of the angle** between two vectors A
and B in the vector space:

```
                  A · B
similarity(A,B) = ───────────
                  ‖A‖ × ‖B‖

where:
  A · B  = dot product  = Σ (Aᵢ × Bᵢ)
  ‖A‖    = magnitude    = √(Σ Aᵢ²)
  ‖B‖    = magnitude    = √(Σ Bᵢ²)
```

The result is always in the range **[0, 1]** for TF-IDF vectors (no negative
weights), where:

| Score | Interpretation |
|---|---|
| **1.0 (100%)** | Identical documents (same angle — 0°) |
| **0.7 – 0.99** | Very high similarity — likely plagiarism |
| **0.4 – 0.7** | Moderate similarity — same topic / partial overlap |
| **0.1 – 0.4** | Low similarity — loosely related |
| **0.0 (0%)** | No shared vocabulary (angle = 90°) |

### 7.2 Why Cosine and Not Euclidean Distance?

**Euclidean distance** measures the straight-line gap between two points.
It is sensitive to document length — a long and short document on the same
topic will appear far apart even though they share the same ideas.

**Cosine similarity** measures direction, not magnitude. Two documents about
the same topic will point in the same direction regardless of their length.

```
Short doc:  [1, 2, 0, 1]  →  direction: mostly dim-2
Long doc:   [4, 8, 0, 4]  →  direction: mostly dim-2  (same!)

Cosine similarity = 1.0   (identical direction)
Euclidean distance = large (different magnitudes)
```

> **This is why cosine similarity is the industry standard for text comparison.**

### 7.3 Worked Calculation

**Doc A vector** (simplified 3D): `[0.5, 0.3, 0.0]`
**Doc B vector** (simplified 3D): `[0.4, 0.4, 0.0]`

```
A · B  = (0.5×0.4) + (0.3×0.4) + (0.0×0.0) = 0.20 + 0.12 + 0.00 = 0.32

‖A‖   = √(0.5² + 0.3² + 0.0²) = √(0.25+0.09) = √0.34 ≈ 0.583
‖B‖   = √(0.4² + 0.4² + 0.0²) = √(0.16+0.16) = √0.32 ≈ 0.566

                0.32
similarity = ─────────── = 0.969 ≈ 96.9%
             0.583 × 0.566
```

This score of **96.9%** indicates the two documents are almost certainly
plagiarised from each other.

---

## 8. Pairwise Document Comparison

Given N documents, the system computes an **N × N similarity matrix** where
entry `[i][j]` holds the cosine similarity between document `i` and document `j`.

### Properties of the Matrix

- **Symmetric:** `similarity(A, B) = similarity(B, A)`
- **Diagonal = 1.0:** Every document is 100% similar to itself
- **Only upper/lower triangle needed:** (N × (N-1)) / 2 unique pairs

### Example Matrix (4 documents)

```
               Doc_A   Doc_B   Doc_C   Doc_D
Doc_A         100.0%   74.1%   45.2%   22.2%
Doc_B          74.1%  100.0%   31.7%   21.5%
Doc_C          45.2%   31.7%  100.0%   21.6%
Doc_D          22.2%   21.5%   21.6%  100.0%
```

**Number of unique pairs** for N documents = N(N-1)/2

| N (docs) | Pairs |
|---|---|
| 2 | 1 |
| 5 | 10 |
| 10 | 45 |
| 20 | 190 |
| 100 | 4,950 |

---

## 9. Sentence-Level Analysis

Document-level similarity gives an overall score but cannot pinpoint **which
sentences** were copied. Sentence-level analysis adds precision.

### How It Works

1. **Sentence tokenization** — split each document into individual sentences
   using `nltk.tokenize.sent_tokenize` (which uses the Punkt model trained
   on punctuation patterns).

2. **Per-sentence preprocessing** — each sentence is preprocessed independently.

3. **Shared TF-IDF vectorization** — all sentences from both documents are
   vectorized together using a single vocabulary so their vectors are
   comparable.

4. **Cross-matrix cosine similarity** — compute similarity between every
   sentence in Doc A against every sentence in Doc B:
   an `|sents_A| × |sents_B|` matrix.

5. **Threshold filtering** — sentence pairs exceeding a lower threshold
   (default 50%) are returned, ranked by similarity.

### Why a Lower Threshold for Sentences?

A single sentence has far fewer unique words than a full document. The TF-IDF
vector is sparser, so even a strong match may score somewhat lower than the
document-level similarity. A threshold of **50%** at sentence level is
roughly equivalent to **70%** at document level.

---

## 10. Similarity Threshold & Flagging

The **threshold** is the minimum similarity score (as a percentage) above
which a document pair is considered potentially plagiarised.

### Choosing the Right Threshold

| Threshold | Effect | Best for |
|---|---|---|
| < 40% | Very lenient — many false positives | Exploratory analysis |
| 50–60% | Moderate — catches paraphrasing | Research papers |
| **70% (default)** | **Balanced — near-verbatim copying** | **General use** |
| 80–90% | Strict — only obvious copying | Short documents |
| > 90% | Very strict — almost exact copies | Code plagiarism |

> **Note:** There is no universally correct threshold. Domain knowledge,
> document type, and acceptable overlap (e.g., standard phrases in legal
> documents) should guide the choice.

### False Positives vs. False Negatives

```
              Predicted: Plagiarised   Predicted: Clean
Actual: Yes       True Positive           False Negative (missed!)
Actual: No        False Positive          True Negative
```

- **Lowering the threshold** → fewer missed cases, but more false alarms
- **Raising the threshold** → fewer false alarms, but some plagiarism missed

---

## 11. Heatmap Visualization

The similarity matrix is visualised as a **colour-coded heatmap** using
`seaborn.heatmap`.

### Reading the Heatmap

- **Colour scale** (YlOrRd — Yellow → Orange → Red):
  - 🟡 Yellow = low similarity (0–30%)
  - 🟠 Orange = moderate similarity (30–70%)
  - 🔴 Red = high similarity (70–100%)
- **Diagonal cells** are masked (shown as "—") because self-similarity
  is always 100% and would distort the colour scale.
- **Each cell** shows the exact percentage rounded to one decimal place.

### Why Visualize?

For large document sets (10+ files), scanning a raw table is impractical.
The heatmap lets you instantly spot clusters of similar documents — a row
or column with many red/orange cells immediately identifies a problematic
document.

---

## 12. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (UI)                        │
│   Drag & Drop Upload → Threshold Slider → Analyze       │
└──────────────────────────┬──────────────────────────────┘
                           │  HTTP POST /analyze
                           │  multipart/form-data
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Flask Backend (app.py)                  │
│                                                         │
│  1. Receive & validate uploaded .txt / .md files        │
│  2. Read file contents (UTF-8)                          │
│  3. Instantiate PlagiarismDetector(threshold)           │
│  4. Call calculate_similarity(documents)                │
│  5. Call detect_plagiarism(names, matrix)               │
│  6. Call sentence_level_comparison(doc1, doc2)          │
│  7. Render heatmap → encode as base64 PNG               │
│  8. Return JSON response                                │
└──────────────────────────┬──────────────────────────────┘
                           │  JSON response
                           ▼
┌─────────────────────────────────────────────────────────┐
│              PlagiarismDetector (plagiarism_detector.py) │
│                                                         │
│  preprocess()              → NLTK tokenize/stopwords    │
│  calculate_similarity()    → sklearn TfidfVectorizer    │
│                               cosine_similarity         │
│  detect_plagiarism()       → threshold comparison       │
│  sentence_level_comparison()→ sentence TF-IDF matrix    │
│  plot_heatmap()            → matplotlib / seaborn       │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Web framework | Flask | HTTP server, routing |
| NLP preprocessing | NLTK | Tokenization, stopwords, sentence splitting |
| Vectorization | scikit-learn `TfidfVectorizer` | TF-IDF matrix construction |
| Similarity | scikit-learn `cosine_similarity` | Pairwise cosine scores |
| Data handling | pandas | DataFrame construction and display |
| Visualization | matplotlib + seaborn | Heatmap generation |
| Frontend | HTML5 + CSS3 + Vanilla JS | Drag-and-drop UI, result rendering |

---

## 13. Limitations & Future Improvements

### Current Limitations

| Limitation | Impact |
|---|---|
| **No semantic understanding** | Cannot detect idea/paraphrase plagiarism where vocabulary differs completely |
| **Vocabulary-dependent** | Two documents on the same topic in different writing styles may score low |
| **English-only** | NLTK stopwords and tokenization tuned for English |
| **No stemming/lemmatization** | "run", "running", "ran" are treated as different terms |
| **File types** | Only `.txt` and `.md` — no PDF, DOCX support |
| **No web corpus comparison** | Compares only uploaded documents, not against the internet |

### Possible Improvements

1. **Stemming / Lemmatization** — Reduce "running" → "run" using
   `nltk.stem.PorterStemmer` or `nltk.stem.WordNetLemmatizer` to improve
   matching across different word forms.

2. **Word Embeddings / Semantic Similarity** — Replace TF-IDF with
   `sentence-transformers` (BERT-based models) to capture meaning even
   when vocabulary differs.

3. **N-gram Support** — Use bigrams and trigrams (`TfidfVectorizer(ngram_range=(1,3))`)
   to catch phrase-level matches that single-word models miss.

4. **PDF / DOCX Parsing** — Add `pdfminer` or `python-docx` to support
   common document formats.

5. **Source Comparison** — Integrate web scraping or an API (e.g., Google
   Custom Search) to compare against public web pages.

6. **Highlighted Report** — Generate an HTML/PDF report with plagiarised
   passages highlighted in the original text.

7. **Multilingual Support** — Add language detection and multilingual
   stopword lists via `langdetect` + `NLTK` multilingual corpora.

---

## 14. Glossary

| Term | Definition |
|---|---|
| **Corpus** | A collection of text documents used as the reference set |
| **Token** | A single unit of text — usually a word |
| **Tokenization** | Splitting a string into individual tokens |
| **Stopword** | A common word (e.g., "the", "is") that is filtered out |
| **Vocabulary** | The set of all unique tokens across the entire corpus |
| **Term Frequency (TF)** | How often a word appears in a specific document, normalised by length |
| **Inverse Document Frequency (IDF)** | A measure of how rare a word is across the corpus |
| **TF-IDF** | The product TF × IDF; a word's combined importance score |
| **Bag of Words (BoW)** | A text representation using raw word counts, ignoring order |
| **Vector Space Model** | Representing documents as vectors in a high-dimensional term space |
| **Cosine Similarity** | The cosine of the angle between two vectors; 1 = identical, 0 = unrelated |
| **Dot Product** | Sum of element-wise products of two vectors: A·B = Σ AᵢBᵢ |
| **Magnitude** | The length of a vector: ‖A‖ = √(Σ Aᵢ²) |
| **Similarity Matrix** | N×N matrix of pairwise similarity scores for N documents |
| **Threshold** | The minimum similarity percentage for a pair to be flagged |
| **False Positive** | Pair flagged as plagiarised when it actually is not |
| **False Negative** | Plagiarised pair that was missed (not flagged) |
| **Sentence Tokenization** | Splitting a document into individual sentences using punctuation cues |
| **Heatmap** | A colour-coded 2D matrix plot where cell colour represents a numerical value |
| **Base64** | Binary-to-text encoding used to embed PNG images directly in JSON responses |
| **Stemming** | Reducing a word to its root form (e.g., "running" → "run") |
| **Lemmatization** | Reducing a word to its dictionary base form considering grammar |
| **N-gram** | A contiguous sequence of N tokens (e.g., bigram: "machine learning") |
| **BERT** | Bidirectional Encoder Representations from Transformers — a deep language model |

---

*Generated for the PlagiaScope Plagiarism Detection System.*
*Libraries: scikit-learn · NLTK · pandas · matplotlib · seaborn · Flask*
