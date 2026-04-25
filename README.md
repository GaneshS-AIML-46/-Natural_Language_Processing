# PlagiaScope — Plagiarism Detection System

> Detect plagiarism instantly using **TF-IDF** and **Cosine Similarity** — with a clean web UI, similarity matrix, sentence-level breakdown, and heatmap visualization.

---

## Features

- **Multi-file upload** — drag & drop `.txt` or `.md` files (2 or more)
- **Text preprocessing** — lowercase, punctuation removal, stopword filtering
- **TF-IDF vectorization** — converts documents into numerical fingerprints
- **Cosine similarity** — pairwise comparison across all document pairs
- **Similarity matrix** — color-coded table (red = flagged, blue = low)
- **Adjustable threshold** — slider from 10% to 99% (default: 70%)
- **Flagged pairs** — lists suspected plagiarism with visual progress bars
- **Sentence-level analysis** — pinpoints exact sentences that were copied
- **Heatmap** — downloadable PNG visualization of the full similarity matrix

---

## Project Structure

```
NLP/
├── app.py                    # Flask backend — handles uploads & analysis
├── plagiarism_detector.py    # Core NLP engine (PlagiarismDetector class)
├── requirements.txt          # Python dependencies
├── concepts.md               # Theory & concept reference
├── templates/
│   └── index.html            # Web UI (dark-themed, responsive)
├── static/
│   ├── style.css             # Premium dark-mode styling
│   └── script.js             # Drag-drop, API calls, result rendering
└── sample_docs/
    ├── document_A.txt              # Original sample document
    ├── document_B_plagiarized.txt  # Near-copy of A (test plagiarism)
    └── document_C_original.txt     # Unrelated document (control)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| NLP Preprocessing | NLTK (tokenizer, stopwords, sentence splitter) |
| Vectorization | scikit-learn `TfidfVectorizer` |
| Similarity | scikit-learn `cosine_similarity` |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Frontend | HTML5, CSS3, Vanilla JavaScript |

---

## Installation

```bash
# 1. Clone or download the project
cd NLP

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Start the web server
python app.py

# Open in your browser
http://127.0.0.1:5000
```

1. **Upload** — drag and drop 2 or more `.txt` / `.md` files
2. **Set threshold** — adjust the slider (default 70%)
3. **Analyze** — click "Analyze Documents"
4. **View results** — similarity matrix, flagged pairs, sentence matches, heatmap

---

## How It Works

```
Upload Files → Preprocess Text → TF-IDF Vectors → Cosine Similarity → Flag & Report
```

| Step | What Happens |
|---|---|
| Preprocessing | Lowercase → remove punctuation → tokenize → remove stopwords |
| TF-IDF | Each document becomes a weighted numeric vector |
| Cosine Similarity | Angle between vectors = similarity score (0–100%) |
| Flagging | Pairs above threshold are marked as potential plagiarism |
| Sentence Analysis | Each sentence pair is scored independently for precision |

---

## Quick Concepts

**TF-IDF** weights words by how frequent they are in a document vs. how rare
they are across all documents — making it a powerful text fingerprinting tool.

**Cosine Similarity** measures the angle between two document vectors.
A score of `1.0` means identical; `0.0` means no shared vocabulary.
It is length-independent, making it ideal for comparing documents of different sizes.

> See [`concepts.md`](./concepts.md) for the full theory with formulas and examples.

---

## Sample Output

| Pair | Similarity | Status |
|---|---|---|
| document_A ↔ document_B_plagiarized | **85.2%** | 🚨 Flagged |
| document_A ↔ document_C_original | **4.1%** | ✅ Clean |
| document_B ↔ document_C_original | **3.7%** | ✅ Clean |

---

## File Format Support

| Format | Supported |
|---|---|
| `.txt` | ✅ Yes |
| `.md` | ✅ Yes |
| `.pdf` | ❌ Not yet |
| `.docx` | ❌ Not yet |

---

## License

This project is for educational purposes. Built with Python, Flask, NLTK, and scikit-learn.
