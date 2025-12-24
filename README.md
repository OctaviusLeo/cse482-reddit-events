# IRIS: Reddit Event Detection & Summarization

End-to-end NLP pipeline that collects public Reddit posts, clusters them into **events**, and generates short **extractive summaries** per event.

This is a compact “AI + SWE” project: API ingestion → preprocessing/vectorization → unsupervised clustering → summarization → evaluation (ROUGE), organized as a reusable Python package under `src/`.

## What this demonstrates (for interviewers)

- **Data ingestion**: Reddit API collection via `praw`, persisted as JSONL.
- **NLP feature engineering**: TF–IDF vectorization + stopwords.
- **Unsupervised learning**: agglomerative clustering (cosine distance) for event discovery.
- **Summarization baseline**: centroid-based extractive selection with evidence IDs.
- **Evaluation**: small gold set + ROUGE scoring.
- **SWE hygiene**: modular pipeline (`collect`, `preprocess`, `events`, `summarize`, `evaluate`) with clear I/O boundaries.

## Tech stack

- Python, pandas, numpy
- scikit-learn (TF–IDF, clustering)
- NLTK (stopwords)
- praw (Reddit API)
- rouge-score (evaluation)

## How it works (high level)

1. **Collect** posts from selected subreddits → `data/raw/reddit_*.jsonl`
2. **Preprocess + vectorize** (title + selftext → cleaned text → TF–IDF) → `data/processed/posts.parquet`
3. **Cluster into events** with agglomerative clustering → `data/events/events_labeled.parquet`
4. **Summarize each event** by selecting the most centroid-similar posts → `data/summaries/summaries.parquet`
5. **Evaluate** summaries vs. a small human-written reference set → ROUGE metrics

## Quickstart (Windows / PowerShell)

From the repo root.

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Make the package importable

This repo uses a `src/` layout but does not include packaging metadata (no `pyproject.toml`). For local runs, set `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
```

4) Download NLTK stopwords (one-time)

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

## Reddit credentials

Create a `.env` file in the repo root:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=iris-reddit-events (by u/your_username)
```

Credentials are loaded in `src/iris_reddit_events/config.py`.

## Run the pipeline

All commands below assume `PYTHONPATH=src` (see Quickstart).

### 1) Collect data

```powershell
python -m iris_reddit_events.collect
```

This writes a timestamped JSONL file under `data/raw/`.

### 2) Run preprocessing → clustering → summarization

```powershell
python -m iris_reddit_events.pipeline
```

Defaults:
- Subreddits: `news`, `worldnews`, `technology`
- Clustering threshold: `distance_threshold=0.8`
- Summary size: `top_k=3` titles

You can adjust these in `src/iris_reddit_events/pipeline.py` (or call the underlying functions directly).

### 3) Evaluate (ROUGE)

1. Create `data/gold_summaries.csv` with:

```csv
event_id,reference
0,"<your human-written summary for event 0>"
3,"..."
```

2. Run evaluation:

```powershell
python -m iris_reddit_events.evaluate --gold data/gold_summaries.csv
```

## Notebooks

Notebooks mirror the pipeline stages:

- `notebooks/00_check_config.ipynb`: confirms paths + `.env` loading.
- `notebooks/01_preprocess.ipynb`: loads raw JSONL, builds TF–IDF.
- `notebooks/02_events.ipynb`: clusters posts into `event_id`.
- `notebooks/03_summarize.ipynb`: generates summaries from labeled events.
- `notebooks/04_eval.ipynb`: reads summaries, creates a gold template, runs ROUGE.

## Outputs

- `data/raw/reddit_*.jsonl`: raw collection results
- `data/processed/posts.parquet`: cleaned post table
- `data/events/events_labeled.parquet`: post table + `event_id`
- `data/summaries/summaries.parquet`: event-level summaries (+ evidence post IDs)

## Repo structure

```text
src/iris_reddit_events/
	collect.py     # Reddit API ingestion
	preprocess.py  # text cleaning + TF–IDF
	events.py      # clustering into events
	summarize.py   # centroid-based extractive summaries
	evaluate.py    # ROUGE evaluation helpers + CLI
	pipeline.py    # end-to-end runner
	config.py      # paths + .env loading
notebooks/       # exploratory / report notebooks
```

## Common issues / troubleshooting

- Parquet read/write error: install an engine (recommended): `pip install pyarrow`
- `LookupError: Resource stopwords not found`: run `python -c "import nltk; nltk.download('stopwords')"`
- `TypeError: ... unexpected keyword argument 'affinity'` (scikit-learn): the clustering API changed in newer versions; pin scikit-learn to an older release or update `events.py` to use `metric=`.
- `No raw files found`: run the collector first so `data/raw/reddit_*.jsonl` exists.

## Future improvements

- Replace TF–IDF with dense embeddings (e.g., sentence-transformers) for more coherent clustering.
- Move from document-level to sentence-level extractive summarization (e.g., TextRank) to reduce redundancy.
- Add a small time-aware event segmentation step (burst detection / windowing).
