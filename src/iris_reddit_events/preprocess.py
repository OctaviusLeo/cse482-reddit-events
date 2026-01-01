from iris_reddit_events.config import RAW_DIR, PROCESSED_DIR
from pathlib import Path
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# nltk.download('stopwords')

from nltk.corpus import stopwords

STOP = stopwords.words("english")  # Keep as list for sklearn

def load_raw() -> pd.DataFrame:
    rows = []
    for path in RAW_DIR.glob("reddit_*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def clean_text(text: str) -> str:
    text = text.lower()
    # remove line breaks
    return " ".join(text.split())

def build_corpus(df: pd.DataFrame):
    docs = (df["title"].fillna("") + " " + df["selftext"].fillna("")).map(clean_text)
    return docs

def vectorize(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    docs = build_corpus(df)
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(docs)
    df_out = df.copy()
    df_out.to_parquet(PROCESSED_DIR / "posts.parquet")
    # save sparse matrix and vocab separately if needed
    return X, vec, df_out
