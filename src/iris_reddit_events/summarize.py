from iris_reddit_events.config import EVENTS_DIR, SUMMARIES_DIR
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def summarize_events(df_events: pd.DataFrame, top_k=3):
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for event_id, group in df_events.groupby("event_id"):
        texts = (group["title"].fillna("") + " " + group["selftext"].fillna("")).tolist()
        if not texts:
            continue
        vec = TfidfVectorizer(max_features=2000, stop_words="english")
        X = vec.fit_transform(texts)
        centroid = X.mean(axis=0)
        sims = (X @ centroid.T).A.ravel()
        top_idx = np.argsort(-sims)[:top_k]

        chosen = group.iloc[top_idx]
        summary_text = " ".join(chosen["title"].fillna("").tolist())
        summaries.append({
            "event_id": int(event_id),
            "summary": summary_text,
            "num_posts": len(group),
            "evidence_ids": chosen["id"].tolist(),
        })

    df_sum = pd.DataFrame(summaries).sort_values("num_posts", ascending=False)
    df_sum.to_parquet(SUMMARIES_DIR / "summaries.parquet")
    return df_sum
