from sklearn.cluster import AgglomerativeClustering
from iris_reddit_events.config import EVENTS_DIR
from pathlib import Path
import pandas as pd
import numpy as np

def cluster_events(X, df_posts, n_clusters=None, distance_threshold=0.8):
    """
    Use AgglomerativeClustering with cosine distance.
    set n_clusters=None and distance_threshold<1 to cut by distance.
    """
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    model = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(X.toarray())  # OK for small N
    df = df_posts.copy()
    df["event_id"] = labels
    df.to_parquet(EVENTS_DIR / "events_labeled.parquet")
    return df
