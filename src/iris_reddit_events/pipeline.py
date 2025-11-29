from iris_reddit_events import collect, preprocess, events, summarize
from iris_reddit_events.config import RAW_DIR, EVENTS_DIR, SUMMARIES_DIR
from pathlib import Path

def main():
    # collect.collect_sample()

    # Load + vectorize
    df = preprocess.load_raw()
    X, vec, df_clean = preprocess.vectorize(df)

    # Cluster
    df_events = events.cluster_events(X, df_clean, n_clusters=None, distance_threshold=0.8)

    # Summarize
    df_sum = summarize.summarize_events(df_events)

    print("Events saved at:", EVENTS_DIR)
    print("Summaries saved at:", SUMMARIES_DIR)
    print(df_sum.head())

if __name__ == "__main__":
    main()
