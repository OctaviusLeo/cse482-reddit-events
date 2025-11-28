from iris_reddit_events.config import RAW_DIR, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
import praw
from pathlib import Path
import json
from datetime import datetime

def get_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

def collect_sample(subreddits=("news", "worldnews", "technology"), limit=300):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"reddit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

    reddit = get_reddit()
    with out_path.open("w", encoding="utf-8") as f:
        for sub in subreddits:
            for post in reddit.subreddit(sub).new(limit=limit):
                record = {
                    "id": post.id,
                    "subreddit": sub,
                    "created_utc": int(post.created_utc),
                    "title": post.title,
                    "selftext": post.selftext or "",
                    "score": int(post.score),
                    "num_comments": int(post.num_comments),
                    "url": post.url,
                    "permalink": post.permalink,
                }
                f.write(json.dumps(record) + "\n")
    print("Wrote:", out_path)

if __name__ == "__main__":
    collect_sample()
