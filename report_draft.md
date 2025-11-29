# IRIS: Event Detection & Summarization from Reddit

## Team
- **Symaedchit Octavius Leo** (solo)
### Contributions
All contributions (design, implementation, experiments, analysis, and writing) were done by me. Specifically:
- **System design & implementation**: Project structure, data collection, preprocessing, clustering, summarization, evaluation scripts.
- **Experiments**: Running the end-to-end pipeline on Reddit data, exploring hyperparameters, manual inspection of clusters and summaries.
- **Evaluation & analysis**: Designing a small human-labeled evaluation set and computing ROUGE scores; qualitative error analysis.
- **Report writing**: All sections of this report.

---

## Abstract
This project builds a small, end-to-end pipeline for **event detection and summarization from Reddit**. Using the official Reddit API, the system collects public posts from a handful of news-related subreddits, clusters them into events based on textual similarity, and generates short extractive summaries for each event. The implementation is done in Python using standard data science libraries (pandas, scikit-learn, NLTK) and is organized as a reusable package.

Event detection is implemented using agglomerative clustering on TF–IDF vectors. Summarization is implemented as a centroid based extractive method that selects sentences (or titles) closest to the cluster centroid. The system outputs both event assignments for each post and a summary with evidence post IDs for each event. A small human labeled set of reference summaries is used to evaluate summary quality with ROUGE, and qualitative inspection is used to judge event coherence. Results show that even simple unsupervised methods produce reasonably coherent events and summaries for many news stories while also revealing common failure modes such as merging unrelated but lexically similar posts. The report concludes with lessons learned and directions for improving clustering and summarization in future iterations.

---

## 1. Introduction
Reddit is a large, user generated content platform where news and discussions about real world events appear quickly and in high volume. For both researchers and end users it is useful to automatically detect **events** (groups of posts about the same story) and produce concise **summaries** that capture the main information without requiring a user to read dozens of posts.

In the context of CSE 482 (Big Data Analysis), this project explores a focused version of that problem:
1. Collect a sample of posts from a small set of public, news-oriented subreddits.
2. Detect events by clustering posts based on text similarity.
3. Generate short, extractive summaries for each event.
4. Evaluate summary quality using a small human-labeled set and ROUGE.

The goal is not to build a production Reddit product, but to implement a **clean, end-to-end data analysis pipeline** that illustrates concepts from the course: data collection, preprocessing, vectorization, clustering, and evaluation of text mining methods.

Section 2 describes the methods: data collection, preprocessing and vectorization, clustering, summarization, and evaluation. 
Section 3 presents results and analysis, including examples of detected events and summaries, and discusses typical strengths and weaknesses. 
Section 4 summarizes lessons learned and outlines possible improvements.

---

## 2. Method

### 2.1 Data Collection
Data is collected using the official Reddit API via the Python `praw` library. For this project, I created a script app on my Reddit account and used OAuth credentials (client ID, client secret, user agent) stored in a local `.env` file. You must go to https://www.reddit.com/prefs/apps and fill out https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164. This file is ignored by version control to avoid leaking secrets.

The data collection script (`collect.py`) fetches **recent posts** from a small set of public subreddits, for example:
- `r/news`
- `r/worldnews`
- `r/technology`

For each subreddit, the script retrieves up to a fixed number of most recent posts (300 posts per subreddit per run) using the `new` listing. For each post, the script stores:
- `id`
- `subreddit`
- `created_utc`
- `title`
- `selftext`
- `score`
- `num_comments`
- `url`
- `permalink`

Records are written to **newline-delimited JSON** files in `data/raw/`, with filenames like `reddit_YYYYMMDD_HHMMSS.jsonl`. The code is designed such that multiple runs can accumulate more data over time.

All data used in this project is **public Reddit content** accessed via the official API under Reddit’s policies, and used solely for offline academic analysis.

### 2.2 Preprocessing & Vectorization
Preprocessing is implemented in `preprocess.py`. It performs the following steps:
1. **Load raw data**: All `reddit_*.jsonl` files in `data/raw/` are read and combined into a single pandas `DataFrame`.
2. **Text construction**: For each post, a document string is created as the concatenation of `title` and `selftext`.
3. **Text cleaning**: The document is lowercased and lightly normalized (removing extra whitespace). Heavy normalization and aggressive filtering are intentionally avoided to keep the pipeline simple and transparent.
4. **Stopword removal and TF–IDF**: A `TfidfVectorizer` (from scikit-learn) is used with an English stopword list (from NLTK), and the vocabulary is capped at a maximum number of features. The result is a sparse TF–IDF matrix `X` where each row corresponds to a post.

The cleaned DataFrame with original metadata is saved to `data/processed/posts.parquet` for reuse. The TF–IDF matrix is returned in memory to downstream components.
This step implements the key feature engineering stage of transforming raw text into a numerical representation suitable for clustering.

### 2.3 Event Detection via Clustering
Event detection is framed as a **clustering problem**: posts that are about the same real world event should appear in the same cluster. For this project, I use **agglomerative clustering** with cosine distance, implemented in `events.py`.

Given the TF–IDF matrix `X`:
- The algorithm starts with each post as its own cluster.
- At each step it merges the two closest clusters according to **average linkage** and **cosine affinity**.
- Instead of fixing the number of clusters in advance a **distance threshold** is used to cut the dendrogram. Posts whose pairwise distance stays below this threshold form a cluster others remain separate.

The resulting cluster label for each post is called `event_id` and is appended to the posts DataFrame. The labeled DataFrame is saved to `data/events/events_labeled.parquet`.

This approach is simple but effective for grouping together news posts that share vocabulary. It tends to create larger clusters for big events and smaller clusters for niche topics or noise.

### 2.4 Summarization (Extractive Baseline)

Given an event (cluster of posts), the goal is to produce a short summary that captures the central information. For this project, I focus on an **extractive baseline** implemented in `summarize.py`:

1. For each event, collect the **documents** (combined title + selftext) for all posts inside the cluster.
2. Build a TF–IDF representation for the documents within this cluster.
3. Compute the **centroid** of the cluster as the mean of the TF–IDF vectors.
4. For each document, compute a similarity score to the centroid (via dot product).
5. Select the top-`k` documents (e.g., `k=3`) with the highest similarity scores.
6. Construct the event summary by concatenating the titles (or top sentences) of these representative posts.

The output is a DataFrame of summaries, with each row containing:

- `event_id`
- `summary` (system-generated extractive summary)
- `num_posts` (size of the event cluster)
- `evidence_ids` (list of post IDs used to construct the summary)

This DataFrame is saved to `data/summaries/summaries.parquet`.

The method is intentionally straightforward, but it has desirable properties:

- It uses only unsupervised information (no labeled training data).
- It is easy to interpret (the summary is directly supported by specific posts).
- It can be improved later by swapping in more advanced extractive or abstractive methods without changing the rest of the pipeline.

### 2.5 Evaluation Protocol

Summarization quality is evaluated using a small human-labeled set and the ROUGE metric, implemented in `evaluate.py` and the notebook `04_eval.ipynb`.

The evaluation protocol is:
1. **Sample events**: Choose a small subset of event IDs (e.g., 10–15) from the generated summaries, focusing on reasonably large clusters that appear to correspond to real events.
2. **Create gold summaries**: For each selected event, manually write a short human reference summary capturing the essential information. These are stored in a CSV file `data/gold_summaries.csv` with columns `event_id` and `reference`.
3. **Align with system summaries**: Join the gold table with system summaries on `event_id`.
4. **Compute ROUGE**: Use the `rouge-score` library to compute ROUGE-1, ROUGE-2, and ROUGE-L precision/recall/F1 between the system summary and the reference for each event, then average across events.

The evaluation is limited in scale (small gold set) but provides a quantitative signal and a structured way to inspect which events are summarized well and which are not. Qualitative inspection of clusters and summaries supplements the numeric scores.

---

## 3. Results and Analysis
### 3.1 Event Examples

Running the pipeline on a recent snapshot of `r/news`, `r/worldnews`, and `r/technology` produced clusters that correspond to familiar news events, such as:
- Major political announcements or elections.
- High-profile technology product releases or outages.
- Large accidents or natural disasters.

For a typical high impact event the pipeline groups together many posts that share the same topic but differ in wording, source, or perspective. The centroid-based summarizer tends to select representative titles that mention the core entities and actions (“Company X announces Y”, “Government passes law Z”).

Smaller clusters often capture niche or localized stories, sometimes mixed with discussion posts.

### 3.2 Summarization Quality

In the ROUGE-based evaluation on a small human-labeled set of events the extractive summaries achieve reasonable overlap with the reference summaries. In many cases the selected titles/phrases cover the main entities (who/what) and actions (what happened).

However, several limitations are visible:
- **Redundancy**: When the top-k representative posts are very similar the summary can repeat nearly the same information instead of adding complementary details.
- **Missing context**: Some summaries capture the core action but omit crucial context such as location or consequences when those appear mostly in less central posts.
- **No sentence-level selection**: Because the system operates at the post/document level it cannot yet pick the single best sentence inside a long selftext.

Despite these issues, the baseline demonstrates that simple unsupervised summarization can produce readable, event level summaries aligned with standard metrics.

### 3.3 Cluster Coherence and Failure Modes

Qualitative inspection of clusters shows several clear patterns:

- **High-coherence clusters**: For large, well known news events, posts use similar vocabulary and the clustering algorithm creates clean, coherent clusters. Summaries from these clusters are usually good.
- **Thematic but multi-event clusters**: Sometimes posts about related but distinct sub events (a multi-day story or multiple updates) are merged into a single cluster. Summaries in these cases may blur together slightly different timelines.
- **Lexical collisions**: In rare cases posts with overlapping keywords but unrelated topics can be grouped together. This is a limitation of purely lexical similarity and the absence of more sophisticated semantics.

These observations suggest that clustering quality is generally adequate for the project goals but could be improved with better representations (sentence embeddings) or additional features (time, subreddit).

---

## 4. Lessons Learned and Future Improvement
### 4.1 Lessons Learned
- **End-to-end structure matters**: Having a clear pipeline (collection -> preprocessing -> clustering -> summarization -> evaluation) made it easier to develop and debug individual components and to reason about where errors come from.
- **Text representation is critical**: Even a simple TF–IDF representation captures enough lexical information to detect many events, but its limitations (ignoring word order and deeper semantics) are visible, especially in borderline clusters.
- **Evaluation is non-trivial**: Designing a fair, informative evaluation for summarization is hard. A small human labeled gold set combined with ROUGE provides some signal, but it is not sufficient to fully capture summary quality. Qualitative inspection remains important.
- **Tooling is a force multiplier**: Using a structured project layout version control and Jupyter notebooks made experimentation smoother and more reproducible, even for a relatively small project.

### 4.2 Future Improvement
Several improvements could significantly enhance the system:
1. **Better document embeddings**  
   Replace TF–IDF with dense embeddings (from pre-trained transformer models) and cluster in that space. This would likely reduce lexical collisions and improve event boundaries.
2. **Sentence-level extractive summarization**  
   Instead of selecting whole posts split documents into sentences and apply centroid or graph-based ranking (TextRank) at the sentence level within each event. This would allow more concise less redundant summaries.
3. **Abstractive component**  
   As a further extension feed the event cluster into a small summarization model (T5/BART) to generate abstractive summaries using the current extractive method as a fallback or as input.
4. **Richer evaluation**  
   Expand the gold set of reference summaries and consider additional metrics (human judgment of informativeness and fluency). Compare multiple summarization strategies (centroid vs. TextRank vs. abstractive).
5. **Temporal modeling**  
   Incorporate timestamps more explicitly, such as detecting event bursts over time and distinguishing early vs. late posts when summarizing evolving stories.

These directions show how the current project can serve as a foundation for more advanced research on event detection and summarization in social media data.

---

## References

No external references are strictly required for this report. All methods used are standard techniques from class (TF–IDF, clustering, extractive summarization, ROUGE), implemented with open source Python libraries.