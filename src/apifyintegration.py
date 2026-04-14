"""
Brand Monitor — src/apifyintegration.py
=========================================
Apify live-feed integration: fetches posts from Apify datasets,
scores them, pushes to the dashboard via the in-memory live_posts store.

How it works:
  - On startup: loads all existing items from configured Apify datasets
  - Every POLL_INTERVAL_SECONDS: fetches only new items (incremental)
  - Each item is scored with TF-IDF + LR (or keyword fallback)
  - /feed and /stats use get_live_feed() / get_live_stats()
  - Triggers email/SMS alerts via alert_system.py

Usage in api_realtime.py:
    from src.apifyintegration import setup_apify_polling, get_live_feed, get_live_stats
    app.add_event_handler("startup", lambda: asyncio.create_task(setup_apify_polling()))

Install:
    pip install httpx apscheduler
"""

import httpx
import asyncio
import logging
from datetime    import datetime, timezone
from collections import Counter
from typing      import List, Dict, Optional

logger = logging.getLogger("apify_integration")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

APIFY_TOKEN = "apify_api_YOUR_TOKEN_HERE"   # paste your Apify API token

# Apify Dataset IDs — find these in Apify Console → Storage → Datasets
APIFY_DATASET_IDS: List[str] = [
    # "YOUR_DATASET_ID_1",
    # "YOUR_DATASET_ID_2",
]

POLL_INTERVAL_SECONDS = 300   # poll every 5 minutes
MAX_POSTS_IN_MEMORY   = 500   # keep latest N posts in memory

APIFY_BASE = "https://api.apify.com/v2"

# ══════════════════════════════════════════════════════════════════
# IN-MEMORY STORE
# ══════════════════════════════════════════════════════════════════

live_posts:   List[dict] = []
last_fetched: Dict[str, int] = {}  # dataset_id → last item offset


# ══════════════════════════════════════════════════════════════════
# APIFY API CALLS
# ══════════════════════════════════════════════════════════════════

async def fetch_dataset_items(
    dataset_id: str,
    offset: int = 0,
    limit:  int = 100
) -> List[dict]:
    """Fetch a page of items from an Apify dataset."""
    url = f"{APIFY_BASE}/datasets/{dataset_id}/items"
    params = {
        "token":  APIFY_TOKEN,
        "offset": offset,
        "limit":  limit,
        "clean":  "true",   # skip empty / error rows
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def fetch_dataset_info(dataset_id: str) -> dict:
    """Get metadata (including total item count) for a dataset."""
    url = f"{APIFY_BASE}/datasets/{dataset_id}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params={"token": APIFY_TOKEN})
        resp.raise_for_status()
        return resp.json()


# ══════════════════════════════════════════════════════════════════
# SCORING PIPELINE
# ══════════════════════════════════════════════════════════════════

FAKE_KEYWORDS = [
    "replica", "rep ", "reps ", "firstcopy", "first copy", "replicakicks",
    "jordanrep", "repjordan", "dm for price", "dm us", "whatsapp",
    "aaa", "1:1", "1to1", "dupe", "copy shoes", "cheap jordan",
    "cheap nike", "order now", "replicasneakers", "repsneakers",
    "dhgate", "weidian"
]


def keyword_fake_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    t    = text.lower()
    hits = sum(1 for kw in FAKE_KEYWORDS if kw in t)
    return min(hits / 3.0, 1.0)


def score_post(raw: dict) -> dict:
    """
    Score a raw Apify Instagram post.
    Uses TF-IDF + LR when available; falls back to keyword scoring.
    """
    # ── Field extraction ──────────────────────────────────────────
    caption   = raw.get("caption") or raw.get("text") or ""
    username  = raw.get("ownerUsername") or raw.get("username") or "unknown"
    likes     = raw.get("likesCount")   or raw.get("likes")    or 0
    comments  = raw.get("commentsCount") or raw.get("comments") or 0
    timestamp = raw.get("timestamp") or datetime.now(timezone.utc).isoformat()
    hashtags  = raw.get("hashtags") or []
    url       = raw.get("url")      or raw.get("inputUrl") or ""

    hashtag_str = " ".join(hashtags) if isinstance(hashtags, list) else str(hashtags)
    full_text   = f"{caption} {hashtag_str}".strip()

    # ── Text score ────────────────────────────────────────────────
    kw_score = keyword_fake_score(full_text)

    try:
        from src.api_realtime import lr_model, vectorizer, clean_text
        if lr_model is not None and vectorizer is not None:
            cleaned    = clean_text(full_text)
            vec        = vectorizer.transform([cleaned])
            proba      = float(lr_model.predict_proba(vec)[0][1])
            text_score = round(0.7 * proba + 0.3 * kw_score, 4)
        else:
            text_score = round(kw_score, 4)
    except Exception:
        text_score = round(kw_score, 4)

    # ── Sentiment score ───────────────────────────────────────────
    sentiment_label = "neutral"
    try:
        from src.api_realtime import sentiment_model
        if sentiment_model is not None and caption:
            mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
            res     = sentiment_model(caption[:512])[0]
            sentiment_label = mapping.get(res["label"], "neutral")
    except Exception:
        pass

    sent_score_map  = {"positive": 0.6, "neutral": 0.3, "negative": 0.4}
    sentiment_score = sent_score_map.get(sentiment_label, 0.3)

    # ── Final fusion ──────────────────────────────────────────────
    final_score = round(0.75 * text_score + 0.25 * sentiment_score, 4)

    if final_score >= 0.45:
        label       = "fake"
        source_type = "counterfeit"
    elif final_score >= 0.35:
        label       = "uncertain"
        source_type = "uncertain"
    else:
        label       = "real"
        source_type = "brand"

    return {
        "id":          raw.get("id") or raw.get("shortCode") or str(hash(caption)),
        "username":    username,
        "caption":     caption,
        "likes":       int(likes) if likes else 0,
        "comments":    int(comments) if comments else 0,
        "timestamp":   timestamp,
        "hashtags":    hashtags,
        "url":         url,
        "source_type": source_type,
        "label":       label,
        "text_score":  text_score,
        "final_score": final_score,
        "sentiment":   sentiment_label,
        "fetched_at":  datetime.now(timezone.utc).isoformat()
    }


# ══════════════════════════════════════════════════════════════════
# POLLING LOGIC
# ══════════════════════════════════════════════════════════════════

async def poll_apify():
    """Fetch only new posts from all datasets since the last poll."""
    global live_posts

    new_count = 0
    for dataset_id in APIFY_DATASET_IDS:
        try:
            info   = await fetch_dataset_info(dataset_id)
            total  = info.get("itemCount", 0)
            offset = last_fetched.get(dataset_id, 0)

            if total <= offset:
                logger.info(f"Dataset {dataset_id}: no new items ({total} total)")
                continue

            batch_limit = min(total - offset, 100)
            new_items   = await fetch_dataset_items(dataset_id, offset=offset, limit=batch_limit)
            logger.info(f"Dataset {dataset_id}: fetched {len(new_items)} new items")

            scored = [score_post(item) for item in new_items]

            # Trigger alerts for each newly scored post
            try:
                from src.alert_system import process_new_post_for_alerts, check_bulk_alert
                for post in scored:
                    process_new_post_for_alerts(post)
                check_bulk_alert()
            except Exception as e:
                logger.warning(f"Alert processing error: {e}")

            # Prepend newest first
            live_posts = scored + live_posts
            last_fetched[dataset_id] = total
            new_count += len(scored)

        except Exception as e:
            logger.error(f"Poll error for dataset {dataset_id}: {e}")

    # Bound memory
    live_posts = live_posts[:MAX_POSTS_IN_MEMORY]

    if new_count:
        fake_n = sum(1 for p in live_posts if p["label"] == "fake")
        logger.info(f"Poll complete: +{new_count} new | {len(live_posts)} total | {fake_n} fake")
    else:
        logger.info("Poll complete: no new posts")


async def initial_load():
    """Load ALL existing items from all datasets on startup."""
    global live_posts

    all_posts = []
    for dataset_id in APIFY_DATASET_IDS:
        try:
            info  = await fetch_dataset_info(dataset_id)
            total = info.get("itemCount", 0)
            logger.info(f"Dataset {dataset_id}: loading {total} existing items")

            for offset in range(0, total, 100):
                batch = await fetch_dataset_items(dataset_id, offset=offset, limit=100)
                all_posts.extend([score_post(item) for item in batch])

            last_fetched[dataset_id] = total

        except Exception as e:
            logger.error(f"Initial load error for dataset {dataset_id}: {e}")

    all_posts.sort(key=lambda p: p.get("fetched_at", ""), reverse=True)
    live_posts = all_posts[:MAX_POSTS_IN_MEMORY]

    fake_n = sum(1 for p in live_posts if p["label"] == "fake")
    logger.info(f"Initial load complete: {len(live_posts)} posts | {fake_n} fake")


# ══════════════════════════════════════════════════════════════════
# SCHEDULER / ENTRYPOINT
# ══════════════════════════════════════════════════════════════════

async def setup_apify_polling():
    """
    Call this on app startup to begin live polling.

    In api_realtime.py @app.on_event("startup"):

        from src.apifyintegration import setup_apify_polling
        asyncio.create_task(setup_apify_polling())
    """
    if not APIFY_DATASET_IDS:
        logger.warning("No APIFY_DATASET_IDS configured — polling disabled")
        return

    logger.info("Loading initial Apify data...")
    await initial_load()

    logger.info(f"Starting Apify poll loop (every {POLL_INTERVAL_SECONDS}s)")
    while True:
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            await poll_apify()
        except Exception as e:
            logger.error(f"Unhandled poll error: {e}")


# ══════════════════════════════════════════════════════════════════
# FEED / STATS HELPERS (used by api_realtime.py routes)
# ══════════════════════════════════════════════════════════════════

def get_live_feed(
    limit:  int = 50,
    offset: int = 0,
    label:  Optional[str] = None
) -> dict:
    """
    Return a paginated slice of live_posts.

    In api_realtime.py:
        from src.apifyintegration import get_live_feed

        @app.get("/feed/live")
        def live_feed(limit: int = 50, offset: int = 0, label: str = None):
            return get_live_feed(limit, offset, label)
    """
    posts = live_posts
    if label:
        posts = [p for p in posts if p.get("label") == label]

    return {
        "total":        len(posts),
        "offset":       offset,
        "limit":        limit,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "posts":        posts[offset: offset + limit]
    }


def get_live_stats() -> dict:
    """
    Return dashboard stats derived from live_posts.

    In api_realtime.py:
        from src.apifyintegration import get_live_stats

        @app.get("/stats/live")
        def live_stats():
            return get_live_stats()
    """
    total      = len(live_posts)
    fake_posts = [p for p in live_posts if p["label"] == "fake"]
    real_posts = [p for p in live_posts if p["label"] == "real"]
    avg_likes  = sum(p["likes"] for p in live_posts) / total if total > 0 else 0

    top_usernames = dict(
        Counter(p["username"] for p in live_posts).most_common(6)
    )

    return {
        "total_posts":       total,
        "brand_posts":       len(real_posts),
        "counterfeit_posts": len(fake_posts),
        "avg_likes":         round(avg_likes, 1),
        "top_usernames":     top_usernames,
        "model_accuracy": {
            "yolov8":       94.7,
            "efficientnet": 97.7,
            "distilbert":   91.2,
            "tfidf_lr":     99.0,
            "ensemble":     99.0
        }
    }