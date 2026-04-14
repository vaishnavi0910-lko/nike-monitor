"""
Brand Monitor — src/apify_scraper.py
=======================================
Live Apify scraping + background scheduler.
Scrapes Instagram hashtags via Apify's Instagram Scraper actor,
stores results in SQLite, and broadcasts new posts over WebSocket.

Env vars:
    APIFY_TOKEN — your Apify API token

Install:
    pip install apscheduler requests
"""

import os
import re
import json
import time
import requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from src.database import insert_posts, log_scrape

from src.constant import FAKE_KEYWORDS as FAKE_WORDS, HIGH_CONFIDENCE_FAKE_PHRASES
# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
ACTOR_ID    = "apify~instagram-scraper"
HEADERS     = {"Content-Type": "application/json"}

# Default hashtags to monitor — customise for your brand
HASHTAGS = [
    "yourbrand",
    "yourbrandshoes",
    "fakeyourbrand",
    "replicayourbrand",
    "fakesneakers",
    "replicasneakers",
    "sneakerreplica"
]

MAX_POSTS_PER_TAG = 50

# Import shared keyword lists from api_realtime to avoid duplicate logic.
# Fall back to a minimal inline list only if the import fails at load time.
try:
    from src.api_realtime import FAKE_KEYWORDS as FAKE_WORDS, HIGH_CONFIDENCE_FAKE_PHRASES
except Exception:
    FAKE_WORDS = [
        "replica", "first copy", "fake", "aaa", "1:1", "dupe",
        "dhgate", "weidian", "dm for price", "cheap jordan", "cheap nike"
    ]
    HIGH_CONFIDENCE_FAKE_PHRASES = FAKE_WORDS

# ══════════════════════════════════════════════════════════════════
# WEBSOCKET BROADCAST CALLBACK
# ══════════════════════════════════════════════════════════════════

_broadcast_fn = None

def set_broadcast(fn):
    """Register the async broadcast function from api_realtime.py."""
    global _broadcast_fn
    _broadcast_fn = fn


def broadcast(data: dict):
    """Fire-and-forget: send data to all WebSocket clients."""
    if _broadcast_fn is None:
        return
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast_fn(data))
    except Exception as e:
        print(f"Broadcast error: {e}")


# ══════════════════════════════════════════════════════════════════
# TEXT HELPERS
# ══════════════════════════════════════════════════════════════════

def is_english(text: str) -> bool:
    if not text or len(text) < 5:
        return False
    return sum(1 for c in text if ord(c) < 128) / len(text) >= 0.6


def get_source_type(caption: str, input_url: str, final_score: float = 0.0) -> str:
    url_lower     = input_url.lower()
    caption_lower = caption.lower()
    if any(w in url_lower     for w in ["fake", "replica", "counterfeit"]):
        return "counterfeit"
    if any(w in caption_lower for w in FAKE_WORDS):
        return "counterfeit"
    # Use text score as fallback signal — threshold matches THRESHOLD_UNCERTAIN (0.4)
    if final_score >= 0.4:
        return "counterfeit"
    return "brand"


def parse_post(post: dict, input_url: str = "") -> dict | None:
    """Parse a raw Apify item into our standard post schema."""
    caption = post.get("caption") or post.get("text") or ""
    if not caption or len(caption.strip()) < 10:
        return None
    if not is_english(caption):
        return None

    # Score text first so source_type can use the score
    try:
        from src.api_realtime import get_text_score
        final_score = get_text_score(caption.strip())
    except Exception as e:
        print(f"Scoring error: {e}")
        final_score = 0.0

    source_type = get_source_type(caption, input_url or post.get("inputUrl", ""), final_score)

    return {
        "id":          str(post.get("id") or post.get("shortCode") or ""),
        "caption":     caption.strip(),
        "username":    post.get("ownerUsername") or post.get("username") or "",
        "likes":       int(post.get("likesCount") or post.get("likes") or 0),
        "comments":    int(post.get("commentsCount") or post.get("comments") or 0),
        "timestamp":   str(post.get("timestamp") or ""),
        "source_type": source_type,
        "source_url":  input_url or post.get("inputUrl", ""),
        "image_url":   post.get("displayUrl") or post.get("imageUrl") or "",
        "hashtags":    ", ".join(post.get("hashtags") or []),
        "platform":    "instagram",
        "final_score": final_score
    }


# ══════════════════════════════════════════════════════════════════
# APIFY API CALLS
# ══════════════════════════════════════════════════════════════════

def start_apify_run(hashtags: list, max_posts: int) -> str | None:
    """Start an Apify Instagram scraper run and return the run ID."""
    if not APIFY_TOKEN:
        print("APIFY_TOKEN not set")
        return None

    urls = [
        f"https://www.instagram.com/explore/tags/{tag.strip().lstrip('#')}/"
        for tag in hashtags
    ]

    payload = {
        "directUrls":    urls,
        "resultsType":   "posts",
        "resultsLimit":  max_posts,
        "addParentData": False,
        "searchType":    "hashtag"
    }

    try:
        resp = requests.post(
            f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs",
            json=payload,
            headers={**HEADERS, "Authorization": f"Bearer {APIFY_TOKEN}"},
            timeout=30
        )
        data   = resp.json()
        run_id = data.get("data", {}).get("id")
        if run_id:
            print(f"Apify run started: {run_id}")
        else:
            print(f"Apify start failed: {data}")
        return run_id
    except Exception as e:
        print(f"Apify start error: {e}")
        return None


def wait_for_run(run_id: str, timeout: int = 300) -> bool:
    """Poll until the Apify run completes (or times out)."""
    if not APIFY_TOKEN or not run_id:
        return False

    start = time.time()
    while time.time() - start < timeout:
        try:
            resp   = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                headers={"Authorization": f"Bearer {APIFY_TOKEN}"},
                timeout=15
            )
            status = resp.json().get("data", {}).get("status", "")
            print(f"  Run status: {status}")
            if status == "SUCCEEDED":
                return True
            if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                print(f"Run ended with status: {status}")
                return False
        except Exception as e:
            print(f"Status check error: {e}")
        time.sleep(10)
    return False


def fetch_run_results(run_id: str) -> list:
    """Download all items from the completed run's dataset."""
    if not APIFY_TOKEN or not run_id:
        return []
    try:
        resp  = requests.get(
            f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items",
            params={"token": APIFY_TOKEN, "format": "json", "limit": 1000},
            timeout=30
        )
        items = resp.json()
        if not isinstance(items, list):
            print(f"Unexpected Apify response type: {type(items)}")
            return []
        return items
    except Exception as e:
        print(f"Fetch results error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# MAIN SCRAPE FUNCTION
# ══════════════════════════════════════════════════════════════════

def run_scrape(
    hashtags:  list = None,
    max_posts: int  = MAX_POSTS_PER_TAG
) -> dict:
    """
    Run a full Apify scrape: start → wait → fetch → store → alert.
    Returns a result summary dict.
    """
    started_at = datetime.now().isoformat()
    hashtags   = hashtags or HASHTAGS

    print(f"\n[{started_at}] Starting Apify scrape")
    print(f"  Hashtags : {hashtags}")
    print(f"  Max posts: {max_posts}")

    run_id = start_apify_run(hashtags, max_posts)
    if not run_id:
        return {"status": "failed", "error": "Could not start Apify run"}

    log_scrape(run_id, "running", 0, started_at)

    print("  Waiting for run to complete...")
    if not wait_for_run(run_id, timeout=300):
        log_scrape(run_id, "failed", 0, started_at)
        return {"status": "failed", "run_id": run_id}

    items = fetch_run_results(run_id)
    print(f"  Fetched {len(items)} raw items")

    posts = []
    for item in items:
        parsed = parse_post(item, item.get("inputUrl", ""))
        if parsed and parsed["id"]:
            posts.append(parsed)

    added = insert_posts(posts)
    log_scrape(run_id, "success", added, started_at)
    print(f"  Added {added} new posts to database")

    # Trigger alerts for every new counterfeit post using its real computed score
    try:
        from src.alert_system import process_new_post_for_alerts, check_bulk_alert
        for post in posts:
            if post.get("source_type") == "counterfeit":
                score = post.get("final_score", 0.0)
                # Derive label from actual score using shared thresholds (0.6/0.4)
                if score >= 0.6:
                    label = "fake"
                elif score >= 0.4:
                    label = "uncertain"
                else:
                    label = "real"
                post["label"]      = label
                post["sentiment"]  = "neutral"
                post["fetched_at"] = datetime.now().isoformat()
                process_new_post_for_alerts(post)
        check_bulk_alert()
    except Exception as e:
        print(f"Alert processing error: {e}")

    # Notify WebSocket clients
    broadcast({
        "type":        "scrape_complete",
        "posts_added": added,
        "total":       len(posts),
        "timestamp":   datetime.now().isoformat()
    })

    return {
        "status":  "success",
        "run_id":  run_id,
        "fetched": len(items),
        "parsed":  len(posts),
        "added":   added
    }


# ══════════════════════════════════════════════════════════════════
# BACKGROUND SCHEDULER
# ══════════════════════════════════════════════════════════════════

_scheduler: BackgroundScheduler = None


def start_scheduler(interval_hours: int = 6):
    """Start the background auto-scraper. Safe to call multiple times."""
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        run_scrape,
        trigger="interval",
        hours=interval_hours,
        id="auto_scrape",
        replace_existing=True
    )
    _scheduler.start()
    print(f"Auto-scraper started — runs every {interval_hours} hours")


def stop_scheduler():
    """Shutdown the background scheduler gracefully."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        print("Scheduler stopped")
