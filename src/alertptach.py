from src.alert_system import (
    process_new_post_for_alerts,
    check_bulk_alert,
    get_alert_log,
    test_alerts
)
 
# Add this new endpoint anywhere in api_realtime.py:
@app.get("/alerts/log")
def alerts_log(limit: int = 20):
    return get_alert_log(limit)
 
 
# ════════════════════════════════════════════════════════════════
# CHANGE 2 — In src/apify_scraper.py
# Inside run_scrape(), after the insert_posts() call:
# ════════════════════════════════════════════════════════════════
 
# Replace this block (already done in the fixed apify_scraper.py):
from src.alert_system import process_new_post_for_alerts, check_bulk_alert
 
for post in posts:
    if post.get("source_type") == "counterfeit":
        post["final_score"] = 0.85
        post["label"]       = "fake"
        post["fetched_at"]  = datetime.now().isoformat()
        process_new_post_for_alerts(post)
 
check_bulk_alert()
 
 
# ════════════════════════════════════════════════════════════════
# CHANGE 3 — Fill in config at top of src/alert_system.py
# ════════════════════════════════════════════════════════════════
 
# Gmail:
GMAIL_SENDER   = "youremail@gmail.com"       # your Gmail
GMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"       # 16-char App Password
ALERT_EMAILS   = ["youremail@gmail.com"]     # recipients
 
# Twilio:
TWILIO_ACCOUNT_SID  = "ACxxxxxxxx..."
TWILIO_AUTH_TOKEN   = "your_token"
TWILIO_FROM_NUMBER  = "+1XXXXXXXXXX"
ALERT_PHONE_NUMBERS = ["+91XXXXXXXXXX"]
 
# Brand name shown in all alerts:
BRAND_NAME = "Your Brand Monitor"
 