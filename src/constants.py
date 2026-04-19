# ---------- THRESHOLDS ----------
THRESHOLD_FAKE = 0.6
THRESHOLD_UNCERTAIN = 0.4


# ---------- KEYWORDS ----------
FAKE_KEYWORDS = [
    "replica", "copy", "fake", "first copy", "ua",
    "mirror quality", "aaa", "master copy",
    "dm for price", "discount", "cheap",
    "wholesale", "bulk", "factory outlet",
    "no bill", "without box"
]


# ---------- HIGH CONFIDENCE ----------
HIGH_CONFIDENCE_FAKE_PHRASES = [
    "dm for price", "dm for details", "dm to order",
    "message for price", "inbox for price",
    "first copy", "master copy", "mirror quality",
    "factory outlet", "no bill", "without box"
]


# ---------- SUSPICIOUS COMBINATIONS ----------
SUSPICIOUS_COMBOS = [
    (("dm", "price"), 0.40),
    (("dm", "buy"), 0.40),
    (("discount", "dm"), 0.35),
    (("cheap", "shoes"), 0.30),
    (("wholesale", "dm"), 0.40),
]


# ---------- HYBRID WEIGHTS ----------
TFIDF_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35

HASHTAG_FAKE_BOOST = 0.20


# ---------- LABELS ----------
LABEL_REAL = "real"
LABEL_FAKE = "fake"
LABEL_UNCERTAIN = "uncertain"


# ---------- SYSTEM ----------
MAX_POSTS = 100
 
