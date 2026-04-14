# ── Scoring thresholds ───────────────────────────────────────────
THRESHOLD_FAKE      = 0.6   # score >= 0.6  → FAKE
THRESHOLD_UNCERTAIN = 0.4   # score >= 0.4  → UNCERTAIN
#                             score <  0.4  → REAL
 
# ── Keyword lists ────────────────────────────────────────────────
FAKE_KEYWORDS = [
    "replica", "copy", "fake", "first copy", "ua",
    "mirror quality", "aaa", "master copy",
    "dm", "dm now", "dm for price",
    "discount", "discounted", "cheap",
    "wholesale", "bulk",
    "imported", "factory outlet",
    "no bill", "without box"
]
 
HIGH_CONFIDENCE_FAKE_PHRASES = [
    "dm for price", "dm for details", "dm to order", "dm if interested",
    "dm me for", "message for price", "inbox for price",
    "replica", "first copy", "master copy", "1:1 quality",
    "aaa quality", "mirror quality", "factory outlet",
    "no bill", "without box", "dhgate", "weidian"
]
 
SUSPICIOUS_COMBOS = [
    ({"dm", "interested"},       0.35),
    ({"dm", "price"},            0.40),
    ({"dm", "buy"},              0.40),
    ({"dm", "order"},            0.35),
    ({"discount", "dm"},         0.35),
    ({"discounted", "dm"},       0.35),
    ({"cheap", "shoes"},         0.30),
    ({"cheap", "sneakers"},      0.30),
    ({"selling", "discounted"},  0.25),
    ({"selling", "cheap"},       0.25),
    ({"selling", "dm"},          0.30),
    ({"wholesale", "dm"},        0.40),
    ({"bulk", "dm"},             0.35),
]
 
# Used by database.py keyword fallback scorer
FAKE_KW = [
    "replica","rep ","reps ","firstcopy","first copy","replicakicks",
    "jordanrep","repjordan","dm for price","dm us","whatsapp",
    "aaa","1:1","1to1","dupe","copy shoes","cheap jordan","cheap nike",
    "order now","replicasneakers","repsneakers","dhgate","weidian",
    "fake shoes","fakeshoes","fake kicks","fakekicks","putian",
    "sneakerreplica","sneaker replica","fake jordan","fakejordan",
    "fake nike","fakenike","counterfeit","knockoff","knock off"
]
 