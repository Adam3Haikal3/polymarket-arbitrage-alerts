"""
Configuration for the Polymarket Dependent Market Detector.

All thresholds, API endpoints, and model hyperparameters are centralized here.
Modify these values to tune the pipeline's sensitivity and performance.
"""

# ─────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────

POLYMARKET_API_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"

# Polygon RPC for on-chain data
POLYGON_RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"

# Conditional Token Contract (all Polymarket trades flow through here)
CONDITIONAL_TOKEN_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Event signatures for trade detection
EVENT_SIGS = {
    "OrderFilled": "0x...",       # Token traded for USDC
    "PositionSplit": "0x...",     # USDC locked, tokens minted
    "PositionsMerge": "0x...",    # Tokens burned, USDC withdrawn
}

# ─────────────────────────────────────────────
# TIER 1: CANDIDATE GENERATION THRESHOLDS
# ─────────────────────────────────────────────

# Maximum days apart for end_date to consider markets related
MAX_END_DATE_DELTA_DAYS = 1

# Minimum entity overlap ratio (Jaccard) to pass Tier 1
MIN_ENTITY_OVERLAP = 0.15

# Minimum embedding cosine similarity to pass Tier 1
MIN_EMBEDDING_SIMILARITY = 0.40

# Topics from Polymarket (used for grouping)
TOPICS = ["Politics", "Economy", "Technology", "Crypto", "Twitter", "Culture", "Sports"]

# ─────────────────────────────────────────────
# TIER 2: CLASSIFIER SETTINGS
# ─────────────────────────────────────────────

# Features to use in the classifier
FEATURE_COLUMNS = [
    # Text features
    "embedding_cosine_sim",
    "entity_jaccard",
    "question_token_jaccard",

    # Structural features
    "same_end_date",
    "same_topic",
    "shared_tags_count",
    "resolution_source_overlap",
    "both_negrisk",

    # Entity/logic features
    "shared_person_count",
    "shared_location_count",
    "shared_org_count",
    "threshold_containment",     # e.g., "wins by 2+" contained in "wins"
    "negation_mismatch",         # one has "not" / opposite framing

    # Price features (computed over trailing window)
    "price_pearson_corr",
    "price_abs_pearson_corr",
    "price_granger_pvalue",
    "volume_correlation",
    "price_cointegration_pvalue",

    # Market structure features
    "condition_count_ratio",     # n_conditions_m1 / n_conditions_m2
    "liquidity_ratio",
    "avg_spread_diff",
]

# Classifier hyperparameters (XGBoost)
CLASSIFIER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,      # Handle class imbalance (few dependent pairs)
    "eval_metric": "aucpr",
    "random_state": 42,
}

# Score threshold to pass to Tier 3 (LLM verification)
DEPENDENCY_SCORE_THRESHOLD = 0.60

# ─────────────────────────────────────────────
# TIER 3: LLM VERIFICATION
# ─────────────────────────────────────────────

LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # or any reasoning model
LLM_MAX_CONDITIONS_PER_MARKET = 5    # Reduce large markets to top-5 by volume
LLM_MAX_RETRIES = 3
LLM_TEMPERATURE = 0.1               # Low temp for deterministic reasoning

# ─────────────────────────────────────────────
# ARBITRAGE MONITORING
# ─────────────────────────────────────────────

# Minimum profit margin (on the dollar) to flag an opportunity
MIN_ARBITRAGE_MARGIN = 0.02          # 2 cents per dollar

# Minimum absolute profit to alert
MIN_ABSOLUTE_PROFIT = 5.0            # $5 minimum

# Price staleness: max blocks before a price is considered stale
MAX_PRICE_STALENESS_BLOCKS = 5000    # ~2.5 hours on Polygon

# Maximum probability to consider a market "undecided"
MAX_DECIDED_PROBABILITY = 0.95

# Monitoring poll interval (seconds)
MONITOR_POLL_INTERVAL = 2.0          # Polygon block time ~2s

# ─────────────────────────────────────────────
# PRICE ANALYSIS
# ─────────────────────────────────────────────

# Rolling window for price correlation (in blocks)
PRICE_CORRELATION_WINDOW = 500       # ~250 seconds on Polygon

# Minimum overlapping data points for valid correlation
MIN_OVERLAP_POINTS = 50

# Granger causality max lag (blocks)
GRANGER_MAX_LAG = 10
