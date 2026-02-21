# Polymarket Dependent Market Detector — MVP

## Architecture Overview

This system detects **dependent markets** on Polymarket — pairs of markets whose
outcomes are logically linked, creating arbitrage opportunities. It replaces the
paper's brute-force LLM approach with a **three-tier pipeline** that is faster,
more scalable, and more accurate.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 1: CANDIDATE GENERATION                 │
│  Fast structural filters to reduce O(n²) pairs to ~1% candidates│
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Same Date   │→ │ Same Topic   │→ │  Entity Overlap    │    │
│  │  Filter      │  │ Filter       │  │  (NER + Thresholds)│    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ~500-2000 candidate pairs
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 2: ML CLASSIFIER                        │
│  Score each candidate pair with dependency probability          │
│                                                                 │
│  Features:                                                      │
│  ├── Text similarity (embedding cosine, Jaccard entities)       │
│  ├── Structural (shared tags, resolution source overlap)        │
│  ├── Price correlation (rolling Pearson, Granger causality)     │
│  ├── Volume correlation (trade activity co-movement)            │
│  └── Parsed logic (entity match, threshold containment)         │
│                                                                 │
│  Model: Gradient Boosted Trees (XGBoost) or Random Forest       │
│  Output: dependency_score ∈ [0, 1]                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ~50-200 high-score pairs
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 3: LLM VERIFICATION                     │
│  Enumerate joint outcome space for high-confidence pairs        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Send pair to LLM with structured prompt                 │   │
│  │  → Get valid_combinations JSON                           │   │
│  │  → Validate: each market has exactly 1 true condition    │   │
│  │  → Check: |V1 × V2| < n·m  →  DEPENDENT                 │   │
│  │  → Extract dependent subsets S, S' per Definition 4      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ~10-50 verified dependent pairs
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ARBITRAGE MONITOR                             │
│  Real-time price monitoring on verified dependent pairs         │
│  Alert when |Σ val(S) - Σ val(S')| > threshold                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Beats the Paper's Approach

| Aspect | Paper (LLM-only) | This MVP |
|--------|-------------------|----------|
| Pairs checked by LLM | 46,360 | ~50-200 |
| False positive rate | High (manual review needed) | Low (ML pre-filter) |
| Handles ambiguous markets | Poorly (reasoning loops) | Entity disambiguation |
| Uses price data for detection | No | Yes (correlation features) |
| Scalability | O(n²) LLM calls | O(n²) cheap filters → O(k) LLM calls |
| New market detection | Re-run everything | Incremental (score new vs existing) |

## Module Breakdown

```
polymarket-arbitrage-detector/
├── README.md
├── requirements.txt
├── config/
│   └── settings.py              # API keys, thresholds, model params
├── src/
│   ├── __init__.py
│   ├── data_collector.py        # Polymarket API + Polygon on-chain
│   ├── entity_extractor.py      # NLP: NER, threshold parsing, logic detection
│   ├── feature_engineer.py      # Build feature vectors for market pairs
│   ├── price_analyzer.py        # Statistical dependency from price series
│   ├── candidate_generator.py   # Tier 1: fast structural filtering
│   ├── classifier.py            # Tier 2: ML dependency classifier
│   ├── llm_verifier.py          # Tier 3: LLM joint-outcome enumeration
│   ├── arbitrage_monitor.py     # Real-time price monitoring + alerts
│   └── pipeline.py              # End-to-end orchestration
├── data/
│   └── synthetic_markets.json   # Demo data for testing
├── models/
│   └── .gitkeep                 # Trained model artifacts
└── demo.py                      # Full pipeline demo
```

## Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

## Data Sources

1. **Polymarket API** (`py-clob-client`): Market metadata, conditions, prices
2. **Polygon RPC** (Alchemy): On-chain trade history (OrderFilled, PositionSplit events)
3. **Kaggle**: Historical snapshots for backtesting
4. **Dune Analytics**: Aggregated metrics for validation
