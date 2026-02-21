#!/usr/bin/env python3
"""
Full Pipeline Demo: Polymarket Dependent Market Detector

This demo creates synthetic markets that mirror real Polymarket patterns
from the paper, then runs the complete 3-tier detection pipeline.

The synthetic markets include:
- Election winner + election margin (known dependent pair)
- Sports winner + sports score (known dependent pair)
- Balance of power combinations (known dependent pair)
- Unrelated markets in the same topic (negative examples)

Run:
    python demo.py
"""

import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ─── Setup logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

# ─── Local imports ──────────────────────────
from src.data_collector import Market, Condition
from src.entity_extractor import EntityExtractor
from src.feature_engineer import FeatureEngineer
from src.candidate_generator import CandidateGenerator
from src.classifier import DependencyClassifier, generate_synthetic_training_data
from src.price_analyzer import PriceAnalyzer, check_rebalancing_arbitrage, check_combinatorial_arbitrage
from config import settings


def create_synthetic_markets() -> list:
    """
    Create synthetic markets that mirror real Polymarket patterns.

    We create 3 known dependent pairs and several independent markets
    to test all tiers of the pipeline.
    """
    election_date = datetime(2024, 11, 5, tzinfo=timezone.utc)
    sports_date = datetime(2024, 10, 15, tzinfo=timezone.utc)

    markets = []

    # ════════════════════════════════════════════
    # DEPENDENT PAIR 1: Election Winner + Margin
    # (mirrors Pairs 5-11 from the paper's Table 7)
    # ════════════════════════════════════════════

    # Market A: Who wins Georgia?
    markets.append(Market(
        market_id="election_ga_winner",
        slug="georgia-presidential-election-winner",
        question="Will a Democrat win Georgia Presidential Election?",
        description="This market will resolve to 'Yes' if the Democratic candidate wins Georgia.",
        conditions=[
            Condition("c1", "Will a Democrat win Georgia Presidential Election?",
                      "", "tok_dem_ga_y", "tok_dem_ga_n", "Yes", 0.45, 0.55, 50000),
            Condition("c2", "Will a Republican win Georgia Presidential Election?",
                      "", "tok_rep_ga_y", "tok_rep_ga_n", "Yes", 0.48, 0.52, 55000),
            Condition("c3", "Will a candidate from another party win Georgia Presidential Election?",
                      "", "tok_oth_ga_y", "tok_oth_ga_n", "Yes", 0.02, 0.98, 1000),
        ],
        end_date=election_date,
        topic="Politics",
        tags=["Politics", "Elections", "Georgia"],
        is_negrisk=True,
        total_liquidity=106000,
        total_volume=500000,
    ))

    # Market B: Georgia margin of victory
    markets.append(Market(
        market_id="election_ga_margin",
        slug="georgia-presidential-election-margin",
        question="Will the Democratic candidate win Georgia by 0%-1.0%?",
        description="This market resolves based on the margin of victory in Georgia.",
        conditions=[
            Condition("c4", "Will the Democratic candidate win Georgia by 0%-1.0%?",
                      "", "tok_dem_ga_01_y", "tok_dem_ga_01_n", "Yes", 0.15, 0.85, 20000),
            Condition("c5", "Will the Democratic candidate win Georgia by 1.0%-2.0%?",
                      "", "tok_dem_ga_12_y", "tok_dem_ga_12_n", "Yes", 0.12, 0.88, 18000),
            Condition("c6", "Will the Democratic candidate win Georgia by 2.0%-3.0%?",
                      "", "tok_dem_ga_23_y", "tok_dem_ga_23_n", "Yes", 0.08, 0.92, 15000),
            Condition("c7", "Will the Democratic candidate win Georgia by 3.0%-4.0%?",
                      "", "tok_dem_ga_34_y", "tok_dem_ga_34_n", "Yes", 0.05, 0.95, 10000),
        ],
        end_date=election_date,
        topic="Politics",
        tags=["Politics", "Elections", "Georgia", "Margin"],
        is_negrisk=True,
        total_liquidity=63000,
        total_volume=250000,
    ))

    # ════════════════════════════════════════════
    # DEPENDENT PAIR 2: Balance of Power
    # (mirrors Pairs 2-4 from the paper's Table 7)
    # ════════════════════════════════════════════

    # Market C: Presidential election winner
    markets.append(Market(
        market_id="election_prez_winner",
        slug="2024-presidential-election-winner",
        question="Who will win the 2024 Presidential Election?",
        description="Resolves to the candidate who wins 270+ electoral votes.",
        conditions=[
            Condition("c8", "Will Donald Trump win the 2024 Presidential Election?",
                      "", "tok_trump_y", "tok_trump_n", "Yes", 0.55, 0.45, 500000),
            Condition("c9", "Will Kamala Harris win the 2024 Presidential Election?",
                      "", "tok_harris_y", "tok_harris_n", "Yes", 0.43, 0.57, 480000),
            Condition("c10", "Will any other candidate win the 2024 Presidential Election?",
                      "", "tok_other_y", "tok_other_n", "Yes", 0.02, 0.98, 5000),
        ],
        end_date=election_date,
        topic="Politics",
        tags=["Politics", "Elections", "Presidential"],
        is_negrisk=True,
        total_liquidity=985000,
        total_volume=3700000,
    ))

    # Market D: Balance of power
    markets.append(Market(
        market_id="election_balance_power",
        slug="2024-balance-of-power",
        question="2024 Balance of Power: R Prez R Senate R House",
        description="Resolves based on which party controls presidency, senate, and house.",
        conditions=[
            Condition("c11", "2024 Balance of Power: R Prez R Senate R House",
                      "", "tok_rrr_y", "tok_rrr_n", "Yes", 0.30, 0.70, 100000),
            Condition("c12", "2024 Balance of Power: R Prez R Senate D House",
                      "", "tok_rrd_y", "tok_rrd_n", "Yes", 0.20, 0.80, 80000),
            Condition("c13", "2024 Balance of Power: D Prez D Senate D House",
                      "", "tok_ddd_y", "tok_ddd_n", "Yes", 0.15, 0.85, 70000),
            Condition("c14", "2024 Balance of Power: D Prez, R Senate, R House",
                      "", "tok_drr_y", "tok_drr_n", "Yes", 0.10, 0.90, 60000),
        ],
        end_date=election_date,
        topic="Politics",
        tags=["Politics", "Elections", "Balance of Power"],
        is_negrisk=True,
        total_liquidity=310000,
        total_volume=1200000,
    ))

    # ════════════════════════════════════════════
    # DEPENDENT PAIR 3: Sports Winner + Score
    # ════════════════════════════════════════════

    markets.append(Market(
        market_id="sports_nba_winner",
        slug="celtics-vs-lakers-winner",
        question="Will the Boston Celtics defeat the Los Angeles Lakers?",
        description="Resolves Yes if Celtics win the game.",
        conditions=[
            Condition("c15", "Will the Boston Celtics defeat the Los Angeles Lakers?",
                      "", "tok_celtics_y", "tok_celtics_n", "Yes", 0.60, 0.40, 30000),
        ],
        end_date=sports_date,
        topic="Sports",
        tags=["Sports", "NBA", "Basketball"],
        is_negrisk=False,
        total_liquidity=30000,
        total_volume=80000,
    ))

    markets.append(Market(
        market_id="sports_nba_margin",
        slug="celtics-vs-lakers-margin",
        question="Will the Boston Celtics win by 10 or more points against the Lakers?",
        description="Resolves Yes if Celtics win by 10+ points.",
        conditions=[
            Condition("c16", "Will the Boston Celtics win by 10 or more points against the Lakers?",
                      "", "tok_celtics_10_y", "tok_celtics_10_n", "Yes", 0.25, 0.75, 15000),
        ],
        end_date=sports_date,
        topic="Sports",
        tags=["Sports", "NBA", "Basketball"],
        is_negrisk=False,
        total_liquidity=15000,
        total_volume=40000,
    ))

    # ════════════════════════════════════════════
    # INDEPENDENT MARKETS (same topic, same date)
    # ════════════════════════════════════════════

    markets.append(Market(
        market_id="election_nc_winner",
        slug="north-carolina-presidential-winner",
        question="Will a Democrat win North Carolina Presidential Election?",
        description="Resolves based on North Carolina presidential race result.",
        conditions=[
            Condition("c17", "Will a Democrat win North Carolina?",
                      "", "tok_dem_nc_y", "tok_dem_nc_n", "Yes", 0.38, 0.62, 40000),
            Condition("c18", "Will a Republican win North Carolina?",
                      "", "tok_rep_nc_y", "tok_rep_nc_n", "Yes", 0.60, 0.40, 42000),
        ],
        end_date=election_date,
        topic="Politics",
        tags=["Politics", "Elections", "North Carolina"],
        is_negrisk=True,
        total_liquidity=82000,
        total_volume=300000,
    ))

    # Crypto market (different topic entirely)
    markets.append(Market(
        market_id="crypto_btc_100k",
        slug="bitcoin-100k-2024",
        question="Will Bitcoin reach $100,000 in 2024?",
        description="Resolves Yes if BTC/USD reaches $100,000 on any major exchange.",
        conditions=[
            Condition("c19", "Will Bitcoin reach $100,000 in 2024?",
                      "", "tok_btc_y", "tok_btc_n", "Yes", 0.35, 0.65, 200000),
        ],
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        topic="Crypto",
        tags=["Crypto", "Bitcoin"],
        is_negrisk=False,
        total_liquidity=200000,
        total_volume=800000,
    ))

    # Senate election (same topic, same date, but different SUBTYPE)
    markets.append(Market(
        market_id="election_ga_senate",
        slug="georgia-senate-election",
        question="Will a Republican win Georgia US Senate Election?",
        description="Resolves based on Georgia Senate race.",
        conditions=[
            Condition("c20", "Will a Republican win Georgia US Senate Election?",
                      "", "tok_rep_ga_sen_y", "tok_rep_ga_sen_n", "Yes", 0.52, 0.48, 35000),
        ],
        end_date=election_date,
        topic="Politics",
        tags=["Politics", "Elections", "Georgia", "Senate"],
        is_negrisk=False,
        total_liquidity=35000,
        total_volume=100000,
    ))

    return markets


def generate_synthetic_prices(n_points: int = 200) -> dict:
    """Generate correlated price series for dependent markets."""
    rng = np.random.RandomState(42)

    # Base signal: Georgia election sentiment
    base_signal = np.cumsum(rng.normal(0, 0.01, n_points)) + 0.5
    base_signal = np.clip(base_signal, 0.01, 0.99)

    prices = {
        # Dependent: GA winner and GA margin should be correlated
        "election_ga_winner": pd.Series(base_signal),
        "election_ga_margin": pd.Series(base_signal * 0.3 + rng.normal(0, 0.02, n_points)),

        # Dependent: Presidential winner and balance of power
        "election_prez_winner": pd.Series(base_signal * 0.8 + rng.normal(0, 0.03, n_points) + 0.1),
        "election_balance_power": pd.Series(base_signal * 0.5 + rng.normal(0, 0.04, n_points) + 0.1),

        # Dependent: Sports winner and margin
        "sports_nba_winner": pd.Series(np.clip(
            np.cumsum(rng.normal(0, 0.02, n_points)) + 0.6, 0.01, 0.99
        )),
        "sports_nba_margin": pd.Series(np.clip(
            np.cumsum(rng.normal(0, 0.02, n_points)) + 0.25, 0.01, 0.99
        )),

        # Independent: NC election (different state, somewhat correlated to national)
        "election_nc_winner": pd.Series(np.clip(
            base_signal * 0.3 + rng.normal(0, 0.05, n_points) + 0.3, 0.01, 0.99
        )),

        # Independent: Bitcoin (completely uncorrelated)
        "crypto_btc_100k": pd.Series(np.clip(
            np.cumsum(rng.normal(0, 0.015, n_points)) + 0.35, 0.01, 0.99
        )),

        # Independent: GA Senate (same state but different race)
        "election_ga_senate": pd.Series(np.clip(
            base_signal * 0.4 + rng.normal(0, 0.04, n_points) + 0.2, 0.01, 0.99
        )),
    }

    return prices


def main():
    """Run the full pipeline demo."""
    logger.info("=" * 70)
    logger.info("  POLYMARKET DEPENDENT MARKET DETECTOR — MVP DEMO")
    logger.info("=" * 70)

    # ── Create synthetic data ───────────────────
    logger.info("\n📦 Creating synthetic markets...")
    markets = create_synthetic_markets()
    prices = generate_synthetic_prices()

    logger.info(f"  Created {len(markets)} synthetic markets:")
    for m in markets:
        logger.info(f"    [{m.topic}] {m.question[:60]}... ({m.condition_count} conditions)")

    # ── Tier 1: Candidate Generation ────────────
    logger.info("\n" + "─" * 50)
    logger.info("🔍 TIER 1: Candidate Generation (Structural Filtering)")
    logger.info("─" * 50)

    generator = CandidateGenerator(
        markets=markets,
        entity_extractor=EntityExtractor(),
    )
    candidates = generator.generate()

    logger.info(f"\n  Candidates found: {len(candidates)}")
    for m1, m2, score in candidates[:10]:
        logger.info(f"  Score {score:.3f}: '{m1.question[:40]}...' × '{m2.question[:40]}...'")

    # ── Tier 2: ML Classification ───────────────
    logger.info("\n" + "─" * 50)
    logger.info("🤖 TIER 2: ML Classification")
    logger.info("─" * 50)

    # Train classifier on synthetic data
    logger.info("  Training classifier on synthetic labeled data...")
    X_train, y_train = generate_synthetic_training_data(n_positive=200, n_negative=800)
    classifier = DependencyClassifier(model_type="gradient_boosting")
    metrics = classifier.train(X_train, y_train, feature_names=list(X_train.columns))

    logger.info(f"  CV AUC: {metrics.get('cv_auc_mean', 0):.3f} ± {metrics.get('cv_auc_std', 0):.3f}")

    # Build features and score candidates
    feature_engineer = FeatureEngineer()
    scored_pairs = []

    for m1, m2, tier1_score in candidates:
        # Build features including price correlation
        p1 = prices.get(m1.market_id)
        p2 = prices.get(m2.market_id)

        features = feature_engineer.build_features(
            m1, m2,
            prices_a=p1,
            prices_b=p2,
        )
        features["tier1_score"] = tier1_score

        scored_pairs.append((m1, m2, features))

    # Score with classifier
    if scored_pairs:
        X = pd.DataFrame([f for _, _, f in scored_pairs])
        scores = classifier.predict_proba(X)

        logger.info(f"\n  ML Dependency Scores:")
        for (m1, m2, _), score in sorted(zip(scored_pairs, scores), key=lambda x: -x[1]):
            flag = "✅" if score >= settings.DEPENDENCY_SCORE_THRESHOLD else "  "
            logger.info(
                f"  {flag} {score:.3f}: '{m1.question[:35]}...' × '{m2.question[:35]}...'"
            )

    # ── Feature Importance ──────────────────────
    logger.info("\n  Top 10 Feature Importances:")
    importance = metrics.get("feature_importance", {})
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {feat:35s} {imp:.4f}")

    # ── Arbitrage Detection Example ─────────────
    logger.info("\n" + "─" * 50)
    logger.info("💰 ARBITRAGE DETECTION EXAMPLE")
    logger.info("─" * 50)

    # Check rebalancing within the Georgia election winner market
    ga_market = markets[0]  # Georgia winner
    yes_prices = [c.price_yes for c in ga_market.conditions]
    arb = check_rebalancing_arbitrage(yes_prices)
    logger.info(f"\n  Market: {ga_market.question}")
    logger.info(f"  YES prices: {[f'{p:.2f}' for p in yes_prices]}")
    logger.info(f"  Sum: {sum(yes_prices):.4f}")
    logger.info(f"  Rebalancing Arbitrage: {arb}")

    # Check combinatorial arbitrage between GA Winner and GA Margin
    dem_ga_yes_price = ga_market.conditions[0].price_yes
    ga_margin_market = markets[1]  # Georgia margin
    margin_yes_prices = [c.price_yes for c in ga_margin_market.conditions]
    
    comp_arb = check_combinatorial_arbitrage([dem_ga_yes_price], margin_yes_prices)
    logger.info(f"\n  Combinatorial Check: GA Democrat Winner vs. Margins")
    logger.info(f"  Market A (Winner = Dem): {dem_ga_yes_price:.4f}")
    logger.info(f"  Market B (Margin = Dem): {sum(margin_yes_prices):.4f} (from {[f'{p:.2f}' for p in margin_yes_prices]})")
    logger.info(f"  Combinatorial Arbitrage: {comp_arb}")

    # ── Price Correlation Analysis ──────────────
    logger.info("\n" + "─" * 50)
    logger.info("📊 PRICE CORRELATION ANALYSIS")
    logger.info("─" * 50)

    price_analyzer = PriceAnalyzer()
    pairs_to_check = [
        ("election_ga_winner", "election_ga_margin", "DEPENDENT (same state, winner vs margin)"),
        ("election_prez_winner", "election_balance_power", "DEPENDENT (prez in balance of power)"),
        ("election_ga_winner", "crypto_btc_100k", "INDEPENDENT (different topics)"),
        ("election_ga_winner", "election_ga_senate", "AMBIGUOUS (same state, different race)"),
        ("sports_nba_winner", "sports_nba_margin", "DEPENDENT (same game, winner vs margin)"),
    ]

    for id1, id2, label in pairs_to_check:
        if id1 in prices and id2 in prices:
            feats = price_analyzer.compute_features(prices[id1], prices[id2])
            logger.info(f"\n  {label}")
            logger.info(f"    Pearson:  {feats['price_pearson_corr']:.3f}")
            logger.info(f"    Returns:  {feats['returns_correlation']:.3f}")
            logger.info(f"    Granger:  {feats['price_granger_pvalue']:.4f}")
            logger.info(f"    MI:       {feats['price_mutual_info']:.3f}")

    # ── Entity Extraction Demo ──────────────────
    logger.info("\n" + "─" * 50)
    logger.info("🏷️  ENTITY EXTRACTION DEMO")
    logger.info("─" * 50)

    extractor = EntityExtractor()
    demo_questions = [
        "Will the Democratic candidate win Georgia by 2.0%-3.0%?",
        "Will Donald Trump win the 2024 Presidential Election?",
        "Will the Boston Celtics win by 10 or more points against the Lakers?",
        "2024 Balance of Power: R Prez R Senate R House",
        "Will Bitcoin reach $100,000 in 2024?",
    ]

    for q in demo_questions:
        e = extractor.extract(q)
        logger.info(f"\n  Q: {q}")
        if e.persons:
            logger.info(f"    Persons: {e.persons}")
        if e.locations:
            logger.info(f"    Locations: {e.locations}")
        if e.organizations:
            logger.info(f"    Orgs: {e.organizations}")
        logger.info(f"    Event: {e.event_type}/{e.event_subtype}")
        logger.info(f"    Direction: {e.direction}")
        if e.threshold_min or e.threshold_max:
            logger.info(f"    Threshold: [{e.threshold_min}, {e.threshold_max}] {e.threshold_unit}")

    logger.info("\n" + "=" * 70)
    logger.info("  DEMO COMPLETE")
    logger.info("=" * 70)
    logger.info("\nNext steps for production:")
    logger.info("  1. Connect to Polymarket API (set API key in config/settings.py)")
    logger.info("  2. Fetch real markets: collector.fetch_all_markets()")
    logger.info("  3. Collect on-chain price history for correlation features")
    logger.info("  4. Run pipeline.run(markets, train_classifier=True)")
    logger.info("  5. Label uncertain pairs (active learning) to improve classifier")
    logger.info("  6. Start monitoring: pipeline.start_monitor(verified_pairs)")


if __name__ == "__main__":
    main()
