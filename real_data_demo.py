#!/usr/वें/env python3
import logging
from config import settings
from src.candidate_generator import CandidateGenerator
from src.entity_extractor import EntityExtractor
from src.classifier import DependencyClassifier, generate_synthetic_training_data
from src.feature_engineer import FeatureEngineer
from src.price_analyzer import check_rebalancing_arbitrage, check_combinatorial_arbitrage
from src.kaggle_loader import load_kaggle_markets
import kagglehub
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("real_data_demo")

def run():
    logger.info("=" * 70)
    logger.info("  REAL DATA DEMO: POLYMARKET ARBITRAGE DETECTOR")
    logger.info("=" * 70)

    # 1. Download and Load Data
    logger.info("\n📦 Loading Kaggle Dataset...")
    dataset_path = kagglehub.dataset_download("ismetsemedov/polymarket-prediction-markets")
    csv_path = os.path.join(dataset_path, "polymarket_markets.csv")
    
    # We load markets across all active categories to find arbitrage opportunities
    markets = load_kaggle_markets(csv_path, limit=300, topic_filter=None)
    logger.info(f"  Loaded {len(markets)} markets.")
    
    # 2. Candidate Generation (Tier 1)
    logger.info("\n" + "─" * 50)
    logger.info("🔍 TIER 1: Candidate Generation")
    logger.info("─" * 50)
    
    generator = CandidateGenerator(markets=markets, entity_extractor=EntityExtractor())
    candidates = generator.generate()
    
    logger.info(f"\n  Found {len(candidates)} candidate pairs.")
    for m1, m2, score in candidates[:5]:
        logger.info(f"  Score {score:.3f}: '{m1.question[:40]}...' × '{m2.question[:40]}...'")

    # 3. Classifier Scoring (Tier 2)
    logger.info("\n" + "─" * 50)
    logger.info("🤖 TIER 2: ML Classification")
    logger.info("─" * 50)
    
    # We use synthetic data to train the model quickly for the demo
    X_train, y_train = generate_synthetic_training_data(n_positive=200, n_negative=800)
    classifier = DependencyClassifier()
    classifier.train(X_train, y_train, feature_names=list(X_train.columns))
    
    feature_engineer = FeatureEngineer()
    import pandas as pd
    scored_pairs = []
    
    if candidates:
        for m1, m2, tier1_score in candidates[:20]: # Only score top 20 for speed
            features = feature_engineer.build_features(m1, m2)
            features["tier1_score"] = tier1_score
            scored_pairs.append((m1, m2, features))
            
        X_test = pd.DataFrame([f for _, _, f in scored_pairs])
        # Replace missing price correlation features with 0 (since we don't have historical prices here)
        X_test = X_test.fillna(0)
        
        # Ensure all columns match training data
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[X_train.columns]
        
        scores = classifier.predict_proba(X_test)
        
        logger.info(f"\n  Top ML Dependency Scores:")
        for (m1, m2, _), score in sorted(zip(scored_pairs, scores), key=lambda x: -x[1])[:5]:
            flag = "✅" if score >= 0.5 else "  "
            logger.info(f"  {flag} {score:.3f}: '{m1.question[:35]}...' × '{m2.question[:35]}...'")

    # 3.5 LLM Verification (Tier 3)
    logger.info("\n" + "─" * 50)
    logger.info("🧠 TIER 3: LLM Verification")
    logger.info("─" * 50)
    
    if scored_pairs:
        top_pair = sorted(zip(scored_pairs, scores), key=lambda x: -x[1])[0]
        m1, m2, ml_score = top_pair[0]
        logger.info(f"  Sending top candidate to Gemini for logical verification:")
        logger.info(f"  M1: {m1.question}")
        logger.info(f"  M2: {m2.question}")
        
        from src.llm_verifier import LLMVerifier
        # Initialize verifier with user's Gemini key
        verifier = LLMVerifier(api_key="AIzaSyDsK1c-JO3sTBiYUV6LNROU2-3aJMFv1Rc")
        
        result = verifier.verify_pair(m1, m2)
        
        if result["is_dependent"]:
            logger.info(f"\n  LLM Result: ✅ DEPENDENT")
        else:
            logger.info(f"\n  LLM Result: ❌ INDEPENDENT")
            
        logger.info(f"  Reasoning: {result['reasoning']}")
        logger.info(f"  Combinations found: {result['n_combinations']} (Max possible: {result['max_combinations']})")

    # 4. Arbitrage Checks
    logger.info("\n" + "─" * 50)
    logger.info("💰 ARBITRAGE DETECTION ON REAL DATA")
    logger.info("─" * 50)

    # Rebalancing Arbitrage Checks across all loaded markets
    found_rebalancing = 0
    for market in markets:
        if market.condition_count > 1:
            yes_prices = [c.price_yes for c in market.conditions]
            arb = check_rebalancing_arbitrage(yes_prices, threshold=0.01)
            
            if arb["has_opportunity"]:
                found_rebalancing += 1
                logger.info(f"\n  [Rebalancing Alert] Market: {market.question}")
                logger.info(f"    Prices: {[round(p, 4) for p in yes_prices]}")
                logger.info(f"    Sum: {arb['sum_yes']:.4f}")
                logger.info(f"    Action: {arb['type'].upper()} all tokens for {arb['profit_per_dollar']:.4f} profit per dollar.")
                
                if found_rebalancing >= 3:
                     break
                     
    if found_rebalancing == 0:
         logger.info("\n  No Rebalancing Arbitrage found in the current sample.")

    # Combinatorial Check (Using top scoring dependent pair from ML)
    if scored_pairs:
        top_pair = sorted(zip(scored_pairs, scores), key=lambda x: -x[1])[0]
        m1, m2, _ = top_pair[0]
        
        # We only want to look at the 'Yes' outcome prices (not the artificially added 'No' outcomes)
        m1_prices = [c.price_yes for c in m1.conditions if getattr(c, 'outcome', 'Yes') == 'Yes']
        m2_prices = [c.price_yes for c in m2.conditions if getattr(c, 'outcome', 'Yes') == 'Yes']
        
        # Simple heuristic for demo: compare the sum of prices between the two dependent markets
        comp_arb = check_combinatorial_arbitrage(m1_prices, m2_prices, threshold=0.03)
        
        logger.info(f"\n  [Combinatorial Arbitrage Check] Top Dependent Pair")
        logger.info(f"    Market 1: {m1.question[:50]}... -> Sum YES: {sum(m1_prices):.4f}")
        logger.info(f"    Market 2: {m2.question[:50]}... -> Sum YES: {sum(m2_prices):.4f}")
        
        if comp_arb['has_opportunity']:
           logger.info(f"    Alert: {comp_arb['direction']} for {comp_arb['profit_per_dollar']:.4f} profit.")
        else:
           logger.info(f"    No combinatorial arbitrage found for this pair.")

    logger.info("\n" + "=" * 70)
    logger.info("  REAL DATA DEMO COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    run()
