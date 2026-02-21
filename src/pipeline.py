"""
Pipeline: End-to-End Orchestration

This module ties together all three tiers into a single runnable pipeline:

    Markets → Tier 1 (Filter) → Tier 2 (Classify) → Tier 3 (Verify) → Monitor

It manages the data flow between modules, handles state persistence,
and provides a clean API for running the full analysis or individual tiers.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_collector import Market, PolymarketCollector
from src.entity_extractor import EntityExtractor
from src.feature_engineer import FeatureEngineer, build_feature_dataframe
from src.price_analyzer import PriceAnalyzer
from src.candidate_generator import CandidateGenerator
from src.classifier import DependencyClassifier, generate_synthetic_training_data
from src.llm_verifier import LLMVerifier
from src.arbitrage_monitor import ArbitrageMonitor
from config import settings

logger = logging.getLogger(__name__)


class DependencyDetectionPipeline:
    """
    End-to-end pipeline for detecting dependent markets on Polymarket.

    Usage:
        # Full pipeline
        pipeline = DependencyDetectionPipeline()
        results = pipeline.run(markets)

        # Or step by step
        pipeline = DependencyDetectionPipeline()
        candidates = pipeline.tier1_filter(markets)
        scored = pipeline.tier2_classify(candidates)
        verified = pipeline.tier3_verify(scored)
        pipeline.start_monitor(verified)
    """

    def __init__(
        self,
        collector: PolymarketCollector = None,
        classifier_path: str = None,
    ):
        # Initialize all components
        self.collector = collector or PolymarketCollector()
        self.entity_extractor = EntityExtractor()
        self.feature_engineer = FeatureEngineer(
            entity_extractor=self.entity_extractor,
            price_analyzer=PriceAnalyzer(),
        )
        self.classifier = DependencyClassifier()
        self.llm_verifier = LLMVerifier()
        self.monitor = ArbitrageMonitor(self.collector)

        # Load pre-trained classifier if available
        if classifier_path and Path(classifier_path).exists():
            self.classifier.load(classifier_path)
            logger.info(f"Loaded pre-trained classifier from {classifier_path}")

        # State tracking
        self.tier1_candidates = []
        self.tier2_scored = []
        self.tier3_verified = []

    def run(
        self,
        markets: list = None,
        fetch_markets: bool = False,
        train_classifier: bool = True,
        verify_with_llm: bool = True,
        start_monitoring: bool = False,
    ) -> dict:
        """
        Run the full pipeline end-to-end.

        Args:
            markets: List of Market objects (or None to fetch from API)
            fetch_markets: If True and markets is None, fetch from Polymarket API
            train_classifier: If True, train classifier on synthetic data first
            verify_with_llm: If True, verify high-scoring pairs with LLM
            start_monitoring: If True, start real-time monitoring after verification

        Returns:
            {
                "tier1_candidates": int,      # Pairs passing structural filters
                "tier2_high_score": int,       # Pairs scoring above threshold
                "tier3_verified": int,         # Pairs confirmed dependent by LLM
                "dependent_pairs": list,       # Full details of dependent pairs
                "execution_time_seconds": float,
            }
        """
        start_time = datetime.utcnow()
        logger.info("=" * 60)
        logger.info("POLYMARKET DEPENDENCY DETECTION PIPELINE")
        logger.info("=" * 60)

        # ── Step 0: Get markets ─────────────────────
        if markets is None and fetch_markets:
            logger.info("Fetching markets from Polymarket API...")
            markets = self.collector.fetch_all_markets(limit=5000)

        if not markets:
            logger.error("No markets provided or fetched")
            return {"error": "No markets"}

        logger.info(f"Starting with {len(markets)} markets")

        # ── Step 1: Train classifier if needed ──────
        if train_classifier and not hasattr(self.classifier.model, 'classes_'):
            logger.info("Training classifier on synthetic data...")
            X_synth, y_synth = generate_synthetic_training_data()
            metrics = self.classifier.train(
                X_synth, y_synth,
                feature_names=list(X_synth.columns),
            )
            logger.info(f"Classifier trained. CV AUC: {metrics.get('cv_auc_mean', 'N/A')}")

        # ── Step 2: Tier 1 — Candidate Generation ───
        logger.info("\n--- TIER 1: Candidate Generation ---")
        self.tier1_candidates = self.tier1_filter(markets)
        logger.info(f"Tier 1 output: {len(self.tier1_candidates)} candidate pairs")

        if not self.tier1_candidates:
            logger.info("No candidates found after Tier 1 filtering")
            return {"tier1_candidates": 0, "tier2_high_score": 0, "tier3_verified": 0}

        # ── Step 3: Tier 2 — ML Classification ──────
        logger.info("\n--- TIER 2: ML Classification ---")
        self.tier2_scored = self.tier2_classify(self.tier1_candidates)
        high_score = [
            (m1, m2, score, features)
            for m1, m2, score, features in self.tier2_scored
            if score >= settings.DEPENDENCY_SCORE_THRESHOLD
        ]
        logger.info(
            f"Tier 2 output: {len(high_score)} pairs above threshold "
            f"(out of {len(self.tier2_scored)})"
        )

        # ── Step 4: Tier 3 — LLM Verification ──────
        verified_pairs = []
        if verify_with_llm and high_score:
            logger.info("\n--- TIER 3: LLM Verification ---")
            verified_pairs = self.tier3_verify(high_score)
            self.tier3_verified = verified_pairs
            logger.info(f"Tier 3 output: {len(verified_pairs)} verified dependent pairs")

        # ── Step 5: Start Monitoring ────────────────
        if start_monitoring and verified_pairs:
            logger.info("\n--- Starting Arbitrage Monitor ---")
            self.start_monitor(verified_pairs, markets)

        # ── Results ─────────────────────────────────
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        results = {
            "total_markets": len(markets),
            "tier1_candidates": len(self.tier1_candidates),
            "tier2_high_score": len(high_score),
            "tier3_verified": len(verified_pairs),
            "dependent_pairs": [
                {
                    "market_a": p["market_a"].question,
                    "market_b": p["market_b"].question,
                    "subsets": p.get("dependent_subsets"),
                    "reasoning": p.get("reasoning", ""),
                }
                for p in verified_pairs
            ],
            "execution_time_seconds": elapsed,
        }

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Markets analyzed: {len(markets)}")
        logger.info(f"  Tier 1 candidates: {len(self.tier1_candidates)}")
        logger.info(f"  Tier 2 high-score: {len(high_score)}")
        logger.info(f"  Tier 3 verified:   {len(verified_pairs)}")
        logger.info(f"{'=' * 60}")

        return results

    def tier1_filter(self, markets: list) -> list:
        """
        Tier 1: Fast structural filtering.

        Returns: List of (market_a, market_b, tier1_score) tuples
        """
        generator = CandidateGenerator(
            markets=markets,
            entity_extractor=self.entity_extractor,
        )
        return generator.generate()

    def tier2_classify(self, candidates: list) -> list:
        """
        Tier 2: ML classification of candidate pairs.

        Args:
            candidates: List of (market_a, market_b, tier1_score) from Tier 1

        Returns:
            List of (market_a, market_b, dependency_score, features) tuples
        """
        # Build features for each pair
        feature_dicts = []
        pair_markets = []

        for m1, m2, tier1_score in candidates:
            features = self.feature_engineer.build_features(m1, m2)
            features["tier1_score"] = tier1_score  # Include Tier 1 signal
            feature_dicts.append(features)
            pair_markets.append((m1, m2))

        if not feature_dicts:
            return []

        # Build DataFrame
        X = pd.DataFrame(feature_dicts)

        # Predict dependency scores
        scores = self.classifier.predict_proba(X)

        # Combine results
        results = []
        for (m1, m2), score, features in zip(pair_markets, scores, feature_dicts):
            results.append((m1, m2, float(score), features))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def tier3_verify(self, high_score_pairs: list) -> list:
        """
        Tier 3: LLM verification of high-scoring pairs.

        Args:
            high_score_pairs: List of (market_a, market_b, score, features)

        Returns:
            List of dicts with verification results for confirmed dependent pairs
        """
        verified = []

        for m1, m2, score, features in high_score_pairs:
            logger.info(
                f"Verifying: '{m1.question[:50]}...' vs '{m2.question[:50]}...' "
                f"(score: {score:.3f})"
            )

            result = self.llm_verifier.verify_pair(m1, m2)

            if result["is_dependent"]:
                verified.append({
                    "market_a": m1,
                    "market_b": m2,
                    "ml_score": score,
                    "dependent_subsets": result["dependent_subsets"],
                    "reasoning": result["reasoning"],
                    "n_valid_combinations": result["n_combinations"],
                    "max_combinations": result["max_combinations"],
                    "features": features,
                })
                logger.info(f"  → DEPENDENT (combinations: {result['n_combinations']}/{result['max_combinations']})")
            else:
                logger.info(f"  → Independent")

        return verified

    def start_monitor(self, verified_pairs: list, all_markets: list = None):
        """Start real-time arbitrage monitoring on verified pairs."""
        # Add all NegRisk markets for rebalancing checks
        if all_markets:
            for m in all_markets:
                if m.is_negrisk and m.condition_count > 1:
                    self.monitor.add_market(m)

        # Add verified dependent pairs for combinatorial checks
        for pair in verified_pairs:
            subsets = pair.get("dependent_subsets")
            if subsets:
                self.monitor.add_dependent_pair(
                    pair["market_a"],
                    pair["market_b"],
                    subsets,
                )

        # Run with logging callback
        def log_alert(alert):
            logger.warning(
                f"🚨 ARBITRAGE: {alert.alert_type} | "
                f"${alert.profit_per_dollar:.4f}/$ | "
                f"Max: ${alert.max_profit:.2f} | "
                f"{alert.strategy}"
            )

        self.monitor.run(on_alert=log_alert, max_iterations=100)

    def save_results(self, results: dict, path: str = "data/pipeline_results.json"):
        """Save pipeline results to JSON."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects
        serializable = {
            k: v for k, v in results.items()
            if isinstance(v, (int, float, str, list, dict, bool))
        }

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
