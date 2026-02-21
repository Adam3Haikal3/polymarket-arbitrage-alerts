"""
Feature Engineer: Build Feature Vectors for Market Pairs

This module takes two Market objects and produces a single feature vector
that captures all signals of potential dependency. The features come from
four sources:

1. TEXT FEATURES: Embedding similarity, token overlap
2. STRUCTURAL FEATURES: Shared dates, tags, topics, market types
3. ENTITY/LOGIC FEATURES: From EntityExtractor — shared entities, thresholds
4. PRICE FEATURES: From PriceAnalyzer — correlation, causality, cointegration

The feature vector feeds into the Tier 2 ML classifier.

Design principle: Each feature should have a clear interpretation for why
it indicates dependency. This helps with model explainability and debugging.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data_collector import Market
from src.entity_extractor import EntityExtractor, MarketEntities, compute_entity_overlap
from src.price_analyzer import PriceAnalyzer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds feature vectors for market pairs.

    Usage:
        engineer = FeatureEngineer()
        features = engineer.build_features(market_a, market_b)
        # features is a dict with ~25 numerical features
    """

    def __init__(self, entity_extractor: EntityExtractor = None,
                 price_analyzer: PriceAnalyzer = None):
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.price_analyzer = price_analyzer or PriceAnalyzer()
        self._entity_cache = {}  # market_id -> MarketEntities
        self._embedding_cache = {}  # market_id -> np.array

    def build_features(
        self,
        market_a: Market,
        market_b: Market,
        prices_a: pd.Series = None,
        prices_b: pd.Series = None,
        embedding_a: np.ndarray = None,
        embedding_b: np.ndarray = None,
    ) -> dict:
        """
        Build a complete feature vector for a market pair.

        Args:
            market_a, market_b: The two markets to compare
            prices_a, prices_b: Optional price series for price features
            embedding_a, embedding_b: Optional pre-computed text embeddings

        Returns:
            Dict of feature_name -> float value
        """
        features = {}

        # ── 1. TEXT FEATURES ────────────────────────
        features.update(self._text_features(market_a, market_b, embedding_a, embedding_b))

        # ── 2. STRUCTURAL FEATURES ──────────────────
        features.update(self._structural_features(market_a, market_b))

        # ── 3. ENTITY/LOGIC FEATURES ────────────────
        features.update(self._entity_features(market_a, market_b))

        # ── 4. PRICE FEATURES ───────────────────────
        if prices_a is not None and prices_b is not None:
            features.update(self.price_analyzer.compute_features(prices_a, prices_b))
        else:
            # Fill price features with NaN if no price data
            features.update({
                "price_pearson_corr": np.nan,
                "price_abs_pearson_corr": np.nan,
                "price_spearman_corr": np.nan,
                "price_granger_pvalue": np.nan,
                "volume_correlation": np.nan,
                "price_cointegration_pvalue": np.nan,
                "price_mutual_info": np.nan,
                "rolling_corr_mean": np.nan,
                "rolling_corr_std": np.nan,
                "rolling_corr_max": np.nan,
                "returns_correlation": np.nan,
                "price_spread_mean": np.nan,
                "price_spread_std": np.nan,
            })

        return features

    def _text_features(
        self,
        m1: Market, m2: Market,
        emb1: np.ndarray = None, emb2: np.ndarray = None,
    ) -> dict:
        """
        Text-based similarity features.

        Embedding cosine similarity is the strongest single text feature.
        Token Jaccard catches cases where embeddings miss (e.g., proper nouns).
        """
        features = {}

        # Cosine similarity of text embeddings
        if emb1 is not None and emb2 is not None:
            cos_sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            )
            features["embedding_cosine_sim"] = float(cos_sim)
        else:
            # Fallback: simple token-level similarity
            features["embedding_cosine_sim"] = self._simple_text_similarity(
                m1.question, m2.question
            )

        # Token Jaccard on questions
        tokens1 = set(m1.question.lower().split())
        tokens2 = set(m2.question.lower().split())
        union = tokens1 | tokens2
        features["question_token_jaccard"] = (
            len(tokens1 & tokens2) / len(union) if union else 0.0
        )

        # Description overlap (if available)
        desc1 = set(m1.description.lower().split()[:100])  # Truncate for speed
        desc2 = set(m2.description.lower().split()[:100])
        union_desc = desc1 | desc2
        features["description_token_jaccard"] = (
            len(desc1 & desc2) / len(union_desc) if union_desc else 0.0
        )

        return features

    def _structural_features(self, m1: Market, m2: Market) -> dict:
        """
        Features from market metadata structure.

        These are cheap to compute and effective as pre-filters.
        """
        features = {}

        # Same end date (critical for event-linked markets)
        same_date = False
        if m1.end_date and m2.end_date:
            delta = abs((m1.end_date - m2.end_date).total_seconds())
            same_date = delta < 86400  # Within 24 hours
        features["same_end_date"] = float(same_date)

        # Same topic
        features["same_topic"] = float(
            m1.topic == m2.topic and m1.topic != "Unknown"
        )

        # Shared tags
        tags1 = set(t.lower() for t in m1.tags)
        tags2 = set(t.lower() for t in m2.tags)
        features["shared_tags_count"] = len(tags1 & tags2)

        # Both NegRisk (multi-condition markets are more likely to have dependencies)
        features["both_negrisk"] = float(m1.is_negrisk and m2.is_negrisk)
        features["either_negrisk"] = float(m1.is_negrisk or m2.is_negrisk)

        # Condition count ratio (similar complexity markets)
        max_c = max(m1.condition_count, m2.condition_count, 1)
        min_c = max(min(m1.condition_count, m2.condition_count), 1)
        features["condition_count_ratio"] = min_c / max_c

        # Liquidity ratio (similar size markets)
        max_l = max(m1.total_liquidity, m2.total_liquidity, 1)
        min_l = max(min(m1.total_liquidity, m2.total_liquidity), 1)
        features["liquidity_ratio"] = min_l / max_l

        # Resolution source overlap (markets using same source are more likely related)
        features["resolution_source_overlap"] = float(
            self._resolution_overlap(m1.resolution_source, m2.resolution_source)
        )

        return features

    def _entity_features(self, m1: Market, m2: Market) -> dict:
        """
        Features from NLP entity extraction.

        This is where we catch the distinctions the paper's LLM missed:
        - Presidential vs Senate elections
        - Popular vote vs Electoral College
        - Different states/locations
        """
        # Extract entities (with caching)
        e1 = self._get_entities(m1)
        e2 = self._get_entities(m2)

        # Compute overlap metrics
        overlap = compute_entity_overlap(e1, e2)

        # Add event type compatibility
        # Same event type but different subtypes = LIKELY NOT dependent
        # (e.g., presidential election vs senate election)
        subtype_mismatch = (
            e1.event_subtype != "" and e2.event_subtype != ""
            and e1.event_subtype != e2.event_subtype
        )
        overlap["event_subtype_mismatch"] = float(subtype_mismatch)

        return overlap

    def _get_entities(self, market: Market) -> MarketEntities:
        """Extract entities with caching."""
        if market.market_id not in self._entity_cache:
            # For NegRisk markets, use the overall market question
            # For single markets, use the condition question
            question = market.question
            if not question and market.conditions:
                question = market.conditions[0].question
            self._entity_cache[market.market_id] = self.entity_extractor.extract(
                question, market.description
            )
        return self._entity_cache[market.market_id]

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback text similarity when embeddings aren't available.
        Uses character n-gram overlap.
        """
        def ngrams(text, n=3):
            text = text.lower().strip()
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        ng1 = ngrams(text1)
        ng2 = ngrams(text2)
        union = ng1 | ng2
        if not union:
            return 0.0
        return len(ng1 & ng2) / len(union)

    def _resolution_overlap(self, res1: str, res2: str) -> bool:
        """Check if resolution descriptions reference the same sources."""
        # Simple keyword overlap on resolution source references
        # In practice, you'd parse URLs and source names
        words1 = set(res1.lower().split()[:50])
        words2 = set(res2.lower().split()[:50])
        # Check for common resolution keywords
        source_words = {"associated", "press", "ap", "reuters", "nyt", "bbc",
                        "espn", "fivethirtyeight", "538", "official", "results"}
        shared = (words1 & words2) - {"the", "a", "an", "to", "of", "in", "and", "or"}
        return len(shared & source_words) > 0 or len(shared) > 10


def build_feature_dataframe(
    feature_dicts: list,
    pair_ids: list = None,
) -> pd.DataFrame:
    """
    Convert a list of feature dicts into a DataFrame ready for the classifier.

    Args:
        feature_dicts: List of dicts from FeatureEngineer.build_features()
        pair_ids: Optional list of (market_id_1, market_id_2) tuples

    Returns:
        DataFrame where each row is a market pair and columns are features
    """
    df = pd.DataFrame(feature_dicts)

    if pair_ids:
        df["market_id_1"] = [p[0] for p in pair_ids]
        df["market_id_2"] = [p[1] for p in pair_ids]

    # Handle NaN values - fill with column median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        if np.isnan(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    return df
