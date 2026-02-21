"""
Candidate Generator: Tier 1 Fast Structural Filtering

The scalability bottleneck in the paper is O(n²) pairwise comparisons.
With 10K+ markets, that's 50M+ pairs — too many for LLM or even ML inference.

This module applies cheap structural filters to reduce the pair space by ~99%:

1. TIME FILTER: Markets must share the same end date (±1 day)
   - Rationale: Dependent markets resolve on the same event
   - Cost: O(n log n) with date sorting

2. TOPIC FILTER: Markets must be in the same topic category
   - Rationale: A politics market can't be dependent on a sports market
   - Cost: O(1) per pair

3. ENTITY OVERLAP FILTER: Markets must share at least one key entity
   - Rationale: Dependent markets discuss the same people/teams/locations
   - Cost: O(n) per market for extraction, O(1) per pair for comparison

4. EMBEDDING SIMILARITY FILTER: Minimum cosine similarity threshold
   - Rationale: Semantically unrelated markets can't be dependent
   - Cost: O(n) for embedding, O(1) per pair for dot product

The output is a list of candidate pairs, each with a "tier1_score" that
prioritizes which pairs to send to Tier 2 first.

Scaling analysis:
- Paper: 10K markets → 50M pairs → 46K LLM calls (after topic+date filter)
- This: 10K markets → 50M pairs → ~1K candidates (after 4 filters) → ~50 LLM calls
"""

import logging
from collections import defaultdict
from itertools import combinations
from typing import Tuple

import numpy as np

from src.data_collector import Market
from src.entity_extractor import EntityExtractor, compute_entity_overlap
from config import settings

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Generates candidate dependent market pairs using fast structural filters.

    Usage:
        generator = CandidateGenerator(markets)
        candidates = generator.generate()
        # candidates is a list of (market_a, market_b, tier1_score) sorted by score
    """

    def __init__(
        self,
        markets: list,
        entity_extractor: EntityExtractor = None,
    ):
        """
        Args:
            markets: List of Market objects to analyze
            entity_extractor: EntityExtractor instance (or creates one)
        """
        self.markets = markets
        self.extractor = entity_extractor or EntityExtractor()
        self._entity_cache = {}
        self._embedding_cache = {}

    def generate(
        self,
        max_date_delta_days: int = None,
        min_entity_overlap: float = None,
        min_embedding_sim: float = None,
    ) -> list:
        """
        Generate candidate pairs through cascading filters.

        Returns:
            List of (market_a, market_b, tier1_score) tuples,
            sorted by tier1_score descending (best candidates first).
        """
        max_date_delta = max_date_delta_days or settings.MAX_END_DATE_DELTA_DAYS
        min_entity = min_entity_overlap or settings.MIN_ENTITY_OVERLAP
        min_embed = min_embedding_sim or settings.MIN_EMBEDDING_SIMILARITY

        logger.info(f"Generating candidates from {len(self.markets)} markets")

        # ── Step 1: Group by (topic, end_date) ──────
        groups = self._group_by_topic_date(max_date_delta)
        total_pairs_before = sum(
            len(list(combinations(g, 2))) for g in groups.values()
        )
        logger.info(
            f"After topic+date grouping: {len(groups)} groups, "
            f"{total_pairs_before} potential pairs"
        )

        # ── Step 2: Pre-compute entities for all markets ──
        logger.info("Extracting entities for all markets...")
        for m in self.markets:
            self._get_entities(m)

        # ── Step 3: Filter within each group ────────
        candidates = []
        for group_key, group_markets in groups.items():
            for m1, m2 in combinations(group_markets, 2):
                score = self._score_pair(m1, m2, min_entity, min_embed)
                if score is not None:
                    candidates.append((m1, m2, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            f"Generated {len(candidates)} candidate pairs "
            f"(reduced from {total_pairs_before})"
        )

        return candidates

    def _group_by_topic_date(self, max_delta_days: int) -> dict:
        """
        Group markets by topic and end date.

        Markets in the same group share:
        - The same topic category
        - An end date within max_delta_days of each other

        We use the end_date rounded to the nearest day as the group key,
        which means markets within the same calendar day are grouped.
        For max_delta_days > 0, we also check adjacent days.
        """
        groups = defaultdict(list)

        for market in self.markets:
            if not market.end_date:
                # Markets without end dates go into a catch-all group
                groups[(market.topic, "no_date")].append(market)
                continue

            date_key = market.end_date.strftime("%Y-%m-%d")
            groups[(market.topic, date_key)].append(market)

        # If max_delta > 0, merge adjacent date groups within the same topic
        if max_delta_days > 0:
            groups = self._merge_adjacent_groups(groups, max_delta_days)

        # Filter out singleton groups (no pairs possible)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}

        return groups

    def _merge_adjacent_groups(self, groups: dict, max_delta: int) -> dict:
        """Merge groups that are within max_delta days of each other."""
        from datetime import datetime, timedelta

        # Group the date-based groups by topic
        topic_groups = defaultdict(list)
        for (topic, date_str), markets in groups.items():
            topic_groups[topic].append((date_str, markets))

        merged = {}
        for topic, date_groups in topic_groups.items():
            # Sort by date
            date_groups.sort(key=lambda x: x[0])

            # Merge adjacent dates
            current_key = None
            current_markets = []

            for date_str, markets in date_groups:
                if date_str == "no_date":
                    merged[(topic, "no_date")] = markets
                    continue

                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                if current_key is None:
                    current_key = date_str
                    current_markets = markets
                else:
                    prev_dt = datetime.strptime(current_key, "%Y-%m-%d")
                    if (dt - prev_dt).days <= max_delta:
                        current_markets.extend(markets)
                    else:
                        merged[(topic, current_key)] = current_markets
                        current_key = date_str
                        current_markets = markets

            if current_key:
                merged[(topic, current_key)] = current_markets

        return merged

    def _score_pair(
        self,
        m1: Market,
        m2: Market,
        min_entity_overlap: float,
        min_embedding_sim: float,
    ) -> float:
        """
        Score a candidate pair. Returns None if the pair doesn't pass filters.

        The score is a weighted combination of:
        - Entity overlap (Jaccard similarity of extracted entities)
        - Text similarity (embedding cosine or fallback)
        - Structural signals (shared tags, similar condition counts)

        Higher score = more likely to be dependent = higher priority for Tier 2.
        """
        # Get pre-computed entities
        e1 = self._get_entities(m1)
        e2 = self._get_entities(m2)

        # ── Entity overlap filter ───────────────────
        overlap = compute_entity_overlap(e1, e2)
        entity_jaccard = overlap["entity_jaccard"]

        if entity_jaccard < min_entity_overlap:
            # Exception: if they share the same event subtype and have NO entities,
            # they might still be related (e.g., generic "Will X happen?" markets)
            if not (overlap["same_event_type"] and overlap["same_event_subtype"]):
                return None

        # ── Event subtype mismatch filter ───────────
        # Different election subtypes (presidential vs senate) = not dependent
        if (
            e1.event_subtype and e2.event_subtype
            and e1.event_subtype != e2.event_subtype
        ):
            return None

        # ── Text similarity filter ──────────────────
        text_sim = self._text_similarity(m1, m2)
        if text_sim < min_embedding_sim and entity_jaccard < 0.3:
            return None

        # ── Compute composite score ─────────────────
        score = (
            0.30 * entity_jaccard
            + 0.25 * text_sim
            + 0.15 * overlap.get("threshold_containment", 0)
            + 0.10 * overlap.get("same_event_type", 0)
            + 0.10 * overlap.get("shared_person_count", 0) / max(
                len(e1.persons) + len(e2.persons), 1
            )
            + 0.10 * overlap.get("shared_location_count", 0) / max(
                len(e1.locations) + len(e2.locations), 1
            )
        )

        # Bonus for threshold containment + same entities
        # (strong signal: "wins" + "wins by X%" for same entity)
        if (
            overlap.get("threshold_containment", 0)
            and (overlap.get("shared_person_count", 0) > 0
                 or overlap.get("shared_location_count", 0) > 0)
        ):
            score += 0.20

        return score

    def _get_entities(self, market: Market) -> 'MarketEntities':
        """Extract entities with caching."""
        if market.market_id not in self._entity_cache:
            question = market.question
            if not question and market.conditions:
                question = market.conditions[0].question
            self._entity_cache[market.market_id] = self.extractor.extract(
                question or "", market.description or ""
            )
        return self._entity_cache[market.market_id]

    def _text_similarity(self, m1: Market, m2: Market) -> float:
        """
        Compute text similarity between market questions.

        Uses pre-computed embeddings if available, otherwise falls back
        to character n-gram overlap.
        """
        emb1 = self._embedding_cache.get(m1.market_id)
        emb2 = self._embedding_cache.get(m2.market_id)

        if emb1 is not None and emb2 is not None:
            return float(np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
            ))

        # Fallback: character n-gram Jaccard
        def ngrams(text, n=3):
            text = text.lower().strip()
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        q1 = m1.question or ""
        q2 = m2.question or ""
        ng1 = ngrams(q1)
        ng2 = ngrams(q2)
        union = ng1 | ng2
        return len(ng1 & ng2) / len(union) if union else 0.0

    def set_embeddings(self, embeddings: dict):
        """
        Pre-load embeddings for all markets.

        Args:
            embeddings: Dict of market_id -> np.ndarray embedding vector
        """
        self._embedding_cache = embeddings
        logger.info(f"Loaded {len(embeddings)} market embeddings")
