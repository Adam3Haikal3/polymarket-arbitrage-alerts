"""
Arbitrage Monitor: Real-Time Price Monitoring for Verified Dependent Pairs

Once dependent market pairs are verified (Tier 3), this module monitors
their prices in real-time to detect and alert on arbitrage opportunities.

Two types of opportunities are monitored:
1. REBALANCING: Within a single market, sum(YES) deviates from 1.0
2. COMBINATORIAL: Between dependent markets, S and S' prices diverge

The monitor polls the CLOB API for current prices and computes potential
profit. When an opportunity exceeds the configured thresholds, it emits
an alert with:
- The affected markets and conditions
- The exact positions to take (buy/sell YES/NO)
- The estimated profit per dollar invested
- The maximum profit given available liquidity
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from src.data_collector import Market, PolymarketCollector
from src.price_analyzer import check_rebalancing_arbitrage, check_combinatorial_arbitrage
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageAlert:
    """An alert emitted when an arbitrage opportunity is detected."""
    timestamp: datetime
    alert_type: str              # "rebalancing_long", "rebalancing_short", "combinatorial"
    market_ids: list             # Affected market IDs
    conditions: list             # Specific conditions involved
    profit_per_dollar: float     # Expected profit per $1 invested
    max_profit: float            # Maximum profit given liquidity
    strategy: str                # Human-readable strategy description
    prices: dict                 # Current prices at time of alert
    confidence: float            # How confident we are (based on price freshness, etc.)


class ArbitrageMonitor:
    """
    Monitors verified dependent market pairs for arbitrage opportunities.

    Usage:
        monitor = ArbitrageMonitor(collector)

        # Add verified dependent pairs
        monitor.add_market(market_a)
        monitor.add_dependent_pair(market_a, market_b, subsets=([0, 1], [0, 1]))

        # Start monitoring with callback
        monitor.run(on_alert=lambda alert: print(alert))
    """

    def __init__(
        self,
        collector: PolymarketCollector = None,
        min_margin: float = None,
        min_profit: float = None,
        poll_interval: float = None,
    ):
        self.collector = collector or PolymarketCollector()
        self.min_margin = min_margin or settings.MIN_ARBITRAGE_MARGIN
        self.min_profit = min_profit or settings.MIN_ABSOLUTE_PROFIT
        self.poll_interval = poll_interval or settings.MONITOR_POLL_INTERVAL

        # Monitored markets and pairs
        self.markets = {}              # market_id -> Market
        self.dependent_pairs = []      # List of (market_a, market_b, S, S') tuples
        self.single_markets = []       # Markets to check for rebalancing

    def add_market(self, market: Market):
        """Add a single market for rebalancing arbitrage monitoring."""
        self.markets[market.market_id] = market
        if market.condition_count > 1:
            self.single_markets.append(market)

    def add_dependent_pair(
        self,
        market_a: Market,
        market_b: Market,
        subsets: tuple,  # (S_indices_in_a, S_indices_in_b)
    ):
        """Add a verified dependent pair for combinatorial arbitrage monitoring."""
        self.markets[market_a.market_id] = market_a
        self.markets[market_b.market_id] = market_b
        self.dependent_pairs.append((market_a, market_b, subsets[0], subsets[1]))

    def check_once(self) -> list:
        """
        Run a single check cycle across all monitored markets and pairs.

        Returns:
            List of ArbitrageAlert objects for any opportunities found.
        """
        alerts = []

        # Collect all token IDs we need prices for
        all_token_ids = []
        for market in self.markets.values():
            for cond in market.conditions:
                all_token_ids.extend([cond.token_id_yes, cond.token_id_no])

        # Fetch current prices in batch
        prices = self.collector.fetch_prices(all_token_ids)

        # ── Check Rebalancing Arbitrage ─────────────
        for market in self.single_markets:
            alert = self._check_rebalancing(market, prices)
            if alert:
                alerts.append(alert)

        # ── Check Combinatorial Arbitrage ───────────
        for ma, mb, s_a, s_b in self.dependent_pairs:
            alert = self._check_combinatorial(ma, mb, s_a, s_b, prices)
            if alert:
                alerts.append(alert)

        return alerts

    def run(
        self,
        on_alert: Callable[[ArbitrageAlert], None] = None,
        max_iterations: int = None,
    ):
        """
        Run continuous monitoring loop.

        Args:
            on_alert: Callback function called for each alert
            max_iterations: Stop after this many iterations (None = forever)
        """
        iteration = 0
        logger.info(
            f"Starting monitor: {len(self.single_markets)} markets, "
            f"{len(self.dependent_pairs)} dependent pairs"
        )

        while max_iterations is None or iteration < max_iterations:
            try:
                alerts = self.check_once()

                for alert in alerts:
                    logger.info(
                        f"ALERT: {alert.alert_type} | "
                        f"Profit/Dollar: ${alert.profit_per_dollar:.4f} | "
                        f"Max Profit: ${alert.max_profit:.2f}"
                    )
                    if on_alert:
                        on_alert(alert)

                time.sleep(self.poll_interval)
                iteration += 1

            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(self.poll_interval * 5)  # Back off on error

    def _check_rebalancing(self, market: Market, prices: dict) -> Optional[ArbitrageAlert]:
        """Check for rebalancing arbitrage within a single market."""
        yes_prices = []
        undecided = True

        for cond in market.conditions:
            p = prices.get(cond.token_id_yes)
            if p is None:
                return None  # Can't check without prices

            if p > settings.MAX_DECIDED_PROBABILITY:
                undecided = False  # Market outcome likely known
            yes_prices.append(p)

        if not undecided:
            return None

        result = check_rebalancing_arbitrage(yes_prices, self.min_margin)

        if not result["has_opportunity"]:
            return None

        # Estimate max profit from available liquidity
        # (simplified — production would check order book depth)
        min_liquidity = market.total_liquidity / market.condition_count
        max_profit = result["profit_per_dollar"] * min_liquidity

        if max_profit < self.min_profit:
            return None

        strategy = (
            f"{'Buy' if result['type'] == 'long' else 'Sell'} all YES tokens. "
            f"Sum of YES = {result['sum_yes']:.4f}, "
            f"profit = ${result['profit_per_dollar']:.4f}/dollar"
        )

        return ArbitrageAlert(
            timestamp=datetime.utcnow(),
            alert_type=f"rebalancing_{result['type']}",
            market_ids=[market.market_id],
            conditions=[c.question for c in market.conditions],
            profit_per_dollar=result["profit_per_dollar"],
            max_profit=max_profit,
            strategy=strategy,
            prices={c.condition_id: prices.get(c.token_id_yes, 0)
                    for c in market.conditions},
            confidence=0.8 if all(p is not None for p in yes_prices) else 0.5,
        )

    def _check_combinatorial(
        self,
        ma: Market, mb: Market,
        s_a: list, s_b: list,
        prices: dict,
    ) -> Optional[ArbitrageAlert]:
        """Check for combinatorial arbitrage between dependent markets."""
        # Get YES prices for dependent subsets
        prices_sa = []
        prices_sb = []

        for i in s_a:
            if i < len(ma.conditions):
                p = prices.get(ma.conditions[i].token_id_yes)
                if p is None:
                    return None
                prices_sa.append(p)

        for i in s_b:
            if i < len(mb.conditions):
                p = prices.get(mb.conditions[i].token_id_yes)
                if p is None:
                    return None
                prices_sb.append(p)

        result = check_combinatorial_arbitrage(prices_sa, prices_sb, self.min_margin)

        if not result["has_opportunity"]:
            return None

        # Estimate max profit
        min_liquidity = min(
            ma.total_liquidity / max(ma.condition_count, 1),
            mb.total_liquidity / max(mb.condition_count, 1),
        )
        max_profit = result["profit_per_dollar"] * min_liquidity

        if max_profit < self.min_profit:
            return None

        strategy = (
            f"Combinatorial: {result['direction']}. "
            f"Sum(S_A) = {result['sum_s1']:.4f}, "
            f"Sum(S_B) = {result['sum_s2']:.4f}, "
            f"diff = ${result['profit_per_dollar']:.4f}"
        )

        return ArbitrageAlert(
            timestamp=datetime.utcnow(),
            alert_type="combinatorial",
            market_ids=[ma.market_id, mb.market_id],
            conditions=[
                f"M1: {[ma.conditions[i].question for i in s_a if i < len(ma.conditions)]}",
                f"M2: {[mb.conditions[i].question for i in s_b if i < len(mb.conditions)]}",
            ],
            profit_per_dollar=result["profit_per_dollar"],
            max_profit=max_profit,
            strategy=strategy,
            prices={"sum_s1": result["sum_s1"], "sum_s2": result["sum_s2"]},
            confidence=0.7,
        )
