"""
Data Collector: Polymarket API + Polygon On-Chain Data

This module handles all data acquisition:
1. Market metadata from Polymarket's Gamma API (questions, conditions, tags, dates)
2. Real-time prices from the CLOB API
3. Historical trade data from Polygon on-chain events

The Polymarket data model:
- A MARKET is an event (e.g., "2024 Presidential Election Winner")
- A CONDITION is one possible outcome within a market (e.g., "Donald Trump")
- Each condition has YES/NO tokens that trade on the CLOB
- NegRisk markets have multiple conditions; single markets have one condition
- All trades settle through the Conditional Token Contract on Polygon

Data Sources:
- Gamma API: https://gamma-api.polymarket.com  (market metadata, no auth needed)
- CLOB API:  https://clob.polymarket.com        (order book, prices)
- Polygon:   Via Alchemy/Infura RPC              (on-chain trade history)
"""

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Condition:
    """A single outcome within a market."""
    condition_id: str
    question: str                # "Will Donald Trump win?"
    description: str             # Full resolution criteria
    token_id_yes: str
    token_id_no: str
    outcome: str                 # "Yes" / "No"
    price_yes: float = 0.0
    price_no: float = 0.0
    volume: float = 0.0
    winner: Optional[bool] = None


@dataclass
class Market:
    """A prediction market with one or more conditions."""
    market_id: str
    slug: str                    # URL-friendly identifier
    question: str                # "Who will win the 2024 Presidential Election?"
    description: str
    conditions: list = field(default_factory=list)  # List[Condition]
    end_date: Optional[datetime] = None
    topic: str = ""
    tags: list = field(default_factory=list)
    is_negrisk: bool = False
    neg_risk_market_id: str = ""
    resolution_source: str = ""
    active: bool = True
    total_liquidity: float = 0.0
    total_volume: float = 0.0

    @property
    def condition_count(self) -> int:
        return len(self.conditions)

    @property
    def condition_questions(self) -> list:
        return [c.question for c in self.conditions]


@dataclass
class Trade:
    """A single executed trade on-chain."""
    tx_hash: str
    block_number: int
    timestamp: int
    user_address: str
    condition_id: str
    token_id: str
    side: str                    # "YES" or "NO"
    action: str                  # "BUY", "SELL", "SPLIT", "MERGE"
    usdc_amount: float
    token_amount: float

    @property
    def price(self) -> float:
        if self.token_amount == 0:
            return 0.0
        return self.usdc_amount / self.token_amount


# ─────────────────────────────────────────────
# POLYMARKET API COLLECTOR
# ─────────────────────────────────────────────

class PolymarketCollector:
    """
    Collects market data from Polymarket's APIs.

    Usage:
        collector = PolymarketCollector()
        markets = collector.fetch_all_markets(limit=1000)
        prices = collector.fetch_prices(market_id="0x...")
    """

    def __init__(self, gamma_base=None, clob_base=None):
        self.gamma_base = gamma_base or settings.POLYMARKET_GAMMA_API
        self.clob_base = clob_base or settings.POLYMARKET_API_BASE
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketArbitrageDetector/1.0",
        })

    def fetch_all_markets(
        self,
        limit: int = 500,
        active_only: bool = False,
        topic: str = None,
        end_date_after: str = None,
        end_date_before: str = None,
    ) -> list:
        """
        Fetch markets from the Gamma API with pagination.

        The Gamma API returns market metadata including:
        - question, description, conditions
        - end_date_iso, tags, resolution sources
        - neg_risk flag and market groupings

        Args:
            limit: Max markets to fetch
            active_only: Only fetch active (unresolved) markets
            topic: Filter by topic string
            end_date_after: ISO date string, markets ending after this date
            end_date_before: ISO date string, markets ending before this date

        Returns:
            List[Market] objects with full metadata
        """
        markets = []
        offset = 0
        page_size = min(limit, 100)

        while offset < limit:
            params = {
                "limit": page_size,
                "offset": offset,
                "order": "end_date_iso",
                "ascending": "false",
            }
            if active_only:
                params["active"] = "true"
            if end_date_after:
                params["end_date_min"] = end_date_after
            if end_date_before:
                params["end_date_max"] = end_date_before

            try:
                resp = self.session.get(
                    f"{self.gamma_base}/markets", params=params, timeout=30
                )
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                for raw in batch:
                    market = self._parse_market(raw)
                    if topic and market.topic != topic:
                        continue
                    markets.append(market)

                offset += page_size
                time.sleep(0.2)  # Rate limiting

            except requests.RequestException as e:
                logger.error(f"API error at offset {offset}: {e}")
                break

        logger.info(f"Fetched {len(markets)} markets from Gamma API")
        return markets

    def fetch_market_by_id(self, market_id: str) -> Optional[Market]:
        """Fetch a single market by its condition_id or slug."""
        try:
            resp = self.session.get(
                f"{self.gamma_base}/markets/{market_id}", timeout=15
            )
            resp.raise_for_status()
            return self._parse_market(resp.json())
        except requests.RequestException as e:
            logger.error(f"Failed to fetch market {market_id}: {e}")
            return None

    def fetch_prices(self, token_ids: list) -> dict:
        """
        Fetch current prices for a list of token IDs from the CLOB.

        Returns:
            Dict mapping token_id -> current mid price
        """
        prices = {}
        for tid in token_ids:
            try:
                resp = self.session.get(
                    f"{self.clob_base}/price",
                    params={"token_id": tid},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                prices[tid] = float(data.get("price", 0))
            except Exception as e:
                logger.warning(f"Price fetch failed for {tid}: {e}")
                prices[tid] = None

        return prices

    def fetch_order_book(self, token_id: str) -> dict:
        """
        Fetch the full order book for a token.

        Returns:
            {"bids": [(price, size), ...], "asks": [(price, size), ...]}
        """
        try:
            resp = self.session.get(
                f"{self.clob_base}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "bids": [(float(b["price"]), float(b["size"])) for b in data.get("bids", [])],
                "asks": [(float(a["price"]), float(a["size"])) for a in data.get("asks", [])],
            }
        except Exception as e:
            logger.warning(f"Order book fetch failed for {token_id}: {e}")
            return {"bids": [], "asks": []}

    def _parse_market(self, raw: dict) -> Market:
        """Parse raw API JSON into a Market object."""
        conditions = []
        tokens = raw.get("tokens", [])

        # For NegRisk markets, conditions are grouped
        # For single markets, there's one YES/NO pair
        if raw.get("neg_risk", False):
            # NegRisk: each token pair is a separate condition
            for i in range(0, len(tokens), 2):
                if i + 1 < len(tokens):
                    conditions.append(Condition(
                        condition_id=raw.get("condition_id", ""),
                        question=tokens[i].get("outcome", ""),
                        description=raw.get("description", ""),
                        token_id_yes=tokens[i].get("token_id", ""),
                        token_id_no=tokens[i + 1].get("token_id", ""),
                        outcome=tokens[i].get("outcome", ""),
                        price_yes=float(tokens[i].get("price", 0)),
                        price_no=float(tokens[i + 1].get("price", 0)),
                        winner=tokens[i].get("winner"),
                    ))
        else:
            # Single condition: one YES token, one NO token
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), {})
            no_token = next((t for t in tokens if t.get("outcome") == "No"), {})
            conditions.append(Condition(
                condition_id=raw.get("condition_id", ""),
                question=raw.get("question", ""),
                description=raw.get("description", ""),
                token_id_yes=yes_token.get("token_id", ""),
                token_id_no=no_token.get("token_id", ""),
                outcome="Yes",
                price_yes=float(yes_token.get("price", 0)),
                price_no=float(no_token.get("price", 0)),
                winner=yes_token.get("winner"),
            ))

        # Parse end date
        end_date = None
        end_str = raw.get("end_date_iso", "")
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Market(
            market_id=raw.get("condition_id", raw.get("id", "")),
            slug=raw.get("market_slug", ""),
            question=raw.get("question", ""),
            description=raw.get("description", ""),
            conditions=conditions,
            end_date=end_date,
            topic=raw.get("assigned_topic", self._infer_topic(raw)),
            tags=raw.get("tags", []),
            is_negrisk=raw.get("neg_risk", False),
            neg_risk_market_id=raw.get("neg_risk_market_id", ""),
            resolution_source=raw.get("description", ""),  # Often embedded in description
            active=raw.get("active", True),
            total_liquidity=float(raw.get("liquidity", 0) or 0),
            total_volume=float(raw.get("volume", 0) or 0),
        )

    def _infer_topic(self, raw: dict) -> str:
        """Fallback topic inference from tags."""
        tags = [t.lower() for t in raw.get("tags", [])]
        for topic in settings.TOPICS:
            if topic.lower() in tags:
                return topic
        return "Unknown"


# ─────────────────────────────────────────────
# ON-CHAIN DATA COLLECTOR
# ─────────────────────────────────────────────

class OnChainCollector:
    """
    Collects historical trade data from the Polygon blockchain.

    All Polymarket trades flow through the Conditional Token Contract.
    We parse three event types:
    - OrderFilled:    Token traded for USDC (buy or sell)
    - PositionSplit:  USDC locked → YES + NO tokens minted
    - PositionsMerge: YES + NO tokens burned → USDC withdrawn

    Usage:
        collector = OnChainCollector(rpc_url="https://polygon-mainnet...")
        trades = collector.fetch_trades(
            token_ids=["0x...", "0x..."],
            from_block=50000000,
            to_block=55000000
        )
    """

    def __init__(self, rpc_url=None):
        self.rpc_url = rpc_url or settings.POLYGON_RPC_URL
        self.contract = settings.CONDITIONAL_TOKEN_CONTRACT

    def fetch_trades(
        self,
        token_ids: list,
        from_block: int,
        to_block: int,
        batch_size: int = 2000,
    ) -> pd.DataFrame:
        """
        Fetch all trades for given token IDs within a block range.

        Uses eth_getLogs to query events from the Conditional Token Contract,
        then parses them into Trade objects.

        Args:
            token_ids: List of token IDs to track
            from_block: Start block (inclusive)
            to_block: End block (inclusive)
            batch_size: Blocks per RPC batch (Alchemy limit ~2000)

        Returns:
            DataFrame with columns matching Trade fields
        """
        all_trades = []
        token_set = set(token_ids)

        for start in range(from_block, to_block, batch_size):
            end = min(start + batch_size - 1, to_block)

            try:
                # Query OrderFilled events
                logs = self._get_logs(start, end, topic0=None)

                for log in logs:
                    trade = self._parse_log(log)
                    if trade and trade.token_id in token_set:
                        all_trades.append(trade)

                time.sleep(0.1)  # Rate limiting for RPC

            except Exception as e:
                logger.error(f"RPC error blocks {start}-{end}: {e}")
                continue

        if not all_trades:
            return pd.DataFrame()

        df = pd.DataFrame([vars(t) for t in all_trades])
        df = df.sort_values("block_number").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} trades across {len(token_ids)} tokens")
        return df

    def _get_logs(self, from_block: int, to_block: int, topic0: str = None) -> list:
        """Make an eth_getLogs RPC call."""
        params = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "address": self.contract,
        }
        if topic0:
            params["topics"] = [topic0]

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [params],
            "id": 1,
        }

        resp = requests.post(self.rpc_url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if "error" in result:
            raise Exception(f"RPC error: {result['error']}")

        return result.get("result", [])

    def _parse_log(self, log: dict) -> Optional[Trade]:
        """
        Parse a raw event log into a Trade object.

        Event decoding depends on the event signature (topic[0]).
        This is a simplified parser — production would use web3.py or similar.
        """
        try:
            topics = log.get("topics", [])
            data = log.get("data", "0x")
            block = int(log.get("blockNumber", "0x0"), 16)
            tx_hash = log.get("transactionHash", "")

            # Simplified: in production, decode based on event ABI
            return Trade(
                tx_hash=tx_hash,
                block_number=block,
                timestamp=0,  # Would need block timestamp lookup
                user_address=topics[1][-40:] if len(topics) > 1 else "",
                condition_id=topics[2] if len(topics) > 2 else "",
                token_id=data[:66] if len(data) > 2 else "",
                side="YES",   # Would decode from event data
                action="BUY", # Would decode from event type
                usdc_amount=0.0,   # Would decode from event data
                token_amount=0.0,  # Would decode from event data
            )
        except Exception as e:
            logger.debug(f"Log parse error: {e}")
            return None


# ─────────────────────────────────────────────
# PRICE HISTORY BUILDER
# ─────────────────────────────────────────────

def build_price_series(trades_df: pd.DataFrame, block_interval: int = 1) -> pd.DataFrame:
    """
    Build a VWAP price series from raw trade data.

    For each block interval, compute the volume-weighted average price (VWAP)
    of executed trades. Carry forward the last known price for up to
    MAX_PRICE_STALENESS_BLOCKS if no trades occur.

    This replicates the paper's approach (Section 6) but as a reusable function.

    Args:
        trades_df: DataFrame with trade data (block_number, usdc_amount, token_amount, side)
        block_interval: Number of blocks to aggregate over

    Returns:
        DataFrame indexed by block with VWAP columns per token/side
    """
    if trades_df.empty:
        return pd.DataFrame()

    # Compute per-trade price
    df = trades_df.copy()
    df["price"] = df["usdc_amount"] / df["token_amount"].replace(0, np.nan)

    # Bucket by block interval
    df["block_bucket"] = (df["block_number"] // block_interval) * block_interval

    # VWAP per bucket per token/side
    vwap = (
        df.groupby(["block_bucket", "token_id", "side"])
        .apply(lambda g: np.average(g["price"].dropna(), weights=g["token_amount"].dropna())
               if len(g["price"].dropna()) > 0 else np.nan)
        .reset_index(name="vwap")
    )

    # Pivot to get one column per token-side
    pivot = vwap.pivot_table(
        index="block_bucket",
        columns=["token_id", "side"],
        values="vwap",
    )

    # Forward-fill up to staleness limit
    max_ffill = settings.MAX_PRICE_STALENESS_BLOCKS // block_interval
    pivot = pivot.ffill(limit=max_ffill)

    return pivot
