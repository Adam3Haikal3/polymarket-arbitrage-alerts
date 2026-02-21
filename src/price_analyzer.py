"""
Price Analyzer: Statistical Dependency Detection from Price Series

This is the module the paper DOESN'T have — and it's arguably the most powerful
signal for dependency detection.

Core insight: If two markets are semantically dependent (their outcomes are logically
linked), their token prices MUST be statistically dependent. When one market moves,
the other should move in a predictable direction.

Example: If "Trump wins Georgia" goes up, then "Trump wins Georgia by 2-3%" should
also go up (or at least not go down). The converse is not necessarily true — correlated
prices don't always mean semantic dependency — but UNCORRELATED prices in candidate
pairs strongly suggest independence.

Statistical tests used:
1. Pearson correlation: Linear relationship between price changes
2. Granger causality: Does one series help predict the other?
3. Cointegration (Engle-Granger): Do the series share a long-run equilibrium?
4. Mutual information: Non-linear dependency measure
5. Rolling correlation: Time-varying relationship strength

Each test produces a feature for the ML classifier. The combination of multiple
tests is much more robust than any single measure.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

logger = logging.getLogger(__name__)


class PriceAnalyzer:
    """
    Computes statistical dependency features between price series of market pairs.

    Usage:
        analyzer = PriceAnalyzer()
        features = analyzer.compute_features(
            prices_m1=pd.Series([0.6, 0.62, 0.58, ...]),
            prices_m2=pd.Series([0.3, 0.32, 0.28, ...]),
        )
        print(features["price_pearson_corr"])  # e.g., 0.87
    """

    def __init__(self, min_overlap: int = None, granger_max_lag: int = None):
        self.min_overlap = min_overlap or settings.MIN_OVERLAP_POINTS
        self.granger_max_lag = granger_max_lag or settings.GRANGER_MAX_LAG

    def compute_features(
        self,
        prices_m1: pd.Series,
        prices_m2: pd.Series,
    ) -> dict:
        """
        Compute all statistical dependency features between two price series.

        Args:
            prices_m1: Price series for market 1 (indexed by block/time)
            prices_m2: Price series for market 2 (same index)

        Returns:
            Dict of feature_name -> float for the ML classifier.
            Returns NaN for features that can't be computed (insufficient data).
        """
        features = {
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
        }

        # Align the two series on the same index
        aligned = pd.DataFrame({"m1": prices_m1, "m2": prices_m2}).dropna()

        if len(aligned) < self.min_overlap:
            logger.debug(
                f"Insufficient overlap: {len(aligned)} < {self.min_overlap}"
            )
            return features

        s1 = aligned["m1"].values
        s2 = aligned["m2"].values

        # ── Pearson Correlation ──────────────────────
        # Measures linear relationship between raw prices.
        # Dependent markets should have |corr| > 0.5 typically.
        try:
            corr, pval = stats.pearsonr(s1, s2)
            features["price_pearson_corr"] = corr
            features["price_abs_pearson_corr"] = abs(corr)
        except Exception:
            pass

        # ── Spearman Rank Correlation ────────────────
        # More robust to non-linear relationships and outliers.
        try:
            corr, pval = stats.spearmanr(s1, s2)
            features["price_spearman_corr"] = corr
        except Exception:
            pass

        # ── Returns Correlation ──────────────────────
        # Correlate price CHANGES rather than levels.
        # This is more meaningful for detecting co-movement.
        try:
            returns1 = np.diff(s1)
            returns2 = np.diff(s2)
            if len(returns1) > 10:
                corr, _ = stats.pearsonr(returns1, returns2)
                features["returns_correlation"] = corr
        except Exception:
            pass

        # ── Granger Causality ────────────────────────
        # Tests whether past values of one series help predict the other.
        # Low p-value = evidence of Granger-causal relationship.
        try:
            features["price_granger_pvalue"] = self._granger_test(s1, s2)
        except Exception:
            pass

        # ── Cointegration (Engle-Granger) ────────────
        # Tests whether two non-stationary series share a long-run equilibrium.
        # Dependent markets should be cointegrated — their spread should be mean-reverting.
        try:
            features["price_cointegration_pvalue"] = self._cointegration_test(s1, s2)
        except Exception:
            pass

        # ── Mutual Information ───────────────────────
        # Captures non-linear dependencies that correlation misses.
        try:
            features["price_mutual_info"] = self._mutual_information(s1, s2)
        except Exception:
            pass

        # ── Rolling Correlation ──────────────────────
        # Time-varying correlation shows if dependency is consistent or episodic.
        try:
            roll_corr = self._rolling_correlation(
                aligned["m1"], aligned["m2"],
                window=min(settings.PRICE_CORRELATION_WINDOW, len(aligned) // 3)
            )
            if len(roll_corr.dropna()) > 0:
                features["rolling_corr_mean"] = roll_corr.mean()
                features["rolling_corr_std"] = roll_corr.std()
                features["rolling_corr_max"] = roll_corr.abs().max()
        except Exception:
            pass

        # ── Price Spread Statistics ──────────────────
        # The spread between dependent markets should be bounded and mean-reverting.
        try:
            spread = s1 - s2
            features["price_spread_mean"] = np.mean(spread)
            features["price_spread_std"] = np.std(spread)
        except Exception:
            pass

        return features

    def _granger_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Simplified Granger causality test.

        Tests H0: past values of x do NOT help predict y.
        Returns the minimum p-value across tested lags.
        Low p-value suggests x Granger-causes y (evidence of dependency).

        Full implementation would use statsmodels.tsa.stattools.grangercausalitytests,
        but this is a lightweight version for the MVP.
        """
        n = len(x)
        if n < self.granger_max_lag + 10:
            return np.nan

        min_pvalue = 1.0

        for lag in range(1, self.granger_max_lag + 1):
            # Restricted model: y_t = a0 + a1*y_{t-1} + ... + aL*y_{t-L} + e
            # Unrestricted: y_t = a0 + a1*y_{t-1} + ... + b1*x_{t-1} + ... + e
            # F-test on the x coefficients

            Y = y[lag:]
            # Build lagged matrices
            X_restricted = np.column_stack([y[lag - i - 1: n - i - 1] for i in range(lag)])
            X_unrestricted = np.column_stack([
                *[y[lag - i - 1: n - i - 1] for i in range(lag)],
                *[x[lag - i - 1: n - i - 1] for i in range(lag)],
            ])

            # Add constant
            X_restricted = np.column_stack([np.ones(len(Y)), X_restricted])
            X_unrestricted = np.column_stack([np.ones(len(Y)), X_unrestricted])

            try:
                # OLS residuals
                _, res_r, _, _ = np.linalg.lstsq(X_restricted, Y, rcond=None)
                _, res_u, _, _ = np.linalg.lstsq(X_unrestricted, Y, rcond=None)

                rss_r = res_r[0] if len(res_r) > 0 else np.sum((Y - X_restricted @ np.linalg.lstsq(X_restricted, Y, rcond=None)[0]) ** 2)
                rss_u = res_u[0] if len(res_u) > 0 else np.sum((Y - X_unrestricted @ np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]) ** 2)

                df1 = lag
                df2 = len(Y) - X_unrestricted.shape[1]

                if rss_u > 0 and df2 > 0:
                    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                    pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
                    min_pvalue = min(min_pvalue, pvalue)
            except Exception:
                continue

        return min_pvalue

    def _cointegration_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Engle-Granger two-step cointegration test.

        Step 1: Regress y on x, get residuals
        Step 2: Test residuals for stationarity (ADF test approximation)

        Low p-value = evidence of cointegration (shared long-run equilibrium).
        """
        n = len(x)
        if n < 30:
            return np.nan

        # Step 1: OLS regression y = a + b*x + e
        X = np.column_stack([np.ones(n), x])
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ coeffs

        # Step 2: ADF test on residuals (simplified as AR(1) test)
        # H0: residuals have a unit root (no cointegration)
        r_lag = residuals[:-1]
        r_diff = np.diff(residuals)

        X_adf = np.column_stack([np.ones(len(r_lag)), r_lag])
        coeffs_adf = np.linalg.lstsq(X_adf, r_diff, rcond=None)[0]

        # The coefficient on r_lag should be negative and significant
        gamma = coeffs_adf[1]
        predictions = X_adf @ coeffs_adf
        rss = np.sum((r_diff - predictions) ** 2)
        se = np.sqrt(rss / (len(r_diff) - 2) * np.linalg.inv(X_adf.T @ X_adf)[1, 1])

        if se > 0:
            t_stat = gamma / se
            # Approximate p-value (ADF critical values differ from t-dist,
            # but this gives a directional signal)
            pvalue = 2 * stats.t.cdf(t_stat, df=len(r_diff) - 2)
            return pvalue

        return np.nan

    def _mutual_information(
        self, x: np.ndarray, y: np.ndarray, n_bins: int = 20
    ) -> float:
        """
        Estimate mutual information between two series using histogram binning.

        MI = sum_{x,y} p(x,y) * log(p(x,y) / (p(x)*p(y)))

        Higher MI = stronger dependency (linear or non-linear).
        """
        # Bin the data
        x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
        y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))

        # Joint and marginal histograms
        joint, _, _ = np.histogram2d(x_bins, y_bins, bins=n_bins)
        joint = joint / joint.sum()  # Normalize to probabilities

        # Marginals
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # Compute MI
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

        return mi

    def _rolling_correlation(
        self, s1: pd.Series, s2: pd.Series, window: int = 50
    ) -> pd.Series:
        """Compute rolling Pearson correlation."""
        return s1.rolling(window=window).corr(s2)


# ─────────────────────────────────────────────
# ARBITRAGE PRICE CHECKS
# ─────────────────────────────────────────────

def check_rebalancing_arbitrage(prices_yes: list, threshold: float = 0.02) -> dict:
    """
    Check for Market Rebalancing Arbitrage within a single market.

    Per Definition 3 in the paper:
    - Long opportunity if sum(YES prices) < 1
    - Short opportunity if sum(YES prices) > 1

    Args:
        prices_yes: List of YES token prices for all conditions in a market
        threshold: Minimum deviation from 1.0 to flag (default 2 cents)

    Returns:
        {
            "has_opportunity": bool,
            "type": "long" | "short" | "none",
            "profit_per_dollar": float,  # Absolute profit per $1 invested
            "sum_yes": float,
        }
    """
    sum_yes = sum(prices_yes)
    deviation = abs(1.0 - sum_yes)

    if deviation < threshold:
        return {
            "has_opportunity": False,
            "type": "none",
            "profit_per_dollar": 0.0,
            "sum_yes": sum_yes,
        }

    return {
        "has_opportunity": True,
        "type": "long" if sum_yes < 1.0 else "short",
        "profit_per_dollar": deviation,
        "sum_yes": sum_yes,
    }


def check_combinatorial_arbitrage(
    prices_s1: list,
    prices_s2: list,
    threshold: float = 0.02,
) -> dict:
    """
    Check for Combinatorial Arbitrage between dependent market subsets.

    Per Definition 4 in the paper:
    Given dependent subsets S ⊂ M1 and S' ⊂ M2, arbitrage exists if
    sum(YES prices in S) ≠ sum(YES prices in S').

    Args:
        prices_s1: YES prices for dependent subset of market 1
        prices_s2: YES prices for dependent subset of market 2
        threshold: Minimum price difference to flag

    Returns:
        {
            "has_opportunity": bool,
            "profit_per_dollar": float,
            "direction": str,  # Which side to buy/sell
            "sum_s1": float,
            "sum_s2": float,
        }
    """
    sum_s1 = sum(prices_s1)
    sum_s2 = sum(prices_s2)
    diff = abs(sum_s1 - sum_s2)

    if diff < threshold:
        return {
            "has_opportunity": False,
            "profit_per_dollar": 0.0,
            "direction": "none",
            "sum_s1": sum_s1,
            "sum_s2": sum_s2,
        }

    direction = "buy_s1_sell_s2" if sum_s1 < sum_s2 else "buy_s2_sell_s1"

    return {
        "has_opportunity": True,
        "profit_per_dollar": diff,
        "direction": direction,
        "sum_s1": sum_s1,
        "sum_s2": sum_s2,
    }
