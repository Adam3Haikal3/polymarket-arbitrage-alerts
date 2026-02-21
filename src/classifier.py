"""
Dependency Classifier: Tier 2 ML Scoring

This module trains and applies a binary classifier that predicts whether
a candidate market pair is truly dependent.

Training data sources:
1. POSITIVE EXAMPLES: The 13 dependent pairs from the paper (Section 5.2)
2. NEGATIVE EXAMPLES: Independent pairs from the same topic+date groups
3. AUGMENTED POSITIVES: Synthetic pairs constructed by:
   - Taking a known dependent pair and slightly modifying thresholds
   - Creating "obvious" dependent pairs from market structure
     (e.g., "Who wins state X?" paired with "State X margin of victory")

The classifier uses XGBoost for several reasons:
- Handles missing features gracefully (price data not always available)
- Good with imbalanced classes (few positive examples)
- Feature importance is interpretable
- Fast inference (thousands of pairs per second)

Active learning loop:
1. Train initial model on labeled data
2. Score all candidate pairs
3. Send highest-uncertainty pairs to human annotator or LLM for labeling
4. Add new labels, retrain
5. Repeat until convergence
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, roc_auc_score,
)
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
)

from config import settings

logger = logging.getLogger(__name__)


class DependencyClassifier:
    """
    Binary classifier for market pair dependency.

    Predicts P(dependent | features) for each candidate pair.
    Pairs scoring above DEPENDENCY_SCORE_THRESHOLD go to Tier 3 (LLM).

    Usage:
        classifier = DependencyClassifier()

        # Train on labeled data
        classifier.train(X_train, y_train)

        # Score new pairs
        scores = classifier.predict_proba(X_new)

        # Get high-confidence predictions
        dependent_pairs = classifier.predict(X_new, threshold=0.6)
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        """
        Args:
            model_type: "gradient_boosting" (default) or "random_forest"
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.threshold = settings.DEPENDENCY_SCORE_THRESHOLD
        self._build_model()

    def _build_model(self):
        """Initialize the underlying sklearn model."""
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=settings.CLASSIFIER_PARAMS.get("n_estimators", 200),
                max_depth=settings.CLASSIFIER_PARAMS.get("max_depth", 6),
                learning_rate=settings.CLASSIFIER_PARAMS.get("learning_rate", 0.1),
                min_samples_leaf=settings.CLASSIFIER_PARAMS.get("min_child_weight", 3),
                subsample=settings.CLASSIFIER_PARAMS.get("subsample", 0.8),
                max_features=settings.CLASSIFIER_PARAMS.get("colsample_bytree", 0.8),
                random_state=settings.CLASSIFIER_PARAMS.get("random_state", 42),
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=settings.CLASSIFIER_PARAMS.get("n_estimators", 200),
                max_depth=settings.CLASSIFIER_PARAMS.get("max_depth", 6),
                min_samples_leaf=settings.CLASSIFIER_PARAMS.get("min_child_weight", 3),
                max_features="sqrt",
                class_weight="balanced",  # Handle imbalance
                random_state=settings.CLASSIFIER_PARAMS.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: list = None,
        cv_folds: int = 5,
    ) -> dict:
        """
        Train the classifier on labeled market pair data.

        Args:
            X: Feature matrix (rows = pairs, columns = features)
            y: Binary labels (1 = dependent, 0 = independent)
            feature_names: Optional feature column names
            cv_folds: Number of cross-validation folds

        Returns:
            Dict with training metrics:
            - cv_auc: Cross-validated AUC-ROC
            - cv_ap: Cross-validated Average Precision
            - feature_importance: Dict of feature -> importance score
        """
        self.feature_names = feature_names or list(X.columns) if hasattr(X, 'columns') else None

        # Handle NaN values
        X_clean = self._impute(X)

        logger.info(
            f"Training on {len(X_clean)} pairs "
            f"({sum(y)} positive, {len(y) - sum(y)} negative)"
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=min(cv_folds, sum(y), len(y) - sum(y)),
                             shuffle=True, random_state=42)

        cv_metrics = {}
        try:
            auc_scores = cross_val_score(
                self.model, X_clean, y, cv=cv, scoring="roc_auc"
            )
            cv_metrics["cv_auc_mean"] = float(np.mean(auc_scores))
            cv_metrics["cv_auc_std"] = float(np.std(auc_scores))
            logger.info(f"CV AUC: {cv_metrics['cv_auc_mean']:.3f} ± {cv_metrics['cv_auc_std']:.3f}")
        except Exception as e:
            logger.warning(f"CV AUC failed: {e}")
            cv_metrics["cv_auc_mean"] = np.nan

        try:
            ap_scores = cross_val_score(
                self.model, X_clean, y, cv=cv, scoring="average_precision"
            )
            cv_metrics["cv_ap_mean"] = float(np.mean(ap_scores))
            cv_metrics["cv_ap_std"] = float(np.std(ap_scores))
        except Exception as e:
            logger.warning(f"CV AP failed: {e}")
            cv_metrics["cv_ap_mean"] = np.nan

        # Train final model on all data
        self.model.fit(X_clean, y)

        # Feature importance
        importance = self._get_feature_importance()
        cv_metrics["feature_importance"] = importance

        logger.info("Top 10 features:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {feat}: {imp:.4f}")

        return cv_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict dependency probability for each pair.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities P(dependent) for each row
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Align features with training columns
        X_aligned = self._align_features(X)
        X_clean = self._impute(X_aligned)
        return self.model.predict_proba(X_clean)[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = None,
    ) -> np.ndarray:
        """
        Predict binary dependency labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold (default from settings)

        Returns:
            Binary array (1 = dependent, 0 = independent)
        """
        threshold = threshold or self.threshold
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def get_uncertain_pairs(
        self,
        X: pd.DataFrame,
        n: int = 20,
    ) -> np.ndarray:
        """
        Active learning: find pairs the model is most uncertain about.

        These are the most valuable pairs to label next (human or LLM).
        Pairs near the decision boundary (probability ~0.5) have highest
        information gain when labeled.

        Args:
            X: Feature matrix
            n: Number of uncertain pairs to return

        Returns:
            Indices of the n most uncertain pairs
        """
        probas = self.predict_proba(X)
        uncertainty = np.abs(probas - 0.5)
        return np.argsort(uncertainty)[:n]

    def save(self, path: str = "models/dependency_classifier.pkl"):
        """Save trained model to disk."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "threshold": self.threshold,
                "model_type": self.model_type,
            }, f)
        logger.info(f"Model saved to {filepath}")

    def load(self, path: str = "models/dependency_classifier.pkl"):
        """Load trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.threshold = data["threshold"]
        self.model_type = data["model_type"]
        logger.info(f"Model loaded from {path}")

    def _impute(self, X) -> np.ndarray:
        """Handle missing values by filling with median."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X, dtype=float)
        # Replace NaN with column median
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return X

    def _align_features(self, X) -> pd.DataFrame:
        """Align prediction features with training features."""
        if self.feature_names is None:
            return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if isinstance(X, pd.DataFrame):
            # Add missing columns with 0, drop extra columns
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            return X[self.feature_names]
        else:
            return pd.DataFrame(X)

    def _get_feature_importance(self) -> dict:
        """Extract feature importance from trained model."""
        if self.feature_names is None:
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))


# ─────────────────────────────────────────────
# TRAINING DATA GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_training_data(
    n_positive: int = 100,
    n_negative: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic training data for bootstrapping the classifier.

    This creates realistic feature vectors based on the patterns observed
    in the paper's findings:

    POSITIVE (dependent) pairs tend to have:
    - High entity overlap (same people, locations, teams)
    - High text similarity
    - Same end date and topic
    - Threshold containment (one is more specific than the other)
    - High price correlation

    NEGATIVE (independent) pairs tend to have:
    - Lower entity overlap (different events)
    - Different event subtypes (presidential vs senate)
    - Lower price correlation
    - Different resolution sources

    This synthetic data is meant for initial training. The active learning
    loop will improve the model with real labeled data over time.
    """
    rng = np.random.RandomState(seed)

    features_list = []
    labels = []

    # ── Generate POSITIVE (dependent) pairs ─────
    for _ in range(n_positive):
        features_list.append({
            "embedding_cosine_sim": rng.beta(8, 2),          # High similarity
            "entity_jaccard": rng.beta(6, 2),                # High overlap
            "question_token_jaccard": rng.beta(5, 3),
            "same_end_date": 1.0,                            # Always same date
            "same_topic": 1.0,                               # Always same topic
            "shared_tags_count": rng.poisson(3),
            "resolution_source_overlap": rng.choice([0, 1], p=[0.3, 0.7]),
            "both_negrisk": rng.choice([0, 1], p=[0.5, 0.5]),
            "shared_person_count": rng.poisson(1.5),
            "shared_location_count": rng.poisson(1.0),
            "shared_org_count": rng.poisson(0.8),
            "threshold_containment": rng.choice([0, 1], p=[0.3, 0.7]),
            "negation_mismatch": rng.choice([0, 1], p=[0.6, 0.4]),
            "price_pearson_corr": rng.beta(7, 2) * 2 - 1,   # Skewed positive
            "price_abs_pearson_corr": rng.beta(7, 2),
            "price_granger_pvalue": rng.beta(2, 5),          # Low p-value (significant)
            "volume_correlation": rng.beta(5, 3),
            "price_cointegration_pvalue": rng.beta(2, 5),
            "condition_count_ratio": rng.beta(5, 2),
            "liquidity_ratio": rng.beta(4, 3),
        })
        labels.append(1)

    # ── Generate NEGATIVE (independent) pairs ───
    for _ in range(n_negative):
        features_list.append({
            "embedding_cosine_sim": rng.beta(3, 5),          # Lower similarity
            "entity_jaccard": rng.beta(2, 6),                # Low overlap
            "question_token_jaccard": rng.beta(2, 5),
            "same_end_date": rng.choice([0, 1], p=[0.3, 0.7]),
            "same_topic": rng.choice([0, 1], p=[0.2, 0.8]),  # Usually same (pre-filtered)
            "shared_tags_count": rng.poisson(1),
            "resolution_source_overlap": rng.choice([0, 1], p=[0.7, 0.3]),
            "both_negrisk": rng.choice([0, 1], p=[0.6, 0.4]),
            "shared_person_count": rng.poisson(0.3),
            "shared_location_count": rng.poisson(0.2),
            "shared_org_count": rng.poisson(0.3),
            "threshold_containment": rng.choice([0, 1], p=[0.8, 0.2]),
            "negation_mismatch": rng.choice([0, 1], p=[0.8, 0.2]),
            "price_pearson_corr": rng.normal(0, 0.3),        # Around zero
            "price_abs_pearson_corr": rng.beta(2, 5),
            "price_granger_pvalue": rng.beta(5, 2),          # High p-value (not significant)
            "volume_correlation": rng.beta(2, 5),
            "price_cointegration_pvalue": rng.beta(5, 2),
            "condition_count_ratio": rng.beta(3, 4),
            "liquidity_ratio": rng.beta(2, 5),
        })
        labels.append(0)

    df = pd.DataFrame(features_list)
    y = np.array(labels)

    # Shuffle
    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    y = y[idx]

    return df, y
