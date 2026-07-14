"""
Granger Causality Testing for Binary Time Series

This module implements Granger causality tests specifically adapted
for binary sensor time series data.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class GrangerCausalityTest:
    """
    Granger causality test for binary time series sensor data.
    Tests whether past values of sensor X help predict sensor Y beyond what
    past values of Y already explain.
    """

    def __init__(self, max_lags: int = 5):
        """
        Initialize Granger causality test.

        Args:
            max_lags: Maximum number of lags to consider
        """
        self.max_lags = max_lags

    def test(self, x: np.ndarray, y: np.ndarray, lags: int | None = None) -> dict:
        """
        Perform Granger causality test.

        Args:
            x: Time series of sensor X (potential cause)
            y: Time series of sensor Y (potential effect)
            lags: Number of lags to use (if None, automatically selected)

        Returns:
            Dictionary with test results
        """
        x, y = self._validate_series(x, y)
        if lags is None:
            lags = self._select_optimal_lags(x, y)
        self._validate_lags(lags, len(y))

        # Restricted model: Y ~ lags of Y
        y_lagged = self._create_lag_matrix(y, lags, include_current=False)
        y_current = y[lags:]

        # Unrestricted model: Y ~ lags of Y + lags of X
        x_lagged = self._create_lag_matrix(x, lags, include_current=False)

        # Fit restricted model (Y predicted by its own lags only)
        restricted_ll = self._fit_binary_regression(y_lagged, y_current)

        # Fit unrestricted model (Y predicted by its own lags + X lags)
        combined_features = np.column_stack([y_lagged, x_lagged])
        unrestricted_ll = self._fit_binary_regression(combined_features, y_current)

        # Likelihood ratio test
        lr_stat = max(0.0, float(2 * (unrestricted_ll - restricted_ll)))
        p_value = float(1 - chi2.cdf(lr_stat, df=lags))

        return {
            "test_statistic": lr_stat,
            "p_value": p_value,
            "lags_used": lags,
            "causality_detected": p_value < 0.05,
            "restricted_ll": restricted_ll,
            "unrestricted_ll": unrestricted_ll,
        }

    def _validate_series(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize input series."""
        x_array = np.asarray(x, dtype=float).ravel()
        y_array = np.asarray(y, dtype=float).ravel()
        if len(x_array) != len(y_array):
            raise ValueError("x and y must have the same length")
        if len(x_array) < 3:
            raise ValueError("x and y must contain at least 3 observations")
        if np.isnan(x_array).any() or np.isnan(y_array).any():
            raise ValueError("x and y must not contain NaN values")
        for name, values in {"x": x_array, "y": y_array}.items():
            unique = set(np.unique(values))
            if not unique <= {0.0, 1.0}:
                raise ValueError(f"{name} must be binary with values 0 and 1")
        return x_array, y_array

    def _validate_lags(self, lags: int, n_obs: int) -> None:
        """Validate requested lag count."""
        if lags < 1:
            raise ValueError("lags must be at least 1")
        if lags >= n_obs:
            raise ValueError("lags must be smaller than the number of observations")

    def _create_lag_matrix(
        self, series: np.ndarray, lags: int, include_current: bool = False
    ) -> np.ndarray:
        """Create matrix of lagged variables."""
        n = len(series)
        lag_matrix = np.zeros((n - lags, lags + (1 if include_current else 0)))

        for row, t in enumerate(range(lags, n)):
            col = 0
            if include_current:
                lag_matrix[row, col] = series[t]
                col += 1
            for lag in range(1, lags + 1):
                lag_matrix[row, col] = series[t - lag]
                col += 1

        return lag_matrix

    def _fit_binary_regression(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit logistic regression and return log-likelihood."""
        try:
            if len(np.unique(y)) < 2:
                return self._constant_log_likelihood(y)

            # Add intercept
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

            # Fit using sklearn for numerical stability
            model = LogisticRegression(fit_intercept=False, max_iter=1000)
            model.fit(X_with_intercept, y)

            # Calculate log-likelihood manually
            linear_pred = X_with_intercept @ model.coef_.T
            probs = 1 / (1 + np.exp(-linear_pred.flatten()))

            # Avoid log(0) issues
            probs = np.clip(probs, 1e-15, 1 - 1e-15)

            log_likelihood = np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            return log_likelihood

        except (FloatingPointError, ValueError) as e:
            logger.warning("Error in binary regression: %s", e)
            return -np.inf

    def _constant_log_likelihood(self, y: np.ndarray) -> float:
        """Return Bernoulli log-likelihood for a constant target."""
        p = float(np.clip(y.mean(), 1e-15, 1 - 1e-15))
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _select_optimal_lags(self, x: np.ndarray, y: np.ndarray) -> int:
        """Select optimal number of lags using BIC."""
        best_lags = 1
        best_bic = np.inf

        for lags in range(1, min(self.max_lags + 1, len(y) // 4)):
            try:
                # Fit model with this number of lags
                y_lagged = self._create_lag_matrix(y, lags, include_current=False)
                x_lagged = self._create_lag_matrix(x, lags, include_current=False)
                y_current = y[lags:]

                combined_features = np.column_stack([y_lagged, x_lagged])
                log_likelihood = self._fit_binary_regression(
                    combined_features, y_current
                )

                n_params = combined_features.shape[1] + 1  # +1 for intercept
                n_obs = len(y_current)
                bic = -2 * log_likelihood + n_params * np.log(n_obs)

                if bic < best_bic:
                    best_bic = bic
                    best_lags = lags

            except (FloatingPointError, ValueError):
                continue

        return best_lags

    def test_all_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Test Granger causality for all sensor pairs.

        Args:
            data: DataFrame with sensor data

        Returns:
            DataFrame with causality test results for all pairs
        """
        sensors = data.columns.tolist()
        results = []

        for cause in sensors:
            for effect in sensors:
                if cause != effect:
                    try:
                        result = self.test(data[cause].values, data[effect].values)
                        results.append(
                            {
                                "cause": cause,
                                "effect": effect,
                                "test_statistic": result["test_statistic"],
                                "p_value": result["p_value"],
                                "causality_detected": result["causality_detected"],
                                "lags_used": result["lags_used"],
                            }
                        )
                    except (FloatingPointError, ValueError) as e:
                        logger.warning("Error testing %s -> %s: %s", cause, effect, e)
                        results.append(
                            {
                                "cause": cause,
                                "effect": effect,
                                "test_statistic": np.nan,
                                "p_value": np.nan,
                                "causality_detected": False,
                                "lags_used": np.nan,
                            }
                        )

        return pd.DataFrame(results)

    def create_causality_summary(self, causality_results: pd.DataFrame) -> dict:
        """
        Create summary statistics from causality test results.

        Args:
            causality_results: DataFrame from test_all_pairs()

        Returns:
            Dictionary with summary statistics
        """
        significant = causality_results[causality_results["causality_detected"]]

        # Count relationships
        total_pairs = len(causality_results)
        significant_pairs = len(significant)
        causality_rate = significant_pairs / total_pairs if total_pairs > 0 else 0

        # Identify most influential sensors
        if len(significant) > 0:
            cause_counts = (
                significant.groupby("cause").size().sort_values(ascending=False)
            )
            effect_counts = (
                significant.groupby("effect").size().sort_values(ascending=False)
            )

            top_causes = cause_counts.head(3).to_dict()
            top_effects = effect_counts.head(3).to_dict()
        else:
            top_causes = {}
            top_effects = {}

        # Average test statistics
        avg_test_stat = causality_results["test_statistic"].mean()
        avg_significant_stat = (
            significant["test_statistic"].mean() if len(significant) > 0 else 0
        )

        return {
            "total_pairs_tested": total_pairs,
            "significant_relationships": significant_pairs,
            "causality_rate": causality_rate,
            "average_test_statistic": avg_test_stat,
            "average_significant_statistic": avg_significant_stat,
            "top_causes": top_causes,
            "top_effects": top_effects,
            "bidirectional_relationships": self._find_bidirectional(significant),
        }

    def _find_bidirectional(
        self, significant_results: pd.DataFrame
    ) -> list[tuple[str, str]]:
        """Find bidirectional causal relationships."""
        bidirectional = []

        for _, row in significant_results.iterrows():
            # Check if reverse relationship also exists
            reverse = significant_results[
                (significant_results["cause"] == row["effect"])
                & (significant_results["effect"] == row["cause"])
            ]

            if len(reverse) > 0:
                pair = tuple(sorted([row["cause"], row["effect"]]))
                if pair not in bidirectional:
                    bidirectional.append(pair)

        return bidirectional
