"""Base Bernoulli Autoregressive Model for sensor data.

This module contains the core single-sensor model implementation
based on Gillam et al. (2022).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm
from typing import Dict, List
from ...utils.data_io import SensorDataset
import logging

logger = logging.getLogger(__name__)


class BernoulliAutoregressiveModel:
    """Bernoulli Autoregressive Model for sensor data.

    Described in "Modeling and forecasting of at home activity in older adults
    using passive sensor technology" by Gillam et al. (2022).

    This model predicts the probability of a sensor triggering in 15-minute
    intervals based on:

    - Previous activations of the same sensor (autoregressive term)
    - Other sensors in the household
    - Daily seasonal patterns
    """

    def __init__(self, sensor_names: List[str], target_sensor: str):
        """
        Initialize the model.

        Args:
            sensor_names: List of all sensor names in the dataset
            target_sensor: Name of the sensor to predict
        """
        self.sensor_names = sensor_names
        self.target_sensor = target_sensor
        self.other_sensors = [s for s in sensor_names if s != target_sensor]

        # Model parameters (will be estimated)
        self.params = {}
        self.selected_sensors = []  # Sensors selected through model selection

        # Data storage
        self.y = None  # Target sensor data
        self.X_other = None  # Other sensor data

        # Constants
        self.INTERVALS_PER_DAY = 96  # 15-minute intervals in 24 hours

    def _log_likelihood(self, param_vector: np.ndarray) -> float:
        """
        Compute the negative log-likelihood for optimization.

        Args:
            param_vector: Flattened parameter vector

        Returns:
            Negative log-likelihood
        """
        try:
            # Convert parameter vector to dictionary
            params = self._vector_to_params(param_vector)

            T = len(self.y)
            ar_cache = np.zeros(T)
            if T > 1:
                ar_cache[1:] = lfilter(
                    [1.0],
                    [1.0, -params["phi_b"]],
                    self.y[:-1],
                )

            sensor_cache = {s: np.zeros(T) for s in self.selected_sensors}
            for s in self.selected_sensors:
                idx = self.other_sensors.index(s)
                psi_key = f"psi_{s}"
                if T > 1:
                    sensor_cache[s][1:] = lfilter(
                        [1.0],
                        [1.0, -params[psi_key]],
                        self.X_other[:-1, idx],
                    )

            seasonal = np.zeros(T)
            if "pi_d" in params and "phi_d" in params:
                for t in range(self.INTERVALS_PER_DAY, T):
                    prev_day = t - self.INTERVALS_PER_DAY
                    start = max(0, prev_day - 1)
                    end = min(len(self.y), prev_day + 2)
                    window_max = np.max(self.y[start:end])
                    seasonal[t] = (
                        params["phi_d"] * seasonal[t - self.INTERVALS_PER_DAY]
                        + window_max
                    )

            delta = params["a"] + params["pi_b"] * ar_cache
            if self.selected_sensors:
                stack = np.vstack(
                    [
                        params[f"tau_{s}"] * sensor_cache[s]
                        for s in self.selected_sensors
                    ]
                )
                delta += stack.sum(axis=0)
            if "pi_d" in params and "phi_d" in params:
                delta += params["pi_d"] * seasonal

            delta = np.clip(delta, -500, 500)
            log_lik = np.sum(self.y * delta - np.log1p(np.exp(delta)))
            return -log_lik  # Return negative for minimization

        except (OverflowError, ValueError, RuntimeWarning):
            return 1e10  # Return large value for invalid parameters

    def _vector_to_params(self, param_vector: np.ndarray) -> Dict:
        """Convert parameter vector to dictionary."""
        params = {}
        idx = 0

        # Always include intercept and autoregressive terms
        params["a"] = param_vector[idx]
        idx += 1
        params["pi_b"] = param_vector[idx]
        idx += 1
        params["phi_b"] = param_vector[idx]
        idx += 1

        # Other sensor parameters
        for sensor in self.selected_sensors:
            params[f"tau_{sensor}"] = param_vector[idx]
            idx += 1
            params[f"psi_{sensor}"] = param_vector[idx]
            idx += 1

        # Seasonal parameters (if included)
        if idx < len(param_vector):
            params["pi_d"] = param_vector[idx]
            idx += 1
            params["phi_d"] = param_vector[idx]
            idx += 1

        return params

    def _params_to_vector(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to vector."""
        vector = [params["a"], params["pi_b"], params["phi_b"]]

        for sensor in self.selected_sensors:
            vector.extend([params[f"tau_{sensor}"], params[f"psi_{sensor}"]])

        if "pi_d" in params and "phi_d" in params:
            vector.extend([params["pi_d"], params["phi_d"]])

        return np.array(vector)

    def _stepwise_selection(
        self, data: pd.DataFrame, max_sensors: int = 5
    ) -> List[str]:
        """
        Perform stepwise forward selection using BIC criterion.

        Args:
            data: DataFrame containing sensor data
            max_sensors: Maximum number of sensors to select

        Returns:
            List of selected sensor names
        """
        logger.info("Performing stepwise sensor selection...")

        self.y = data[self.target_sensor].values
        self.X_other = data[self.other_sensors].values

        selected = []
        available = self.other_sensors.copy()

        # Always start with autoregressive term only
        current_model_bic = self._fit_model_subset(  # noqa: E501
            selected, include_seasonal=False
        )
        logger.info(f"Baseline model BIC: {current_model_bic:.2f}")

        while len(selected) < max_sensors and available:
            best_sensor = None
            best_new_bic = float("inf")

            # Try adding each available sensor
            for sensor in available:
                test_sensors = selected + [sensor]

                # Try without seasonal term
                bic_no_seasonal = self._fit_model_subset(
                    test_sensors, include_seasonal=False
                )

                # Try with seasonal term if we haven't already included it
                bic_seasonal = self._fit_model_subset(
                    test_sensors, include_seasonal=True
                )

                test_bic = min(bic_no_seasonal, bic_seasonal)

                if test_bic < best_new_bic:
                    best_new_bic = test_bic
                    best_sensor = sensor

            # Add sensor if it improves BIC
            if best_new_bic < current_model_bic:
                selected.append(best_sensor)
                available.remove(best_sensor)
                current_model_bic = best_new_bic
                logger.info(  # noqa: E501
                    f"Added sensor '{best_sensor}', new BIC: {current_model_bic:.2f}"  # noqa: E501
                )
            else:
                logger.info("No improvement found, stopping selection")
                break

        # Final check for seasonal term
        if len(selected) > 0:
            bic_with_seasonal = self._fit_model_subset(  # noqa: E501
                selected, include_seasonal=True
            )
            if bic_with_seasonal < current_model_bic:
                logger.info(  # noqa: E501
                    f"Including seasonal term, final BIC: {bic_with_seasonal:.2f}"  # noqa: E501
                )
                self.include_seasonal = True
            else:
                self.include_seasonal = False
        else:
            self.include_seasonal = False

        logger.info(f"Selected sensors: {selected}")
        return selected

    def _fit_model_subset(
        self, sensors: List[str], include_seasonal: bool = False
    ) -> float:
        """
        Fit model with subset of sensors and return BIC.

        Args:
            sensors: List of sensor names to include
            include_seasonal: Whether to include seasonal component

        Returns:
            BIC value
        """
        try:
            self.selected_sensors = sensors

            # Initialize parameters
            n_params = 3 + 2 * len(sensors)  # a, pi_b, phi_b, + 2 per sensor
            if include_seasonal:
                n_params += 2  # pi_d, phi_d

            # Starting values
            init_params = np.zeros(n_params)
            init_params[0] = -3.0  # Intercept
            init_params[1] = 0.5  # pi_b
            init_params[2] = 0.5  # phi_b

            idx = 3
            for i in range(len(sensors)):
                init_params[idx] = 0.5  # tau_j
                init_params[idx + 1] = 0.5  # psi_j
                idx += 2

            if include_seasonal:
                init_params[idx] = 0.5  # pi_d
                init_params[idx + 1] = 0.8  # phi_d

            # Set bounds to ensure stability
            bounds = []
            bounds.append((-10, 10))  # a
            bounds.append((-5, 5))  # pi_b
            bounds.append((-0.99, 0.99))  # phi_b (stability)

            for i in range(len(sensors)):
                bounds.append((-5, 5))  # tau_j
                bounds.append((-0.99, 0.99))  # psi_j (stability)

            if include_seasonal:
                bounds.append((-5, 5))  # pi_d
                bounds.append((-0.99, 0.99))  # phi_d (stability)

            # Optimize
            result = minimize(
                self._log_likelihood,
                init_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500},
            )

            if result.success:
                # Calculate BIC
                log_lik = -result.fun
                n_obs = len(self.y)
                bic = -2 * log_lik + n_params * np.log(n_obs)
                return bic
            else:
                return float("inf")

        except Exception as e:
            logger.warning(f"Error in model fitting: {e}")
            return float("inf")

    def fit(
        self,
        data: SensorDataset | pd.DataFrame,
        perform_selection: bool = True,
    ) -> Dict:
        """
        Fit the Bernoulli autoregressive model.

        Args:
            data: DataFrame with sensor data (15-minute intervals)
            perform_selection: Whether to perform automatic sensor selection

        Returns:
            Dictionary containing fitted parameters and model info
        """
        logger.info(f"Fitting model for target sensor: {self.target_sensor}")

        # Prepare data
        df = data.to_dataframe() if isinstance(data, SensorDataset) else data
        self.y = df[self.target_sensor].values.astype(float)
        self.X_other = df[self.other_sensors].values.astype(float)

        # Perform sensor selection
        if perform_selection:
            self.selected_sensors = self._stepwise_selection(df)
        else:
            self.selected_sensors = self.other_sensors
            self.include_seasonal = True

        # Fit final model
        n_params = 3 + 2 * len(self.selected_sensors)
        if hasattr(self, "include_seasonal") and self.include_seasonal:
            n_params += 2

        # Initialize parameters
        init_params = np.zeros(n_params)
        init_params[0] = -3.0  # Intercept
        init_params[1] = 0.5  # pi_b
        init_params[2] = 0.5  # phi_b

        idx = 3
        for i in range(len(self.selected_sensors)):
            init_params[idx] = 0.5  # tau_j
            init_params[idx + 1] = 0.5  # psi_j
            idx += 2

        if hasattr(self, "include_seasonal") and self.include_seasonal:
            init_params[idx] = 0.5  # pi_d
            init_params[idx + 1] = 0.8  # phi_d

        # Set bounds
        bounds = []
        bounds.append((-10, 10))  # a
        bounds.append((-5, 5))  # pi_b
        bounds.append((-0.99, 0.99))  # phi_b

        for i in range(len(self.selected_sensors)):
            bounds.append((-5, 5))  # tau_j
            bounds.append((-0.99, 0.99))  # psi_j

        if hasattr(self, "include_seasonal") and self.include_seasonal:
            bounds.append((-5, 5))  # pi_d
            bounds.append((-0.99, 0.99))  # phi_d

        # Final optimization
        logger.info("Fitting final model...")
        result = minimize(
            self._log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        if result.success:
            self.params = self._vector_to_params(result.x)

            # Calculate final statistics
            log_lik = -result.fun
            n_obs = len(self.y)
            bic = -2 * log_lik + n_params * np.log(n_obs)

            logger.info(f"Model fitted successfully. BIC: {bic:.2f}")

            return {
                "parameters": self.params,
                "selected_sensors": self.selected_sensors,
                "include_seasonal": hasattr(self, "include_seasonal")
                and self.include_seasonal,
                "log_likelihood": log_lik,
                "bic": bic,
                "convergence": True,
            }
        else:
            logger.error("Model fitting failed to converge")
            return {"convergence": False}

    def predict_probabilities(
        self, data: pd.DataFrame | SensorDataset, start_idx: int = None
    ) -> np.ndarray:
        """
        Predict one-step-ahead probabilities.

        Args:
            data: DataFrame with sensor data
            start_idx: Starting index for prediction
                (if None, start from beginning)

        Returns:
            Array of predicted probabilities
        """
        if not self.params:
            raise ValueError("Model must be fitted before prediction")

        # Prepare data
        df = data.to_dataframe() if isinstance(data, SensorDataset) else data
        self.y = df[self.target_sensor].values.astype(float)
        self.X_other = df[self.other_sensors].values.astype(float)

        if start_idx is None:
            start_idx = 0

        T = len(self.y)
        ar_cache = np.zeros(T)
        sensor_cache = {s: np.zeros(T) for s in self.selected_sensors}
        for t in range(1, T):
            ar_cache[t] = (
                self.params["phi_b"] * ar_cache[t - 1] + self.y[t - 1]
            )  # noqa: E501
            for s in self.selected_sensors:
                idx = self.other_sensors.index(s)
                psi_key = f"psi_{s}"
                sensor_cache[s][t] = (
                    self.params[psi_key] * sensor_cache[s][t - 1]
                    + self.X_other[t - 1, idx]
                )

        seasonal = np.zeros(T)
        if "pi_d" in self.params and "phi_d" in self.params:
            for t in range(self.INTERVALS_PER_DAY, T):
                prev_day = t - self.INTERVALS_PER_DAY
                start = max(0, prev_day - 1)
                end = min(len(self.y), prev_day + 2)
                window_max = np.max(self.y[start:end])
                seasonal[t] = (
                    self.params["phi_d"] * seasonal[t - self.INTERVALS_PER_DAY]
                    + window_max
                )

        delta = self.params["a"] + self.params["pi_b"] * ar_cache
        for s in self.selected_sensors:
            delta += self.params[f"tau_{s}"] * sensor_cache[s]
        if "pi_d" in self.params and "phi_d" in self.params:
            delta += self.params["pi_d"] * seasonal

        delta = delta[start_idx:]
        probabilities = np.exp(delta) / (1 + np.exp(delta))
        return probabilities

    def compute_quantile_intervals(
        self, probabilities: np.ndarray, confidence: float = 0.95
    ) -> Dict:
        """
        Compute quantile intervals using Poisson binomial distribution.

        Args:
            probabilities: Array of predicted probabilities
                (grouped by time of day)
            confidence: Confidence level (default 0.95)

        Returns:
            Dictionary with quantile information
        """
        # Group probabilities by time of day (15-minute intervals)
        n_intervals = len(probabilities)
        n_days = n_intervals // self.INTERVALS_PER_DAY

        if n_days == 0:
            logger.warning("Not enough data for daily quantile intervals")
            return {}

        # Reshape probabilities by time of day
        prob_by_time = probabilities[
            : n_days * self.INTERVALS_PER_DAY
        ].reshape(  # noqa: E501
            n_days, self.INTERVALS_PER_DAY
        )

        alpha = 1 - confidence
        lower_quantiles = []
        upper_quantiles = []
        means = []

        for time_idx in range(self.INTERVALS_PER_DAY):
            time_probs = prob_by_time[:, time_idx]

            # Compute mean and variance for this time period
            mean = np.sum(time_probs)
            variance = np.sum(time_probs * (1 - time_probs))

            means.append(mean)

            # Approximate quantiles using normal distribution
            # (for computational efficiency vs exact Poisson binomial)
            std = np.sqrt(variance)

            # Normal approximation quantiles
            lower_q = max(0, mean + norm.ppf(alpha / 2) * std)
            upper_q = min(n_days, mean + norm.ppf(1 - alpha / 2) * std)

            lower_quantiles.append(lower_q)
            upper_quantiles.append(upper_q)

        return {
            "lower_quantiles": np.array(lower_quantiles),
            "upper_quantiles": np.array(upper_quantiles),
            "means": np.array(means),
            "time_of_day": np.arange(self.INTERVALS_PER_DAY),
        }
