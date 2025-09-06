import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


class BaseHMM(BaseEstimator):
    """Simple Hidden Markov Model base class.

    Provides a lightweight implementation that supports both batch and online
    learning through :meth:`fit` and :meth:`partial_fit`. Missing data are
    imputed using column means. Visualization helpers are provided for state
    transition and emission statistics, and model selection utilities compute
    AIC and BIC metrics.
    """

    def __init__(
        self, n_states: int = 2, n_iter: int = 10, random_state: int | None = None
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.trans_mat_ = None
        self.means_ = None
        self.fitted_ = False

    # ------------------------------------------------------------------
    def _handle_missing(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values with column means."""
        X = np.asarray(X, dtype=float)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        return X

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y=None):
        """Batch learning using a k-means style EM procedure."""
        X = self._handle_missing(X)
        rng = np.random.RandomState(self.random_state)
        means = X[rng.choice(len(X), self.n_states, replace=False)]

        for _ in range(self.n_iter):
            dists = ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
            states = dists.argmin(axis=1)
            for k in range(self.n_states):
                if np.any(states == k):
                    means[k] = X[states == k].mean(axis=0)

        self.means_ = means
        trans = np.zeros((self.n_states, self.n_states))
        for i, j in zip(states[:-1], states[1:]):
            trans[i, j] += 1
        trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)
        self.trans_mat_ = trans
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    def partial_fit(self, X: np.ndarray, y=None):
        """Online update that refines parameters with new data."""
        if not self.fitted_:
            return self.fit(X, y)
        X = self._handle_missing(X)
        dists = ((X[:, None, :] - self.means_[None, :, :]) ** 2).sum(axis=2)
        states = dists.argmin(axis=1)
        for k in range(self.n_states):
            if np.any(states == k):
                self.means_[k] = (self.means_[k] + X[states == k].mean(axis=0)) / 2
        for i, j in zip(states[:-1], states[1:]):
            self.trans_mat_[i, j] += 1
        self.trans_mat_ = self.trans_mat_ / np.maximum(
            self.trans_mat_.sum(axis=1, keepdims=True), 1
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction.")
        X = self._handle_missing(X)
        dists = ((X[:, None, :] - self.means_[None, :, :]) ** 2).sum(axis=2)
        return dists.argmin(axis=1)

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y=None) -> float:
        """Pseudo log-likelihood based on squared distances to state means."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before scoring.")
        X = self._handle_missing(X)
        dists = ((X[:, None, :] - self.means_[None, :, :]) ** 2).sum(axis=2)
        return -dists.min(axis=1).sum()

    # ------------------------------------------------------------------
    def compute_aic(self, X: np.ndarray) -> float:
        """Calculate Akaike Information Criterion."""
        ll = self.score(X)
        n_params = self.n_states * X.shape[1] + self.n_states * (self.n_states - 1)
        return 2 * n_params - 2 * ll

    # ------------------------------------------------------------------
    def compute_bic(self, X: np.ndarray) -> float:
        """Calculate Bayesian Information Criterion."""
        ll = self.score(X)
        n_params = self.n_states * X.shape[1] + self.n_states * (self.n_states - 1)
        n = len(X)
        return n_params * np.log(n) - 2 * ll

    # ------------------------------------------------------------------
    def plot_state_transitions(self, ax=None):
        """Visualize the state transition matrix."""
        if ax is None:
            _, ax = plt.subplots()
        im = ax.imshow(self.trans_mat_, cmap="Blues")
        ax.set_title("State Transition Matrix")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        plt.colorbar(im, ax=ax)
        return ax

    # ------------------------------------------------------------------
    def plot_emissions(self, ax=None):
        """Visualize state emission means."""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.means_)
        ax.set_title("State Emission Means")
        ax.set_xlabel("State")
        return ax
