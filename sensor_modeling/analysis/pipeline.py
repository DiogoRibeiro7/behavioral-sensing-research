"""Unified analysis pipeline for sensor modeling.

This module exposes :class:`AnalysisPipeline` which orchestrates the
available modeling approaches (autoregressive, HMM, change-point and NHPP)
on a common :class:`~sensor_modeling.utils.data_io.SensorDataset` input.

The pipeline hides the complexity of individual models while presenting a
simple interface for end users::

    from sensor_modeling.analysis import AnalysisPipeline
    from sensor_modeling.utils.data_io import SensorDataset

    pipeline = AnalysisPipeline()
    results = pipeline.run(dataset)
    pipeline.generate_report(results, output_dir="out")

Each model is executed in parallel and their results are collected in a
single dictionary that can be further analysed or exported.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict
import logging

import pandas as pd

from ..utils.data_io import SensorDataset
from ..models.bernoulli_ar.base_model import BernoulliAutoregressiveModel
from ..hmm.base import BaseHMM
from ..change_point.embedding_cpd import EmbeddingCPD
from ..models.nhpp_pelt.model import NHPPPELT, NHPPConfig

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Master analysis class coordinating all modeling approaches."""

    def __init__(self, models: Dict[str, Any] | None = None):
        self.models = models or {}
        self.results: Dict[str, Any] = {}
        self.dataset: SensorDataset | None = None

    # ------------------------------------------------------------------
    def _default_models(self, sensor_names: list[str]) -> Dict[str, Any]:
        """Create default model instances for the given sensors."""
        ar_model = BernoulliAutoregressiveModel(sensor_names, sensor_names[0])
        hmm_model = BaseHMM()
        cpd_model = EmbeddingCPD()
        nhpp_model = NHPPPELT(NHPPConfig(n_basis=3, min_seg_len=1))
        return {
            "ar": ar_model,
            "hmm": hmm_model,
            "cpd": cpd_model,
            "nhpp": nhpp_model,
        }

    # ------------------------------------------------------------------
    def run(self, data: pd.DataFrame | SensorDataset) -> Dict[str, Any]:
        """Run all configured models on *data* in parallel."""
        self.dataset = data if isinstance(data, SensorDataset) else SensorDataset(data)
        df = self.dataset.to_dataframe()
        sensor_names = list(df.columns)
        if not self.models:
            self.models = self._default_models(sensor_names)

        def _run(name: str, model: Any) -> Dict[str, Any]:
            try:
                if name == "nhpp":
                    model.fit(self.dataset, sensor=sensor_names[0])
                    return {"changepoints": getattr(model, "changepoints_", [])}
                if name == "cpd":
                    series = df[sensor_names[0]].values
                    model.fit(series)
                    return {"changepoints": model.predict()}
                if name == "hmm":
                    X = df.values
                    model.fit(X)
                    states = model.predict(X)
                    return {"states": states}
                if name == "ar":
                    model.fit(df)
                    probs = model.predict_probabilities(df)
                    return {"probabilities": probs.iloc[:5].to_dict("list")}
                return {"status": "skipped"}
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Model %s failed: %s", name, exc)
                return {"error": str(exc)}

        with ThreadPoolExecutor() as ex:
            futures = {n: ex.submit(_run, n, m) for n, m in self.models.items()}
            self.results = {n: f.result() for n, f in futures.items()}
        return self.results

    # ------------------------------------------------------------------
    def generate_report(self, results: Dict[str, Any], output_dir: str) -> None:
        """Generate LaTeX, HTML and FHIR reports for *results*."""
        from .reporting import (
            generate_latex_report,
            create_html_dashboard,
            export_to_fhir,
        )

        generate_latex_report(results, f"{output_dir}/analysis.tex")
        create_html_dashboard(results, f"{output_dir}/dashboard.html")
        export_to_fhir(results, f"{output_dir}/analysis_fhir.json")
