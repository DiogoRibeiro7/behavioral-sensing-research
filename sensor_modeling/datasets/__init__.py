"""Adapters bringing public annotated datasets into the canonical model.

Everything else in this package is validated on a simulator this project wrote.
That establishes properties of the inference model, not of any real home. These
adapters exist so the same pipeline can be run against data it did not generate,
which is the only way to find out whether the architecture survives contact with
reality.

An adapter converts a published recording into :class:`Observation` values and a
ground-truth series. It does not adjust, clean or reinterpret the data to suit
the pipeline: where a dataset cannot answer a question the pipeline asks, the
adapter reports that rather than inventing an answer.

No dataset is redistributed here. Each adapter documents where to obtain the
recording and under what terms.
"""

from .casas import (
    CASAS_ACTIVITY_STATES,
    ActivityInterval,
    CasasReadError,
    CasasRecording,
    casas_sensor_specs,
    read_casas,
    truth_series,
)
from .casas_hh import HH_ACTIVITY_STATES, HH_LOCATIONS, hh_sensor_specs, read_casas_hh
from .evaluate import DatasetEvaluation, evaluate_recording
from .rates import (
    RateReport,
    RateSample,
    fit_emission_defaults,
    measure_event_rates,
    pooled_rate_report,
)

__all__ = [
    "ActivityInterval",
    "DatasetEvaluation",
    "HH_ACTIVITY_STATES",
    "HH_LOCATIONS",
    "hh_sensor_specs",
    "read_casas_hh",
    "RateReport",
    "RateSample",
    "fit_emission_defaults",
    "measure_event_rates",
    "pooled_rate_report",
    "evaluate_recording",
    "CASAS_ACTIVITY_STATES",
    "CasasReadError",
    "CasasRecording",
    "casas_sensor_specs",
    "read_casas",
    "truth_series",
]
