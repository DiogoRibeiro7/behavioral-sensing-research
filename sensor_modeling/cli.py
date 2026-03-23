"""Unified command-line interface for sensor modeling."""

from __future__ import annotations

import argparse

from .models.bernoulli_ar.base_model import BernoulliAutoregressiveModel
from .models.nhpp_pelt.model import NHPPPELT, NHPPConfig
from .utils.data_io import SensorDataset
from .utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensor modeling CLI")
    sub = parser.add_subparsers(dest="model", required=True)

    ar_p = sub.add_parser("bernoulli-ar", help="Run Bernoulli autoregressive model")
    ar_p.add_argument("data", help="Path to CSV sensor data")
    ar_p.add_argument("target", help="Target sensor column")

    nhpp_p = sub.add_parser("nhpp-pelt", help="Run NHPP-PELT changepoint model")
    nhpp_p.add_argument("data", help="Path to CSV sensor data")
    nhpp_p.add_argument("sensor", help="Sensor column to use")

    args = parser.parse_args()
    setup_logging()

    dataset = SensorDataset.from_csv(args.data)

    if args.model == "bernoulli-ar":
        model = BernoulliAutoregressiveModel(list(dataset.data.columns), args.target)
        model.fit(dataset)
    elif args.model == "nhpp-pelt":
        cfg = NHPPConfig()
        model = NHPPPELT(cfg)
        model.fit(dataset, sensor=args.sensor)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
