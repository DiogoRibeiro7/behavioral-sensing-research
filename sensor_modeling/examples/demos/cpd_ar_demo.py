"""Demo: combine Bernoulli AR model with change point detection."""

from __future__ import annotations

from sensor_modeling.models.bernoulli_ar import BernoulliAutoregressiveModel
from sensor_modeling.change_point import EmbeddingCPD
from sensor_modeling.utils import simulate_sensor_data, SensorDataset


def main() -> None:
    data = simulate_sensor_data(num_sensors=1, length=200)
    sensor_name = data.columns[0]
    dataset = SensorDataset(data)

    model = BernoulliAutoregressiveModel([sensor_name], sensor_name)
    model.fit(dataset)
    probs = model.predict_probabilities(dataset)

    cpd = EmbeddingCPD().fit(probs)
    cps = cpd.predict(plot=True)
    print("Detected change points:", cps)


if __name__ == "__main__":
    main()
