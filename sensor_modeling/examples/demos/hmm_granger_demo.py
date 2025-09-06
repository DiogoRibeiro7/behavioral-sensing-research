"""Example combining HMM state sequences with Granger causality analysis."""

import pandas as pd
from sensor_modeling.utils import simulate_sensor_data, SensorDataset
from sensor_modeling.hmm import HierarchicalHMM
from sensor_modeling.analysis import GrangerCausalityTest


def main():
    # Simulate binary sensor data for two sensors
    data = simulate_sensor_data(n_sensors=2, n_timesteps=200)
    dataset = SensorDataset(data)
    df = dataset.to_dataframe()

    # Fit an HMM to each sensor independently
    state_df = pd.DataFrame()
    for sensor in df.columns:
        model = HierarchicalHMM(n_states=3, random_state=0)
        model.fit(df[[sensor]].values)
        state_df[sensor] = model.predict(df[[sensor]].values)

    # Apply Granger causality test on the inferred state sequences
    gtest = GrangerCausalityTest(max_lags=3)
    results = gtest.test_all_pairs(state_df)
    print(results)


if __name__ == "__main__":
    main()
