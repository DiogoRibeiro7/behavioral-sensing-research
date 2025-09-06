"""Property-based tests for simulation utilities."""

from hypothesis import given, strategies as st

from sensor_modeling.utils.data_io import simulate_sensor_data


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
)
def test_simulate_sensor_data_shapes(days: int, sensors: int) -> None:
    """Simulated dataset shapes follow the specified days and sensors."""
    dataset = simulate_sensor_data(n_days=days, n_sensors=sensors)
    assert dataset.data.shape[0] == days * 96
    assert dataset.data.shape[1] == sensors
