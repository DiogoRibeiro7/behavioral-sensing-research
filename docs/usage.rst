Usage Examples
==============

A minimal Bernoulli Autoregressive workflow::

    from sensor_modeling.models import BernoulliAutoregressiveModel
    import pandas as pd

    df = pd.DataFrame({"a": [0, 1, 0], "b": [1, 0, 1]})
    model = BernoulliAutoregressiveModel(df.columns.tolist(), "a")
    model.fit(df)
    predictions = model.predict(df)

    print(predictions)

