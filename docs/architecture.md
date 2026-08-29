# Architecture Overview

The toolkit is organized into four core layers:

* `sensor_modeling.models` – statistical models such as the Bernoulli autoregressive family and hidden Markov variants.
* `sensor_modeling.analysis` – preprocessing utilities and evaluation pipelines.
* `sensor_modeling.visualization` – dashboards and web applications for inspecting model outputs.
* `sensor_modeling.cli` – command-line interface that links the components for batch experiments.

## Ambient sensing layers

The multimodal ambient-sensing pipeline adds a further set of layers on top of
the modelling core:

* `sensor_modeling.observations` -- the canonical, hardware-neutral
  observation model, sensor registry, and boundary validation.
* `sensor_modeling.health` -- online sensor reliability, emitted as an
  evidence weight.
* `sensor_modeling.context` -- occupancy estimation and uncertainty-aware
  attribution of activity.
* `sensor_modeling.states` and `sensor_modeling.fusion` -- the latent
  behavioural state ontology and the recursive multimodal filter.
* `sensor_modeling.baseline` -- adaptive, non-stationary personal baselines
  and behavioural change verdicts.
* `sensor_modeling.alerts` -- restrained, explainable alerting.
* `sensor_modeling.simulation` -- synthetic households with ground truth.
* `sensor_modeling.evaluation` -- metrics and paired ablation experiments.
* `sensor_modeling.online` -- incremental orchestration of the chain.

See [ambient_architecture](ambient_architecture.md) for the full description, [inference](inference.md) for
what each layer claims, [evaluation](evaluation.md) for how it is measured, and
[limitations](limitations.md) for what it does not establish.
