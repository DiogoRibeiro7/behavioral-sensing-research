"""Unified package for sensor modeling and analysis.

The package has two layers. The original modelling core provides statistical
models, analysis routines and visualisation for behavioural sensor data. On
top of it sits a multimodal ambient-sensing pipeline that runs from raw sensor
traffic to an explained alert:

.. code-block:: text

    observations -> health -> context -> fusion -> baseline -> alerts

See ``docs/ambient_architecture.rst`` for the architecture and
``docs/limitations.rst`` for what the platform does not establish.
"""

__version__ = "0.1.3"

#: Subpackages of the original modelling core.
CORE_MODULES = (
    "models",
    "analysis",
    "utils",
    "change_point",
    "hmm",
    "data",
    "visualization",
)

#: Subpackages of the multimodal ambient-sensing pipeline, in the order the
#: data flows through them.
AMBIENT_MODULES = (
    "observations",
    "health",
    "context",
    "states",
    "fusion",
    "baseline",
    "alerts",
    "online",
    "simulation",
    "evaluation",
    "interop",
)

__all__ = [*CORE_MODULES, *AMBIENT_MODULES, "CORE_MODULES", "AMBIENT_MODULES"]
