"""Sensible default observation models derived from the sensor registry.

Hand-writing an emission model per sensor is the right thing to do for a study
that has calibration data. For everything else -- a first deployment, an
ablation sweep, the worked example -- the registry already says enough to build
a defensible model: the modality says what kind of likelihood applies, and the
room says which states should expect the sensor to fire.

Deriving the models from declarations rather than from sensor names is what
keeps the platform sensor-agnostic. Adding a new device to a deployment means
adding a :class:`~sensor_modeling.observations.registry.SensorSpec`, not
editing inference code.

The defaults are starting points and are meant to be replaced by fitted values
where data allows. They are deliberately conservative: rates differ between
states by roughly an order of magnitude rather than by several, so no single
sensor can dominate the posterior on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..observations.registry import SensorRegistry, SensorSpec
from ..observations.types import Modality, ObservationKind
from ..observations.units import Unit
from ..states.ontology import BehaviouralState, StateOntology
from .emissions import (
    BernoulliEmission,
    BetaEmission,
    EmissionModel,
    GaussianEmission,
    PoissonEventEmission,
)

logger = logging.getLogger(__name__)

S = BehaviouralState


@dataclass
class EmissionDefaults:
    """Rate and level assumptions used to build default emission models.

    Parameters
    ----------
    active_rate
        Activations per hour a room sensor produces when the resident is in
        that room and fully active.
    activity_levels
        Fraction of ``active_rate`` each state actually generates. This is the
        distinction between *where* a state puts the resident and *how much
        they move* while there, and it is not optional: a sleeping person is
        in the bedroom but almost perfectly still, so a bedroom sensor's
        silence is expected during sleep rather than evidence against it.
    idle_rate
        Background rate: false activations, pets, draughts. Never zero, so a
        single stray activation cannot rule a state out permanently.
    away_rate
        Rate while the resident is out. Non-zero because visitors exist.
    door_rate
        Rate for an entrance door in the states either side of a crossing.
    wearable_levels
        Expected wearable activity magnitude per state.
    wearable_sigma
        Standard deviation of the wearable signal.
    presence_probability
        Probability a personal beacon reads in-range in each home state.
    bed_probability
        Probability a bed sensor reads occupied in the in-bed states.
    """

    active_rate: float = 40.0
    activity_levels: dict[BehaviouralState, float] | None = None
    idle_rate: float = 0.15
    away_rate: float = 0.05
    door_rate: float = 4.0
    wearable_levels: dict[BehaviouralState, float] | None = None
    wearable_sigma: float = 0.35
    presence_probability: float = 0.95
    bed_probability: float = 0.97

    def __post_init__(self) -> None:
        """Validate the assumptions and supply per-state defaults."""
        rates = (self.active_rate, self.idle_rate, self.away_rate, self.door_rate)
        if any(rate < 0 for rate in rates):
            raise ValueError("emission rates must be non-negative")
        if self.wearable_sigma <= 0:
            raise ValueError("wearable_sigma must be positive")
        for name in ("presence_probability", "bed_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.activity_levels is None:
            self.activity_levels = {
                S.SLEEPING: 0.02,
                S.BED_AWAKE: 0.35,
                S.HOME_INACTIVE: 0.25,
                S.HOME_ACTIVE: 1.0,
                S.KITCHEN_ACTIVITY: 1.0,
                S.BATHROOM_ACTIVITY: 1.0,
                S.AWAY: 0.0,
            }
        if any(level < 0 for level in self.activity_levels.values()):
            raise ValueError("activity levels must be non-negative")
        if self.wearable_levels is None:
            self.wearable_levels = {
                S.SLEEPING: 0.05,
                S.BED_AWAKE: 0.3,
                S.HOME_INACTIVE: 0.4,
                S.HOME_ACTIVE: 1.4,
                S.KITCHEN_ACTIVITY: 1.1,
                S.BATHROOM_ACTIVITY: 0.9,
                S.AWAY: 1.8,
            }


def _event_rates(
    spec: SensorSpec, ontology: StateOntology, defaults: EmissionDefaults
) -> dict[BehaviouralState, float]:
    """Return per-state activation rates for an event sensor.

    A rate is the product of two independent things: whether the state puts
    the resident within range of this sensor, and how much they move while in
    that state. Conflating the two is the classic way to make a bed sensor
    and a bedroom motion sensor argue with each other all night.
    """
    levels = defaults.activity_levels or {}
    located = {ontology.room_of(state) for state in ontology.states}
    # A state that names no room could put the resident in any of the rooms
    # the ontology does name, plus somewhere unmonitored, so its rate for any
    # one sensor is diluted across those possibilities.
    spread = max(len({room for room in located if room}) + 1, 1)

    rates: dict[BehaviouralState, float] = {}
    for state in ontology.states:
        activity = float(levels.get(state, 0.5))
        room = ontology.room_of(state)
        if state is S.AWAY:
            rate = defaults.away_rate
        elif spec.room is None:
            rate = defaults.active_rate * activity
        elif room == spec.room:
            rate = defaults.active_rate * activity
        elif room is None:
            rate = defaults.active_rate * activity / spread
        else:
            rate = defaults.idle_rate
        rates[state] = max(rate, defaults.idle_rate)
    return rates


def _door_rates(
    ontology: StateOntology, defaults: EmissionDefaults
) -> dict[BehaviouralState, float]:
    """Return per-state rates for an entrance door.

    A door fires at the boundary between being in and being out, so it is
    most informative about the states either side of a crossing rather than
    about any state in isolation.
    """
    return {
        state: (
            defaults.door_rate
            if state in (S.AWAY, S.HOME_ACTIVE)
            else defaults.idle_rate
        )
        for state in ontology.states
    }


def default_emission_for(
    spec: SensorSpec,
    ontology: StateOntology,
    defaults: EmissionDefaults | None = None,
) -> EmissionModel | None:
    """Build a default observation model for one declared sensor.

    Returns
    -------
    EmissionModel | None
        ``None`` when the modality carries no information about behavioural
        state. A room-occupancy count, for instance, is evidence about *who
        is present* and belongs to the occupancy layer, not here.
    """
    settings = defaults or EmissionDefaults()
    levels = settings.wearable_levels or {}

    if spec.modality is Modality.DOOR:
        return PoissonEventEmission(
            spec.sensor_id, rates=_door_rates(ontology, settings)
        )

    if spec.kind is ObservationKind.EVENT:
        return PoissonEventEmission(
            spec.sensor_id, rates=_event_rates(spec, ontology, settings)
        )

    if spec.modality is Modality.BED_PRESSURE:
        occupied = {
            state: (
                settings.bed_probability
                if state in (S.SLEEPING, S.BED_AWAKE)
                else 1.0 - settings.bed_probability
            )
            for state in ontology.states
        }
        return BernoulliEmission(spec.sensor_id, probabilities=occupied)

    if spec.modality in (Modality.WEARABLE_MOTION, Modality.WEARABLE_PHYSIOLOGY):
        return GaussianEmission(
            spec.sensor_id,
            means={state: levels.get(state, 0.5) for state in ontology.states},
            sigmas={state: settings.wearable_sigma for state in ontology.states},
            # A wearable samples far more often than a contact sensor fires.
            # Without this discount its sampling rate, rather than its
            # informativeness, would decide how much it counts.
            weight=0.25,
        )

    if spec.modality is Modality.PROXIMITY:
        present = {
            state: (
                1.0 - settings.presence_probability
                if state is S.AWAY
                else settings.presence_probability
            )
            for state in ontology.states
        }
        return BernoulliEmission(spec.sensor_id, probabilities=present)

    if spec.modality is Modality.RADAR:
        if spec.unit is Unit.PROBABILITY:
            return BetaEmission(
                spec.sensor_id,
                means={
                    state: (0.05 if state is S.AWAY else 0.9)
                    for state in ontology.states
                },
                weight=0.25,
            )
        # A track count is presence evidence: zero people means nobody home.
        return GaussianEmission(
            spec.sensor_id,
            means={
                state: (0.05 if state is S.AWAY else 1.0) for state in ontology.states
            },
            sigmas={state: 0.7 for state in ontology.states},
            weight=0.25,
        )

    logger.debug(
        "No default emission model for modality %s ('%s')",
        spec.modality.value,
        spec.sensor_id,
    )
    return None


def _group_sizes(registry: SensorRegistry) -> dict[str, int]:
    """Count how many sensors share each declared redundancy group."""
    sizes: dict[str, int] = {}
    for spec in registry:
        if spec.redundancy_group:
            sizes[spec.redundancy_group] = sizes.get(spec.redundancy_group, 0) + 1
    return sizes


def default_emissions(
    registry: SensorRegistry,
    ontology: StateOntology,
    defaults: EmissionDefaults | None = None,
) -> list[EmissionModel]:
    """Build default observation models for every sensor that has one.

    Sensors sharing a ``redundancy_group`` have their evidence weight divided
    across the group. The filter combines sensors as conditionally
    independent given the state, so without this, several views of the same
    physical event accumulate as several independent observations and drive
    the posterior toward certainty it has not earned.
    """
    sizes = _group_sizes(registry)
    models: list[EmissionModel] = []
    for spec in registry:
        model = default_emission_for(spec, ontology, defaults)
        if model is None:
            continue
        if spec.redundancy_group:
            model.weight /= sizes[spec.redundancy_group]
        models.append(model)
    if not models:
        raise ValueError(
            "no sensor in the registry carries information about behavioural state"
        )
    return models
