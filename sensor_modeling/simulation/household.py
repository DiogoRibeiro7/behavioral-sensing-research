"""A synthetic household with controlled ground truth.

The simulator exists so that inference can be evaluated against something
known. That only works if the generative process is genuinely different from
the inference model, so this module deliberately does *not* sample from the
continuous-time chain the filter uses. It generates a **schedule**: a person
who wakes at roughly the same time each morning, makes breakfast, goes out,
comes back, and goes to bed, with stochastic timings and durations.

Inference then has to recover that schedule through a Markov model that knows
nothing about schedules. If the two shared a generator, good results would
prove only that the code can invert its own assumptions.

What the ground truth records:

.. code-block:: text

    resident location and behavioural state, minute by minute
    sleep periods and night-time bathroom trips
    visitor arrivals, departures, and their own movements
    room transitions and door crossings
    which sensor activations came from whom

Sensor records are then generated *from* that truth, with per-sensor rates,
false activations, and misses. Degrading the stream further -- dropouts,
stuck sensors, a wearable left off -- is the separate concern of
:mod:`sensor_modeling.simulation.faults`, so that robustness studies can vary
faults without regenerating behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import numpy as np

from ..context.occupancy import OccupancyContext
from ..observations.observation import Observation
from ..observations.registry import SensorRegistry, SensorSpec
from ..observations.types import Modality, ObservationKind
from ..observations.units import Unit
from ..states.ontology import BehaviouralState

logger = logging.getLogger(__name__)

S = BehaviouralState
LISBON = ZoneInfo("Europe/Lisbon")


@dataclass(frozen=True)
class Episode:
    """One period of known resident behaviour.

    Attributes
    ----------
    start, end
        Absolute bounds of the episode.
    state
        The resident's true behavioural state.
    room
        The room they are in, or ``None`` when out of the home.
    """

    start: datetime
    end: datetime
    state: BehaviouralState
    room: str | None

    @property
    def duration(self) -> timedelta:
        """Length of the episode."""
        return self.end - self.start

    def contains(self, moment: datetime) -> bool:
        """Whether *moment* falls inside the episode."""
        return self.start <= moment < self.end


@dataclass(frozen=True)
class VisitorPeriod:
    """A period during which someone other than the resident is present."""

    start: datetime
    end: datetime
    room_sequence: tuple[str, ...]

    def room_at(self, moment: datetime) -> str | None:
        """Return which room the visitor occupies at *moment*."""
        if not self.start <= moment < self.end:
            return None
        span = (self.end - self.start).total_seconds()
        if span <= 0:
            return self.room_sequence[0]
        progress = (moment - self.start).total_seconds() / span
        index = min(
            int(progress * len(self.room_sequence)), len(self.room_sequence) - 1
        )
        return self.room_sequence[index]


@dataclass(frozen=True)
class GroundTruth:
    """Everything the simulator knows and the inference layer must not see."""

    episodes: tuple[Episode, ...]
    visitors: tuple[VisitorPeriod, ...]
    tz: tzinfo

    def state_at(self, moment: datetime) -> BehaviouralState | None:
        """Return the resident's true state at *moment*, if simulated."""
        for episode in self.episodes:
            if episode.contains(moment):
                return episode.state
        return None

    def room_at(self, moment: datetime) -> str | None:
        """Return the resident's true room at *moment*."""
        for episode in self.episodes:
            if episode.contains(moment):
                return episode.room
        return None

    def visitor_at(self, moment: datetime) -> bool:
        """Whether anyone other than the resident is present at *moment*."""
        return any(period.start <= moment < period.end for period in self.visitors)

    def context_at(self, moment: datetime) -> OccupancyContext:
        """Return the true occupancy context at *moment*."""
        state = self.state_at(moment)
        resident_home = state is not None and state is not S.AWAY
        visitor = self.visitor_at(moment)
        if resident_home and visitor:
            return OccupancyContext.RESIDENT_WITH_VISITOR
        if resident_home:
            return OccupancyContext.RESIDENT_ALONE
        return OccupancyContext.VISITOR_ONLY if visitor else OccupancyContext.EMPTY

    def states_at(self, moments: Sequence[datetime]) -> list[BehaviouralState | None]:
        """Return the true state at each of *moments*, efficiently.

        Uses a single ordered sweep rather than scanning every episode for
        every moment, which matters when evaluating months of minute-level
        estimates.
        """
        results: list[BehaviouralState | None] = []
        index = 0
        episodes = self.episodes
        for moment in moments:
            while index < len(episodes) and episodes[index].end <= moment:
                index += 1
            if index < len(episodes) and episodes[index].contains(moment):
                results.append(episodes[index].state)
            else:
                results.append(None)
        return results

    def daily_hours(self, state: BehaviouralState) -> dict[date, float]:
        """Return true hours spent in *state* per local calendar day."""
        totals: dict[date, float] = {}
        for episode in self.episodes:
            if episode.state is not state:
                continue
            cursor = episode.start
            while cursor < episode.end:
                local = cursor.astimezone(self.tz)
                next_midnight = datetime.combine(
                    local.date() + timedelta(days=1),
                    time.min,
                    tzinfo=local.tzinfo,
                )
                boundary = min(next_midnight.astimezone(timezone.utc), episode.end)
                hours = (boundary - cursor).total_seconds() / 3600.0
                totals[local.date()] = totals.get(local.date(), 0.0) + hours
                cursor = boundary
        return totals


@dataclass
class BehaviourShift:
    """A persistent change in the resident's routine, injected at a known day.

    Parameters
    ----------
    start_day
        Days after the simulation begins at which the change takes effect.
    sleep_delta_hours
        Change in nightly sleep duration, negative for less sleep.
    night_bathroom_extra
        Additional expected night-time bathroom trips per night.
    outing_probability_delta
        Change in the probability of going out on a given day.
    ramp_days
        Days over which the change phases in linearly. Zero produces a step
        on ``start_day``; a positive value produces a gradual decline, which
        is a different detection problem: a step is visible in a single day's
        deviation, whereas a slow trend never is and can only be found by
        looking across a window.
    """

    start_day: int
    sleep_delta_hours: float = 0.0
    night_bathroom_extra: float = 0.0
    outing_probability_delta: float = 0.0
    ramp_days: int = 0

    def __post_init__(self) -> None:
        """Validate the injected shift."""
        if self.start_day < 0:
            raise ValueError("start_day must be non-negative")
        if self.ramp_days < 0:
            raise ValueError("ramp_days must be non-negative")

    def active_on(self, day_index: int) -> bool:
        """Whether the shift applies on the given simulation day."""
        return day_index >= self.start_day

    def strength_on(self, day_index: int) -> float:
        """Return how far the change has taken effect, in ``[0, 1]``."""
        if day_index < self.start_day:
            return 0.0
        if self.ramp_days <= 0:
            return 1.0
        elapsed = day_index - self.start_day
        return min(elapsed / self.ramp_days, 1.0)


@dataclass
class HouseholdConfig:
    """Configuration of the synthetic household.

    Parameters
    ----------
    days
        Number of days to simulate.
    start
        Local date the simulation begins.
    tz
        Timezone of the household, so routines follow the local clock.
    seed
        Random seed. Every run with the same configuration is identical.
    wake_hour, sleep_hour
        Mean local wake and bedtime, in hours.
    timing_jitter_minutes
        Standard deviation of daily timing variation.
    night_bathroom_rate
        Expected night-time bathroom trips per night.
    outing_probability
        Probability of leaving the home on a given day.
    visitor_probability
        Probability of a visitor on a given day.
    carer_weekday_visits
        Whether a regular carer visits on weekday mornings while the resident
        may be out, which is what makes attribution non-trivial.
    shift
        An optional persistent behavioural change to inject.
    """

    days: int = 60
    start: date = date(2024, 3, 4)
    tz: tzinfo = LISBON
    seed: int = 20240304
    wake_hour: float = 7.0
    sleep_hour: float = 23.0
    timing_jitter_minutes: float = 35.0
    night_bathroom_rate: float = 0.8
    outing_probability: float = 0.55
    visitor_probability: float = 0.2
    carer_weekday_visits: bool = True
    shift: BehaviourShift | None = None

    def __post_init__(self) -> None:
        """Validate the household configuration."""
        if self.days < 1:
            raise ValueError("days must be at least 1")
        if not 0.0 <= self.wake_hour < 24.0 or not 0.0 < self.sleep_hour < 24.0:
            raise ValueError("wake_hour and sleep_hour must be valid hours of day")
        if self.wake_hour >= self.sleep_hour:
            raise ValueError("wake_hour must precede sleep_hour within the same day")
        if self.timing_jitter_minutes < 0:
            raise ValueError("timing_jitter_minutes must be non-negative")
        if self.night_bathroom_rate < 0:
            raise ValueError("night_bathroom_rate must be non-negative")
        for name in ("outing_probability", "visitor_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


def build_registry() -> SensorRegistry:
    """Return the sensor deployment of the synthetic household.

    Deliberately multimodal, so that ablation studies have something to
    ablate: object contacts, room motion, a door, bed pressure, a radar
    presence feature, a wearable, and a personal beacon.
    """
    return SensorRegistry.from_specs(
        [
            SensorSpec(
                "front_door",
                Modality.DOOR,
                room="hall",
                description="Contact on the entrance door; fires on any crossing.",
            ),
            SensorSpec(
                "bedroom_motion",
                Modality.MOTION,
                room="bedroom",
                description="PIR covering the bedroom.",
            ),
            SensorSpec(
                "bathroom_motion",
                Modality.MOTION,
                room="bathroom",
                description="PIR covering the bathroom.",
            ),
            SensorSpec(
                "kitchen_motion",
                Modality.MOTION,
                room="kitchen",
                description="PIR covering the kitchen.",
            ),
            SensorSpec(
                "living_motion",
                Modality.MOTION,
                room="living",
                description="PIR covering the living room.",
            ),
            SensorSpec(
                "fridge_contact",
                Modality.CONTACT,
                room="kitchen",
                description="Contact on the fridge door. Records the door "
                "opening, which is not a record of eating.",
            ),
            SensorSpec(
                "bed_pressure",
                Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                room="bedroom",
                expected_interval=timedelta(minutes=5),
                value_range=(0.0, 1.0),
                description="Load cell reporting bed occupancy as a level.",
            ),
            SensorSpec(
                "living_radar",
                Modality.RADAR,
                kind=ObservationKind.SAMPLE,
                unit=Unit.COUNT,
                room="living",
                expected_interval=timedelta(minutes=1),
                value_range=(0.0, 6.0),
                description="Derived mmWave feature: number of tracked people "
                "in the home. Not raw radar; no identity is inferred.",
            ),
            SensorSpec(
                "wearable_motion",
                Modality.WEARABLE_MOTION,
                kind=ObservationKind.SAMPLE,
                room=None,
                expected_interval=timedelta(minutes=1),
                value_range=(0.0, 10.0),
                attributable=True,
                description="Accelerometer activity magnitude from a worn "
                "device. Attributable to the resident when worn.",
            ),
            SensorSpec(
                "resident_beacon",
                Modality.PROXIMITY,
                kind=ObservationKind.STATE,
                expected_interval=timedelta(minutes=1),
                value_range=(0.0, 1.0),
                attributable=True,
                description="Whether the resident's personal tag is in range "
                "of the home hub.",
            ),
        ]
    )


#: Activation rate per hour of each motion-style sensor when the resident is
#: in that room and active, and when they are not. The non-zero idle rate is
#: what produces the false activations any real deployment has.
ACTIVE_RATE = 45.0
IDLE_RATE = 0.12

#: Expected wearable activity magnitude in each true state.
WEARABLE_LEVEL: dict[BehaviouralState, float] = {
    S.SLEEPING: 0.03,
    S.BED_AWAKE: 0.25,
    S.HOME_INACTIVE: 0.35,
    S.HOME_ACTIVE: 1.4,
    S.KITCHEN_ACTIVITY: 1.1,
    S.BATHROOM_ACTIVITY: 0.9,
    S.AWAY: 1.9,
}


def _localise(day: date, hour: float, zone: tzinfo) -> datetime:
    """Return the absolute instant of a local wall-clock hour on *day*.

    Wall-clock arithmetic is done once, here, and everything downstream works
    in absolute time. A local hour that does not exist because of a
    spring-forward is nudged past the gap rather than silently resolving to
    the wrong instant.
    """
    clamped = min(max(hour, 0.0), 23.999)
    naive = datetime.combine(day, time.min) + timedelta(hours=clamped)
    local = naive.replace(tzinfo=zone)
    if local.astimezone(timezone.utc).astimezone(zone).hour != local.hour:
        local = (naive + timedelta(hours=1)).replace(tzinfo=zone)
    return local.astimezone(timezone.utc)


def _plan_day(
    day_index: int,
    day: date,
    config: HouseholdConfig,
    rng: np.random.Generator,
) -> tuple[list[Episode], list[VisitorPeriod]]:
    """Generate one day of true behaviour and visitor activity."""
    zone = config.tz
    jitter = config.timing_jitter_minutes / 60.0
    shift = config.shift
    # How far the injected change has taken effect today. A step shift is
    # fully on from its start day; a ramped one phases in.
    strength = shift.strength_on(day_index) if shift is not None else 0.0

    weekend = day.weekday() >= 5
    wake_hour = config.wake_hour + (1.2 if weekend else 0.0)
    wake_hour += rng.normal(0.0, jitter)
    sleep_hour = config.sleep_hour + rng.normal(0.0, jitter)
    if strength > 0.0 and shift is not None:
        # Less sleep is taken from the front of the night: the resident wakes
        # earlier rather than going to bed later, which is the pattern most
        # often reported in the ambient-monitoring literature.
        wake_hour -= shift.sleep_delta_hours * strength

    wake = _localise(day, wake_hour, zone)
    bedtime = _localise(day, sleep_hour, zone)
    midnight = _localise(day, 0.0, zone)

    episodes: list[Episode] = []

    # --- night: sleeping, interrupted by bathroom trips ---------------
    night_rate = config.night_bathroom_rate + (
        shift.night_bathroom_extra * strength if shift is not None else 0.0
    )
    trips = int(rng.poisson(max(night_rate, 0.0)))
    trip_starts = sorted(
        midnight + timedelta(hours=float(h))
        for h in rng.uniform(1.0, max(wake_hour - 0.5, 1.5), size=trips)
    )
    cursor = midnight
    for trip_start in trip_starts:
        if trip_start <= cursor or trip_start >= wake:
            continue
        episodes.append(Episode(cursor, trip_start, S.SLEEPING, "bedroom"))
        trip_end = min(trip_start + timedelta(minutes=float(rng.uniform(3, 9))), wake)
        episodes.append(Episode(trip_start, trip_end, S.BATHROOM_ACTIVITY, "bathroom"))
        cursor = trip_end
    if cursor < wake:
        episodes.append(Episode(cursor, wake, S.SLEEPING, "bedroom"))

    # --- morning routine ----------------------------------------------
    cursor = wake
    morning_bathroom = cursor + timedelta(minutes=float(rng.uniform(8, 20)))
    episodes.append(Episode(cursor, morning_bathroom, S.BED_AWAKE, "bedroom"))
    cursor = morning_bathroom
    after_bathroom = cursor + timedelta(minutes=float(rng.uniform(8, 18)))
    episodes.append(Episode(cursor, after_bathroom, S.BATHROOM_ACTIVITY, "bathroom"))
    cursor = after_bathroom
    breakfast_end = cursor + timedelta(minutes=float(rng.uniform(20, 45)))
    episodes.append(Episode(cursor, breakfast_end, S.KITCHEN_ACTIVITY, "kitchen"))
    cursor = breakfast_end

    daytime, cursor = _plan_daytime(day, cursor, bedtime, config, rng, zone, strength)
    episodes.extend(daytime)

    next_midnight = _localise(day + timedelta(days=1), 0.0, zone)
    if next_midnight > cursor:
        episodes.append(Episode(cursor, next_midnight, S.SLEEPING, "bedroom"))

    return episodes, _plan_visitors(day, config, rng, zone)


def _plan_daytime(
    day: date,
    start: datetime,
    bedtime: datetime,
    config: HouseholdConfig,
    rng: np.random.Generator,
    zone: tzinfo,
    shift_strength: float,
) -> tuple[list[Episode], datetime]:
    """Generate the waking day from after breakfast until bedtime.

    Returns the episodes and the moment the day's routine finishes, which is
    normally bedtime but can be earlier if the day was compressed.
    """
    episodes: list[Episode] = []
    cursor = start
    shift = config.shift

    outing_probability = config.outing_probability + (
        shift.outing_probability_delta * shift_strength if shift is not None else 0.0
    )
    if rng.random() < min(max(outing_probability, 0.0), 1.0):
        out_start = cursor + timedelta(minutes=float(rng.uniform(30, 150)))
        episodes.append(Episode(cursor, out_start, S.HOME_INACTIVE, "living"))
        out_end = min(
            out_start + timedelta(hours=float(rng.uniform(1.0, 4.0))),
            bedtime - timedelta(hours=1),
        )
        if out_end > out_start:
            episodes.append(Episode(out_start, out_end, S.AWAY, None))
        cursor = max(out_end, out_start)

    lunch_start = max(cursor, _localise(day, 12.5 + rng.normal(0, 0.4), zone))
    if lunch_start > cursor:
        episodes.append(Episode(cursor, lunch_start, S.HOME_INACTIVE, "living"))
    lunch_end = lunch_start + timedelta(minutes=float(rng.uniform(25, 55)))
    episodes.append(Episode(lunch_start, lunch_end, S.KITCHEN_ACTIVITY, "kitchen"))
    cursor = lunch_end

    afternoon_end = min(
        cursor + timedelta(hours=float(rng.uniform(2.0, 4.5))),
        bedtime - timedelta(hours=1.5),
    )
    if afternoon_end > cursor:
        episodes.extend(_alternate(cursor, afternoon_end, rng))
        cursor = afternoon_end

    dinner_end = min(cursor + timedelta(minutes=float(rng.uniform(30, 60))), bedtime)
    if dinner_end > cursor:
        episodes.append(Episode(cursor, dinner_end, S.KITCHEN_ACTIVITY, "kitchen"))
        cursor = dinner_end

    if bedtime > cursor:
        episodes.extend(_alternate(cursor, bedtime, rng))
        cursor = bedtime

    return episodes, cursor


def _alternate(
    start: datetime, end: datetime, rng: np.random.Generator
) -> list[Episode]:
    """Fill a span with alternating active and inactive living-room blocks."""
    episodes: list[Episode] = []
    cursor = start
    active = False
    while cursor < end:
        span = timedelta(minutes=float(rng.uniform(15, 60)))
        block_end = min(cursor + span, end)
        episodes.append(
            Episode(
                cursor,
                block_end,
                S.HOME_ACTIVE if active else S.HOME_INACTIVE,
                "living",
            )
        )
        cursor = block_end
        active = not active
    return episodes


def _plan_visitors(
    day: date, config: HouseholdConfig, rng: np.random.Generator, zone: tzinfo
) -> list[VisitorPeriod]:
    """Generate visitor periods for one day."""
    periods: list[VisitorPeriod] = []
    if config.carer_weekday_visits and day.weekday() < 5:
        start = _localise(day, 9.0 + rng.normal(0, 0.3), zone)
        periods.append(
            VisitorPeriod(
                start,
                start + timedelta(minutes=float(rng.uniform(30, 60))),
                ("hall", "kitchen", "living", "hall"),
            )
        )
    if rng.random() < config.visitor_probability:
        start = _localise(day, 15.0 + rng.normal(0, 1.0), zone)
        periods.append(
            VisitorPeriod(
                start,
                start + timedelta(hours=float(rng.uniform(1.0, 3.0))),
                ("hall", "living", "kitchen", "living", "hall"),
            )
        )
    return periods


@dataclass(frozen=True)
class SimulationResult:
    """A simulated household: what happened, and what the sensors recorded."""

    registry: SensorRegistry
    observations: tuple[Observation, ...]
    truth: GroundTruth
    config: HouseholdConfig = field(repr=False)

    @property
    def start(self) -> datetime:
        """First instant of simulated time."""
        return self.truth.episodes[0].start

    @property
    def end(self) -> datetime:
        """Last instant of simulated time."""
        return self.truth.episodes[-1].end

    def observations_for(self, sensor_ids: Sequence[str]) -> tuple[Observation, ...]:
        """Return only the records from *sensor_ids*, for ablation studies."""
        wanted = set(sensor_ids)
        return tuple(obs for obs in self.observations if obs.sensor_id in wanted)


def _emit_events(
    sensor_id: str,
    modality: Modality,
    times: np.ndarray,
    origin: datetime,
    generated_by: str,
) -> list[Observation]:
    """Turn Poisson arrival offsets into activation observations."""
    return [
        Observation(
            timestamp=origin + timedelta(seconds=float(offset)),
            sensor_id=sensor_id,
            modality=modality,
            kind=ObservationKind.EVENT,
            value=1.0,
            source="sim-hub",
            context={"generated_by": generated_by},
        )
        for offset in times
    ]


def _poisson_times(
    rate_per_hour: float, span: timedelta, rng: np.random.Generator
) -> np.ndarray:
    """Sample event offsets in seconds from a homogeneous Poisson process."""
    seconds = span.total_seconds()
    if seconds <= 0 or rate_per_hour <= 0:
        return np.empty(0)
    expected = rate_per_hour * seconds / 3600.0
    count = int(rng.poisson(expected))
    return np.sort(rng.uniform(0.0, seconds, size=count))


_ROOM_SENSORS = {
    "bedroom": ("bedroom_motion", Modality.MOTION),
    "bathroom": ("bathroom_motion", Modality.MOTION),
    "kitchen": ("kitchen_motion", Modality.MOTION),
    "living": ("living_motion", Modality.MOTION),
}


def _room_observations(
    truth: GroundTruth, rng: np.random.Generator
) -> list[Observation]:
    """Generate motion and contact activations from resident and visitors."""
    records: list[Observation] = []

    for episode in truth.episodes:
        for room, (sensor_id, modality) in _ROOM_SENSORS.items():
            occupied = episode.room == room and episode.state is not S.AWAY
            rate = ACTIVE_RATE if occupied else IDLE_RATE
            if occupied and episode.state in (S.SLEEPING, S.HOME_INACTIVE):
                rate = 6.0 if episode.state is S.HOME_INACTIVE else 0.6
            offsets = _poisson_times(rate, episode.duration, rng)
            records.extend(
                _emit_events(sensor_id, modality, offsets, episode.start, "resident")
            )

        if episode.state is S.KITCHEN_ACTIVITY:
            offsets = _poisson_times(9.0, episode.duration, rng)
            records.extend(
                _emit_events(
                    "fridge_contact",
                    Modality.CONTACT,
                    offsets,
                    episode.start,
                    "resident",
                )
            )

    # Visitors trip the same ambient sensors. This is the contamination that
    # makes attribution necessary rather than decorative.
    for period in truth.visitors:
        span = period.end - period.start
        steps = max(len(period.room_sequence), 1)
        slice_span = span / steps
        for index, room in enumerate(period.room_sequence):
            entry = period.start + slice_span * index
            sensor = _ROOM_SENSORS.get(room)
            if sensor is None:
                continue
            sensor_id, modality = sensor
            offsets = _poisson_times(ACTIVE_RATE * 0.7, slice_span, rng)
            records.extend(_emit_events(sensor_id, modality, offsets, entry, "visitor"))
            if room == "kitchen":
                offsets = _poisson_times(6.0, slice_span, rng)
                records.extend(
                    _emit_events(
                        "fridge_contact",
                        Modality.CONTACT,
                        offsets,
                        entry,
                        "visitor",
                    )
                )
    return records


def _door_observations(truth: GroundTruth) -> list[Observation]:
    """Generate door crossings at every genuine entry and exit."""
    crossings: list[tuple[datetime, str]] = []
    for previous, current in zip(truth.episodes, truth.episodes[1:]):
        if (previous.state is S.AWAY) != (current.state is S.AWAY):
            crossings.append((current.start, "resident"))
    for period in truth.visitors:
        crossings.append((period.start, "visitor"))
        crossings.append((period.end, "visitor"))

    return [
        Observation(
            timestamp=moment,
            sensor_id="front_door",
            modality=Modality.DOOR,
            kind=ObservationKind.EVENT,
            value=1.0,
            source="sim-hub",
            context={"generated_by": who},
        )
        for moment, who in sorted(crossings)
    ]


def _sampled_observations(
    truth: GroundTruth, rng: np.random.Generator, interval: timedelta
) -> list[Observation]:
    """Generate the periodic sampled and state-reporting sensors."""
    records: list[Observation] = []
    cursor = truth.episodes[0].start
    end = truth.episodes[-1].end

    moments: list[datetime] = []
    while cursor < end:
        moments.append(cursor)
        cursor += interval
    states = truth.states_at(moments)

    for moment, state in zip(moments, states):
        if state is None:
            continue
        in_bed = state in (S.SLEEPING, S.BED_AWAKE)
        records.append(
            Observation(
                timestamp=moment,
                sensor_id="bed_pressure",
                modality=Modality.BED_PRESSURE,
                kind=ObservationKind.STATE,
                value=1.0 if in_bed else 0.0,
                source="sim-hub",
                context={"generated_by": "resident"},
            )
        )

        # The radar reports how many people it can track. It is a derived
        # feature with real uncertainty: it occasionally loses a still person
        # and occasionally splits one into two.
        people = (0 if state is S.AWAY else 1) + (1 if truth.visitor_at(moment) else 0)
        if state in (S.SLEEPING, S.HOME_INACTIVE) and rng.random() < 0.12:
            people = max(people - 1, 0)
        elif rng.random() < 0.04:
            people += 1
        records.append(
            Observation(
                timestamp=moment,
                sensor_id="living_radar",
                modality=Modality.RADAR,
                kind=ObservationKind.SAMPLE,
                value=float(people),
                unit=Unit.COUNT,
                confidence=0.75,
                source="sim-radar",
                context={"generated_by": "environment"},
            )
        )

        level = WEARABLE_LEVEL.get(state, 0.5)
        records.append(
            Observation(
                timestamp=moment,
                sensor_id="wearable_motion",
                modality=Modality.WEARABLE_MOTION,
                kind=ObservationKind.SAMPLE,
                value=float(max(rng.normal(level, 0.22), 0.0)),
                source="sim-wearable",
                context={"generated_by": "resident"},
            )
        )
        records.append(
            Observation(
                timestamp=moment,
                sensor_id="resident_beacon",
                modality=Modality.PROXIMITY,
                kind=ObservationKind.STATE,
                value=0.0 if state is S.AWAY else 1.0,
                source="sim-wearable",
                context={"generated_by": "resident"},
            )
        )
    return records


def simulate(config: HouseholdConfig | None = None) -> SimulationResult:
    """Simulate a synthetic household and its sensor record.

    Every run with the same configuration produces byte-identical output, so
    experiments built on it are reproducible from the seed alone.
    """
    settings = config or HouseholdConfig()
    rng = np.random.default_rng(settings.seed)

    episodes: list[Episode] = []
    visitors: list[VisitorPeriod] = []
    for day_index in range(settings.days):
        day = settings.start + timedelta(days=day_index)
        day_episodes, day_visitors = _plan_day(day_index, day, settings, rng)
        episodes.extend(day_episodes)
        visitors.extend(day_visitors)

    episodes = [episode for episode in episodes if episode.duration > timedelta(0)]
    episodes.sort(key=lambda episode: episode.start)
    truth = GroundTruth(tuple(episodes), tuple(visitors), settings.tz)

    records = _room_observations(truth, rng)
    records.extend(_door_observations(truth))
    records.extend(_sampled_observations(truth, rng, timedelta(minutes=1)))
    records.sort(key=lambda obs: obs.timestamp)

    logger.info(
        "Simulated %d days: %d episodes, %d visitor periods, %d observations",
        settings.days,
        len(episodes),
        len(visitors),
        len(records),
    )
    return SimulationResult(
        registry=build_registry(),
        observations=tuple(records),
        truth=truth,
        config=settings,
    )
