"""Event sensors drawn again from a simulated truth, under a declared profile.

:func:`~sensor_modeling.simulation.household.simulate` plans every day first
and then draws the sensor record from that plan. This module draws the event
sensors' record again from the same plan, under a :class:`SensorProfile` that
states how the sensors behave, so that two records of one home differ in the
sensors and not in the resident.

What a profile can change:

*A hold-off.* A passive infrared sensor that has just reported stays quiet for
a while, whatever happens in front of it. Activations of one motion sensor
closer together than ``hold_off_seconds`` to the last one it reported are not
reported.

*How much presence makes a sensor fire.* ``presence_scale`` multiplies the
rates at which a room's motion sensor fires while the resident is in that room,
active, still or asleep, and while a visitor is.

*Spill-over.* While the resident is at home and awake, a motion sensor of a
room they are not in fires at ``spill_rate`` an hour, as one that sees through
a doorway or into an open-plan space would: every such sensor, the bathroom's
and the hallway's among them, at the same rate whether the resident is moving
or still. Asleep, out, or with no resident at home, it fires at the
simulator's idle rate. A visitor causes no spill-over.

*A hallway.* A motion sensor in a hallway reports once at each change of the
resident's room, leaving or coming home included, and once at each change of a
visitor's; it fires at the visitor rate while a visitor is in the hallway, and
otherwise spills over like any other.

*Paired contact rows.* A contact that logs its opening and its closing as two
rows records each activation of the fridge or the entrance door twice, the
second row 1 to 17 seconds after the first, uniformly, so the median gap is 9
seconds.

The profile draws from its own generator, seeded with the home's seed and a
stream, so the record it gives depends on the plan, the seed, the stream and
the profile alone. With :data:`STANDARD_PROFILE` it draws the event sensors by
the same rules as the simulator, not the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..observations.observation import Observation
from ..observations.registry import SensorRegistry, SensorSpec
from ..observations.types import Modality, ObservationKind
from ..states.ontology import BehaviouralState
from .household import ACTIVE_RATE, IDLE_RATE, GroundTruth

S = BehaviouralState

#: The simulator's in-room rates per hour: active, still and asleep, and the
#: share of the active rate a visitor produces.
INACTIVE_RATE = 6.0
ASLEEP_RATE = 0.6
VISITOR_SHARE = 0.7

#: The fridge's rates per hour in the kitchen, for the resident and a visitor.
FRIDGE_RATE = 9.0
VISITOR_FRIDGE_RATE = 6.0

#: The rooms with a motion sensor in the simulated home, and their sensors.
ROOM_SENSORS: dict[str, str] = {
    "bedroom": "bedroom_motion",
    "bathroom": "bathroom_motion",
    "kitchen": "kitchen_motion",
    "living": "living_motion",
}
HALLWAY = "hall_motion"
HALLWAY_ROOM = "hall"
FRIDGE = "fridge_contact"
FRONT_DOOR = "front_door"

#: The second row of a paired contact comes this many seconds after the first.
PAIR_GAP_SECONDS = (1.0, 17.0)

#: The generator stream a profile draws from, beside the home's seed.
PROFILE_STREAM = 7


@dataclass(frozen=True)
class SensorProfile:
    """How the event sensors behave.

    Attributes
    ----------
    hold_off_seconds
        The shortest gap between two activations one motion sensor reports.
    presence_scale
        The multiple of the simulator's in-room rates.
    spill_rate
        Activations per hour of a motion sensor outside the resident's room
        while the resident is at home and awake.
    hallway
        Whether the home has a hallway motion sensor.
    paired_contacts
        Whether a contact records each activation as two rows.
    """

    hold_off_seconds: float = 0.0
    presence_scale: float = 1.0
    spill_rate: float = IDLE_RATE
    hallway: bool = False
    paired_contacts: bool = False

    def __post_init__(self) -> None:
        """Validate the profile."""
        if self.hold_off_seconds < 0.0:
            raise ValueError("hold_off_seconds must be non-negative")
        if self.presence_scale <= 0.0:
            raise ValueError("presence_scale must be positive")
        if self.spill_rate < 0.0:
            raise ValueError("spill_rate must be non-negative")

    def to_dict(self) -> dict[str, float | bool]:
        """The profile, in a stable serialisable form."""
        return {
            "hold_off_seconds": self.hold_off_seconds,
            "presence_scale": self.presence_scale,
            "spill_rate": self.spill_rate,
            "hallway": self.hallway,
            "paired_contacts": self.paired_contacts,
        }


#: The simulator's own rules.
STANDARD_PROFILE = SensorProfile()


def profile_registry(
    registry: SensorRegistry, profile: SensorProfile
) -> SensorRegistry:
    """*registry*, with the hallway's motion sensor when the profile has one."""
    specs = list(registry)
    if profile.hallway and HALLWAY not in registry:
        specs.append(
            SensorSpec(
                HALLWAY,
                Modality.MOTION,
                room=HALLWAY_ROOM,
                description="PIR covering the hallway every room change passes.",
            )
        )
    return SensorRegistry.from_specs(specs)


def _times(
    rate_per_hour: float, span: timedelta, rng: np.random.Generator
) -> np.ndarray:
    seconds = span.total_seconds()
    if seconds <= 0 or rate_per_hour <= 0:
        return np.empty(0)
    count = int(rng.poisson(rate_per_hour * seconds / 3600.0))
    return np.sort(rng.uniform(0.0, seconds, size=count))


def _event(moment: datetime, sensor: str, modality: Modality, who: str) -> Observation:
    return Observation(
        timestamp=moment,
        sensor_id=sensor,
        modality=modality,
        kind=ObservationKind.EVENT,
        value=1.0,
        source="sim-hub",
        context={"generated_by": who},
    )


def _awake_at_home(state: BehaviouralState) -> bool:
    return state not in (S.SLEEPING, S.AWAY)


def _in_room_rate(state: BehaviouralState, profile: SensorProfile) -> float:
    if state is S.SLEEPING:
        base = ASLEEP_RATE
    elif state is S.HOME_INACTIVE:
        base = INACTIVE_RATE
    else:
        base = ACTIVE_RATE
    return base * profile.presence_scale


def _motion(
    truth: GroundTruth, profile: SensorProfile, rng: np.random.Generator
) -> list[tuple[datetime, str, str]]:
    """Every motion activation before the hold-off: moment, sensor, source."""
    sensors = dict(ROOM_SENSORS)
    if profile.hallway:
        sensors[HALLWAY_ROOM] = HALLWAY
    found: list[tuple[datetime, str, str]] = []
    opening = truth.episodes[0] if truth.episodes else None
    previous_room: str | None = (
        opening.room if opening is not None and opening.state is not S.AWAY else None
    )
    for episode in truth.episodes:
        home = episode.state is not S.AWAY
        for room, sensor in sensors.items():
            if home and episode.room == room:
                rate = _in_room_rate(episode.state, profile)
            elif home and _awake_at_home(episode.state):
                rate = profile.spill_rate
            else:
                rate = IDLE_RATE
            for offset in _times(rate, episode.duration, rng):
                found.append(
                    (
                        episode.start + timedelta(seconds=float(offset)),
                        sensor,
                        "resident",
                    )
                )
        where = episode.room if home else None
        if profile.hallway and where != previous_room and (where or previous_room):
            found.append((episode.start, HALLWAY, "resident"))
        previous_room = where
    for period in truth.visitors:
        steps = max(len(period.room_sequence), 1)
        piece = (period.end - period.start) / steps
        last: str | None = None
        for index, visited in enumerate(period.room_sequence):
            entry = period.start + piece * index
            if profile.hallway and visited != last:
                found.append((entry, HALLWAY, "visitor"))
            last = visited
            seen_by = sensors.get(visited)
            if seen_by is None:
                continue
            rate = ACTIVE_RATE * VISITOR_SHARE * profile.presence_scale
            for offset in _times(rate, piece, rng):
                found.append(
                    (entry + timedelta(seconds=float(offset)), seen_by, "visitor")
                )
    return found


def hold_off(
    activations: list[tuple[datetime, str, str]], seconds: float
) -> list[tuple[datetime, str, str]]:
    """The activations a motion sensor reports, given its hold-off.

    An activation closer than *seconds* to the last one its sensor reported is
    not reported; the hold-off restarts only at a reported activation.
    """
    if seconds <= 0.0:
        return sorted(activations)
    last: dict[str, datetime] = {}
    kept: list[tuple[datetime, str, str]] = []
    for moment, sensor, who in sorted(activations):
        previous = last.get(sensor)
        if previous is not None and (moment - previous).total_seconds() < seconds:
            continue
        last[sensor] = moment
        kept.append((moment, sensor, who))
    return kept


def _contacts(
    truth: GroundTruth, rng: np.random.Generator
) -> list[tuple[datetime, str, Modality, str]]:
    found: list[tuple[datetime, str, Modality, str]] = []
    for episode in truth.episodes:
        if episode.state is S.KITCHEN_ACTIVITY:
            for offset in _times(FRIDGE_RATE, episode.duration, rng):
                found.append(
                    (
                        episode.start + timedelta(seconds=float(offset)),
                        FRIDGE,
                        Modality.CONTACT,
                        "resident",
                    )
                )
    for period in truth.visitors:
        steps = max(len(period.room_sequence), 1)
        piece = (period.end - period.start) / steps
        for index, room in enumerate(period.room_sequence):
            if room != "kitchen":
                continue
            entry = period.start + piece * index
            for offset in _times(VISITOR_FRIDGE_RATE, piece, rng):
                found.append(
                    (
                        entry + timedelta(seconds=float(offset)),
                        FRIDGE,
                        Modality.CONTACT,
                        "visitor",
                    )
                )
    for previous, current in zip(truth.episodes, truth.episodes[1:]):
        if (previous.state is S.AWAY) != (current.state is S.AWAY):
            found.append((current.start, FRONT_DOOR, Modality.DOOR, "resident"))
    for period in truth.visitors:
        found.append((period.start, FRONT_DOOR, Modality.DOOR, "visitor"))
        found.append((period.end, FRONT_DOOR, Modality.DOOR, "visitor"))
    return found


def profile_observations(
    truth: GroundTruth, seed: int, profile: SensorProfile, stream: int = 0
) -> list[Observation]:
    """The event sensors' record of *truth* under *profile*, in time order.

    The draws come from a generator seeded with *seed*, :data:`PROFILE_STREAM`
    and *stream*, so the record is a function of the plan, the seed, the
    stream and the profile. Two records of one home drawn on different streams
    share no sensor noise.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), PROFILE_STREAM, int(stream)])
    )
    motion = hold_off(_motion(truth, profile, rng), profile.hold_off_seconds)
    records = [
        _event(moment, sensor, Modality.MOTION, who) for moment, sensor, who in motion
    ]
    for moment, sensor, modality, who in _contacts(truth, rng):
        records.append(_event(moment, sensor, modality, who))
        if profile.paired_contacts:
            gap = float(rng.uniform(*PAIR_GAP_SECONDS))
            records.append(
                _event(moment + timedelta(seconds=gap), sensor, modality, who)
            )
    records.sort(key=lambda obs: (obs.timestamp, obs.sensor_id))
    return records
