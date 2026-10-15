"""Planning for the matched-sensor protocol: TIHM's sensors, measured and matched.

The simulator's event sensors fire at rates written into its code. This module
measures, from the TIHM activity file alone, what TIHM's event sensors do, and
chooses the :class:`~sensor_modeling.simulation.sensor_profile.SensorProfile`
whose simulated record looks most like it.

**What is read.** ``Activity.csv`` only: each row's home, location and time.
No label, no physiology, no sleep-mat record and no output of the pipeline is
read, so nothing here says how well the pipeline does on TIHM.

**What is measured.** For each home, on its days with at least one record other
than its first and last:

- the activations a day of each room's motion sensor: kitchen, bathroom,
  bedroom, living room and hallway;
- the share of motion activations whose previous motion activation came from
  another room's sensor;
- the bedroom's motion activations between midnight and six in the morning, a
  night;
- the median gap between two activations of one motion sensor less than ten
  minutes apart.

Each is summarised over homes by its median, a home with no such sensor left
out. Two facts are measured over the whole file: the shortest gap between two
activations of one motion sensor, which is the hold-off, and the share of gaps
between two rows of one contact that are under a minute.

**What is chosen.** The hold-off, the hallway and the paired contact rows are
set from the facts. The presence scale and the spill-over rate are chosen on a
grid, on simulated homes whose seeds belong to no protocol, as the point whose
eight medians are nearest TIHM's on the logarithmic scale.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..evaluation import ExperimentRecord
from ..external.tihm import FILES, TIMEZONE, check_files
from ..fusion import ONLINE, NotEnumerated
from ..observations.observation import Observation
from ..simulation.faults import DegradationConfig, degrade
from ..simulation.household import HouseholdConfig, simulate
from ..simulation.sensor_profile import (
    FRIDGE,
    FRONT_DOOR,
    HALLWAY,
    ROOM_SENSORS,
    SensorProfile,
    profile_observations,
)
from .silent_home_protocol import event_sensors

EXPERIMENT = "matched-sensors-planning"
RESULT_SCHEMA = "matched-sensors-planning/1"

#: The motion sensors' roles, in the order the moments are reported.
ROLES = ("kitchen", "bathroom", "bedroom", "living", "hall")

#: TIHM's locations and the simulated sensors, by role. Anything else is a
#: contact or a door.
TIHM_ROLES = {
    "Kitchen": "kitchen",
    "Bathroom": "bathroom",
    "Bedroom": "bedroom",
    "Lounge": "living",
    "Hallway": "hall",
}
TIHM_CONTACTS = ("Fridge Door", "Front Door", "Back Door")
SIMULATED_ROLES = {sensor: room for room, sensor in ROOM_SENSORS.items()}
SIMULATED_ROLES[HALLWAY] = "hall"
CONTACT = "contact"

#: The moments, in order.
PER_DAY = tuple(f"{role}_per_day" for role in ROLES)
SWITCH_SHARE = "switch_share"
NIGHT_BEDROOM = "night_bedroom"
RETRIGGER_GAP = "retrigger_gap_seconds"
MOMENTS = (*PER_DAY, SWITCH_SHARE, NIGHT_BEDROOM, RETRIGGER_GAP)

#: The night's hours, and the longest gap read as a retrigger.
NIGHT_HOURS = (0, 6)
RETRIGGER_WINDOW_SECONDS = 600.0

#: A simulated median of zero is read as this, so its distance is finite.
FLOOR = 1e-3


@dataclass(frozen=True)
class Planning:
    """The planning's settings.

    Attributes
    ----------
    seed_root, homes, days
        The simulated homes the grid is tried on: the first distinct values of
        ``SeedSequence(seed_root).generate_state(4 * homes)`` modulo 1,000,000.
    presence_scales, spill_rates
        The grid.
    """

    seed_root: int = 20261009
    homes: int = 30
    days: int = 42
    presence_scales: tuple[float, ...] = (
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        1.0,
        1.25,
        1.5,
        2.0,
    )
    spill_rates: tuple[float, ...] = (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
    )

    def seeds(self) -> tuple[int, ...]:
        """The simulated homes' seeds."""
        state = np.random.SeedSequence(self.seed_root).generate_state(
            4 * self.homes, dtype=np.uint32
        )
        unique: list[int] = []
        for value in state:
            seed = int(value % 1_000_000)
            if seed not in unique:
                unique.append(seed)
            if len(unique) == self.homes:
                break
        return tuple(sorted(unique))


# ----------------------------------------------------------------------------
# The moments of one record
# ----------------------------------------------------------------------------
def home_moments(
    records: Sequence[tuple[datetime, str, str]], zone: tzinfo
) -> dict[str, float | None]:
    """One home's moments, from its records: aware moment, sensor and role.

    A role is one of :data:`ROLES` or :data:`CONTACT`. A moment about a role
    the home has no sensor for is ``None``.
    """
    ordered = sorted(records)
    days = sorted({moment.astimezone(zone).date() for moment, _, _ in ordered})
    counted = set(days[1:-1])
    present = {role for _, _, role in ordered if role != CONTACT}
    result: dict[str, float | None] = dict.fromkeys(MOMENTS, None)
    if not counted:
        return result
    motion = [
        (moment, sensor, role)
        for moment, sensor, role in ordered
        if role != CONTACT and moment.astimezone(zone).date() in counted
    ]
    counts: dict[str, int] = defaultdict(int)
    night = 0
    for moment, _, role in motion:
        counts[role] += 1
        local = moment.astimezone(zone)
        if role == "bedroom" and NIGHT_HOURS[0] <= local.hour < NIGHT_HOURS[1]:
            night += 1
    for role, name in zip(ROLES, PER_DAY):
        if role in present:
            result[name] = counts[role] / len(counted)
    if len(motion) >= 2:
        switches = sum(
            1 for before, after in zip(motion, motion[1:]) if before[2] != after[2]
        )
        result[SWITCH_SHARE] = switches / (len(motion) - 1)
    if "bedroom" in present:
        result[NIGHT_BEDROOM] = night / len(counted)
    last: dict[str, datetime] = {}
    gaps: list[float] = []
    for moment, sensor, _ in motion:
        if sensor in last:
            gap = (moment - last[sensor]).total_seconds()
            if gap < RETRIGGER_WINDOW_SECONDS:
                gaps.append(gap)
        last[sensor] = moment
    if gaps:
        result[RETRIGGER_GAP] = float(np.median(gaps))
    return result


def cohort_moments(homes: Iterable[Mapping[str, float | None]]) -> dict[str, Any]:
    """Each moment's median over the homes that have it, and their number."""
    rows = list(homes)
    summary: dict[str, Any] = {}
    for name in MOMENTS:
        values = [float(value) for row in rows if (value := row[name]) is not None]
        summary[name] = {
            "median": float(np.median(values)) if values else None,
            "homes": len(values),
        }
    return summary


def distance(simulated: Mapping[str, Any], target: Mapping[str, Any]) -> float:
    """The sum over moments of the squared log ratio of the two medians."""
    total = 0.0
    for name in MOMENTS:
        aim = target[name]["median"]
        got = simulated[name]["median"]
        if aim is None or aim <= 0:
            continue
        total += math.log(max(got if got is not None else 0.0, FLOOR) / aim) ** 2
    return total


# ----------------------------------------------------------------------------
# TIHM
# ----------------------------------------------------------------------------
def read_tihm(directory: Path) -> dict[str, list[tuple[datetime, str, str]]]:
    """Every home's records from the activity file, with each one's role."""
    problems = check_files(directory)
    if "Activity.csv" in problems:
        raise ValueError(f"Activity.csv {problems['Activity.csv']}")
    zone = ZoneInfo(TIMEZONE)
    homes: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    with (Path(directory) / "Activity.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["patient_id", "location_name", "date"]:
            raise ValueError(f"Activity.csv has columns {header}")
        for home, location, moment in reader:
            when = datetime.strptime(moment, "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone)
            homes[home].append((when, location, TIHM_ROLES.get(location, CONTACT)))
    return dict(homes)


def sensor_facts(
    homes: Mapping[str, Sequence[tuple[datetime, str, str]]],
) -> dict[str, Any]:
    """The hold-off of each motion location, and how contacts pair their rows."""
    by_sensor: dict[str, list[float]] = defaultdict(list)
    for records in homes.values():
        last: dict[str, datetime] = {}
        for moment, sensor, _ in sorted(records):
            if sensor in last:
                by_sensor[sensor].append((moment - last[sensor]).total_seconds())
            last[sensor] = moment
    motion = {
        location: {
            "gaps": len(by_sensor[location]),
            "shortest_seconds": float(min(by_sensor[location])),
        }
        for location in TIHM_ROLES
        if by_sensor[location]
    }
    contacts = {}
    for location in TIHM_CONTACTS:
        gaps = np.array(by_sensor[location])
        if gaps.size:
            short = gaps[gaps < 60.0]
            contacts[location] = {
                "gaps": int(gaps.size),
                "share_under_a_minute": float(short.size / gaps.size),
                "median_gap_under_a_minute_seconds": (
                    float(np.median(short)) if short.size else None
                ),
            }
    return {
        "motion": motion,
        "hold_off_seconds": min(row["shortest_seconds"] for row in motion.values()),
        "contacts": contacts,
    }


# ----------------------------------------------------------------------------
# Simulated homes
# ----------------------------------------------------------------------------
def _role(sensor: str) -> str:
    return SIMULATED_ROLES.get(sensor, CONTACT)


def simulated_records(
    observations: Iterable[Observation],
) -> list[tuple[datetime, str, str]]:
    """A simulated record as moment, sensor and role."""
    return [(o.timestamp, o.sensor_id, _role(o.sensor_id)) for o in observations]


def planning_truths(planning: Planning) -> list[tuple[int, Any, list[Observation]]]:
    """Each planning home's seed, plan and the simulator's own event record."""
    sensors = event_sensors()
    homes = []
    for seed in planning.seeds():
        result = simulate(HouseholdConfig(days=planning.days, seed=seed))
        delivered, _ = degrade(result.observations_for(sensors), DegradationConfig())
        homes.append((seed, result.truth, delivered))
    return homes


def profile_moments(
    homes: Sequence[tuple[int, Any, list[Observation]]], profile: SensorProfile
) -> dict[str, Any]:
    """The cohort moments of the planning homes drawn under *profile*."""
    zone = HouseholdConfig().tz
    return cohort_moments(
        home_moments(
            simulated_records(profile_observations(truth, seed, profile)), zone
        )
        for seed, truth, _ in homes
    )


def plan(planning: Planning, directory: Path) -> dict[str, Any]:
    """Measure TIHM, try the grid and choose the profile."""
    tihm = read_tihm(directory)
    zone = ZoneInfo(TIMEZONE)
    target = cohort_moments(home_moments(records, zone) for records in tihm.values())
    facts = sensor_facts(tihm)
    homes = planning_truths(planning)
    standard = cohort_moments(
        home_moments(simulated_records(delivered), HouseholdConfig().tz)
        for _, _, delivered in homes
    )
    grid = []
    for scale in planning.presence_scales:
        for spill in planning.spill_rates:
            profile = SensorProfile(
                hold_off_seconds=facts["hold_off_seconds"],
                presence_scale=scale,
                spill_rate=spill,
                hallway=True,
                paired_contacts=True,
            )
            moments = profile_moments(homes, profile)
            grid.append(
                {
                    "presence_scale": scale,
                    "spill_rate": spill,
                    "distance": distance(moments, target),
                    "moments": moments,
                }
            )
    best = min(
        grid,
        key=lambda row: (row["distance"], row["presence_scale"], row["spill_rate"]),
    )
    return {
        "tihm": {
            "homes": len(tihm),
            "moments": target,
            "sensor_facts": facts,
            "files": {"Activity.csv": FILES["Activity.csv"]},
        },
        "standard": {"moments": standard, "distance": distance(standard, target)},
        "grid": grid,
        "chosen": {
            "presence_scale": best["presence_scale"],
            "spill_rate": best["spill_rate"],
            "hold_off_seconds": facts["hold_off_seconds"],
            "hallway": True,
            "paired_contacts": True,
            "distance": best["distance"],
            "moments": best["moments"],
            "at_the_edge_of_the_grid": best["presence_scale"]
            in (planning.presence_scales[0], planning.presence_scales[-1])
            or best["spill_rate"]
            in (planning.spill_rates[0], planning.spill_rates[-1]),
        },
        "seeds": list(planning.seeds()),
    }


def run_planning(
    directory: Path, planning: Planning | None = None, *, output_dir: Path | None = None
) -> tuple[ExperimentRecord, Path | None]:
    """Run the planning and write its record."""
    planning = planning or Planning()
    results = plan(planning, directory)
    record = ExperimentRecord(
        experiment=EXPERIMENT,
        configuration={
            **asdict(planning),
            "result_schema": RESULT_SCHEMA,
            "moments": list(MOMENTS),
            "night_hours": list(NIGHT_HOURS),
            "retrigger_window_seconds": RETRIGGER_WINDOW_SECONDS,
            "contacts_in_the_simulator": [FRIDGE, FRONT_DOOR],
        },
        inference=ONLINE,
        evidence=NotEnumerated(
            "no pipeline is run: the moments are counts and gaps of sensor "
            "records, TIHM's and the simulator's"
        ),
        seeds=[planning.seed_root],
        results=results,
        data_source="TIHM Activity.csv and simulated",
        notes=[
            "Reads TIHM's Activity.csv only. No label, physiology, sleep-mat "
            "record or output of the pipeline is read.",
            "Made before the matched-sensor protocol was frozen, to choose the "
            "profile it declares.",
            "The simulated homes' seeds belong to no protocol. Each home's plan "
            "is simulated once and its event sensors are drawn again under each "
            "point of the grid.",
            "The record holds medians over homes and per-location gap counts, "
            "never a home's records, so the dataset is not redistributed.",
        ],
    )
    path = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = record.write(Path(output_dir) / f"{EXPERIMENT}.json")
    return record, path


def nearest(
    grid: Sequence[Mapping[str, Any]], scale: float, spill: float
) -> Mapping[str, Any]:
    """The grid row at a point."""
    for row in grid:
        if math.isclose(row["presence_scale"], scale) and math.isclose(
            row["spill_rate"], spill
        ):
            return row
    raise KeyError((scale, spill))


__all__ = [
    "MOMENTS",
    "Planning",
    "cohort_moments",
    "distance",
    "home_moments",
    "plan",
    "read_tihm",
    "run_planning",
    "sensor_facts",
]
