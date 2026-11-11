"""Instance model, shared constants and the deterministic instance generator.

All experiment-wide constants live here so that every other module can import
them from a single place.  ``TwoRadiusInstance`` is an immutable description of
one problem: facilities (depots and charging stations) placed in a fixed square,
a flight radius ``R`` for facility--facility edges, a service radius ``r <= R/2``
for facility--demand coverage, and a per-demand requirement of one or two
internally vertex-disjoint depot-to-demand paths.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import cached_property

import networkx as nx


ROOT = "rho"


TOL = 1e-8


ROOT_GAP_TOL = 1e-6


# Absolute value at or above this threshold is a CPLEX infinity sentinel
# (-1.0e+75 / 1.0e+75) rather than a relaxation value.
ROOT_VALUE_SENTINEL = 1e30


# Node-log display interval for the FLOW-IP solve.  Only the root node line is
# needed, so periodic node lines are suppressed to bound the captured log.
MIP_NODE_LOG_INTERVAL = 10 ** 9


# Uniform flight radius for the pre-registered experiment.  A facility layout is
# accepted only when the facility graph, with the virtual root joined to every
# depot, is two-root-connected at this radius; otherwise the layout is discarded
# and resampled.  The radius therefore stays identical across all tiers instead
# of being shrunk per tier through a critical-radius multiplier.
DEFAULT_FLIGHT_RADIUS = 20.0


# Rejection-sampling budget for the facility layout at the fixed radius.
MAX_FACILITY_DRAWS = 500


# The critical radius is recomputed only for small tiers: the binary search runs
# a two-root-connectivity test per step, which is too expensive on the very
# dense facility graphs of the largest tiers.
CRITICAL_RADIUS_MAX_STATIONS = 500


# Compact FLOW-IP memory guard.  When the estimated number of structural
# variables exceeds this budget the benchmark is skipped, the reason is
# recorded, and no IP optimum is produced.
MAX_FLOW_IP_VARIABLES = 20_000_000


FLOW_IP_TOO_LARGE_STATUS = "model_too_large"


# FLOW-IP deliberately not attempted for a tier because the benchmark cannot
# reach optimality there anyway.  The case still reports its algorithm result
# and its LP relaxation lower bound.
FLOW_IP_SKIPPED_STATUS = "skipped_by_request"


DEFAULT_SCALES = (
    (5, 50, 25),
    (10, 100, 50),
    (20, 200, 100),
    (30, 300, 150),
    (50, 500, 250),
)


DEFAULT_SEEDS = (101, 102, 103, 104, 105)


@dataclass(frozen=True)
class TwoRadiusInstance:
    variant: str
    facility_positions: dict[str, tuple[float, float]]
    demand_positions: dict[str, tuple[float, float]]
    depots: tuple[str, ...]
    stations: tuple[str, ...]
    requirements: dict[str, int]
    costs: dict[str, int]
    flight_radius: float
    service_radius: float
    seed: int

    def __post_init__(self) -> None:
        facilities = set(self.depots + self.stations)
        if self.variant not in ("selectable", "preinstalled"):
            raise ValueError("Variant must be selectable or preinstalled")
        if facilities != set(self.facility_positions):
            raise ValueError("Facility partition does not match facility positions")
        if set(self.demand_positions) != set(self.requirements):
            raise ValueError("Demand requirements do not match demand positions")
        if facilities & set(self.demand_positions):
            raise ValueError("Facilities and demands must use distinct identifiers")
        if len(self.depots) < 2:
            raise ValueError("At least two depots are required")
        if self.flight_radius <= 0 or self.service_radius <= 0:
            raise ValueError("Both radii must be positive")
        if 2 * self.service_radius > self.flight_radius + 1e-10:
            raise ValueError("The instance must satisfy 2r <= R")
        if any(value not in (1, 2) for value in self.requirements.values()):
            raise ValueError("Demand requirements must belong to {1, 2}")
        if set(self.costs) != facilities or any(value < 0 for value in self.costs.values()):
            raise ValueError("Every facility must have a nonnegative opening cost")
        if self.variant == "preinstalled" and any(self.costs[d] != 0 for d in self.depots):
            raise ValueError("Preinstalled depots must have zero opening cost")
        for demand, requirement in self.requirements.items():
            if len(self.coverage[demand]) < requirement:
                raise ValueError(f"Demand {demand} has insufficient terminal candidates")

    @cached_property
    def facilities(self) -> tuple[str, ...]:
        return self.depots + self.stations

    @cached_property
    def physical_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (u, v)
            for u, v in itertools.combinations(self.facilities, 2)
            if math.dist(self.facility_positions[u], self.facility_positions[v])
            <= self.flight_radius + 1e-12
        )

    @cached_property
    def backbone_edges(self) -> tuple[tuple[str, str], ...]:
        return self.physical_edges + tuple((ROOT, depot) for depot in self.depots)

    @cached_property
    def coverage(self) -> dict[str, tuple[str, ...]]:
        return {
            demand: tuple(
                facility
                for facility in self.facilities
                if math.dist(self.facility_positions[facility], point)
                <= self.service_radius + 1e-12
            )
            for demand, point in self.demand_positions.items()
        }

    def as_dict(self) -> dict:
        data = asdict(self)
        data["physical_edges"] = self.physical_edges
        data["coverage"] = self.coverage
        return data


def _root_biconnected(
    facilities: tuple[str, ...],
    depots: tuple[str, ...],
    positions: dict[str, tuple[float, float]],
    radius: float,
) -> bool:
    graph = nx.Graph()
    graph.add_nodes_from((ROOT,) + facilities)
    graph.add_edges_from((ROOT, depot) for depot in depots)
    graph.add_edges_from(
        (u, v)
        for u, v in itertools.combinations(facilities, 2)
        if math.dist(positions[u], positions[v]) <= radius + 1e-12
    )
    protected: set[str] = set()
    for block in nx.biconnected_components(graph):
        if ROOT in block and len(block) >= 3:
            protected.update(block)
    return set(facilities) <= protected


def _critical_flight_radius(
    facilities: tuple[str, ...],
    depots: tuple[str, ...],
    positions: dict[str, tuple[float, float]],
) -> float:
    distances = sorted(
        {math.dist(positions[u], positions[v])
         for u, v in itertools.combinations(facilities, 2)}
    )
    low, high = 0, len(distances) - 1
    answer = None
    while low <= high:
        middle = (low + high) // 2
        if _root_biconnected(facilities, depots, positions, distances[middle]):
            answer = distances[middle]
            high = middle - 1
        else:
            low = middle + 1
    if answer is None:
        raise ValueError("Full facility graph is not two-root-connected")
    return answer


def generate_two_radius_instance(
    num_depots: int,
    num_stations: int,
    num_demands: int,
    variant: str,
    requirement_mode: str,
    seed: int,
    *,
    side: float = 100.0,
    eta: float = 0.10,
    fixed_radius: float | None = DEFAULT_FLIGHT_RADIUS,
    service_ratio: float = 0.50,
    mixed_two_probability: float = 0.50,
    depot_cost: tuple[int, int] = (100, 100),
    station_cost: tuple[int, int] = (10, 30),
    max_facility_draws: int = MAX_FACILITY_DRAWS,
    max_demand_draws: int = 100_000,
) -> tuple[TwoRadiusInstance, dict]:
    """Generate facilities first, then demands using distinct R and r.

    Facilities are placed by rejection sampling: a layout is accepted only when
    the facility graph is two-root-connected at ``fixed_radius`` (or, when
    ``fixed_radius`` is None, the legacy critical-radius rule is used).
    Demand locations are then sampled independently subject only to having
    enough covering facilities within ``service_radius``; the model imposes no
    demand--demand distance rule.  For a fixed seed, selectable and
    preinstalled variants share geometry, demand locations, requirements, and
    station costs.
    """
    if (
        num_depots < 2
        or num_stations < 1
        or num_demands < 1
        or side <= 0
        or eta < 0
        or (fixed_radius is not None and fixed_radius <= 0)
        or not 0 < service_ratio <= 0.5
        or requirement_mode not in ("all1", "mixed", "all2")
        or not 0 <= mixed_two_probability <= 1
        or max_facility_draws < 1
        or depot_cost[0] < 0
        or depot_cost[0] > depot_cost[1]
        or station_cost[0] < 0
        or station_cost[0] > station_cost[1]
    ):
        raise ValueError("Invalid generation parameters")
    rng = random.Random(seed)
    depots = tuple(f"d{i}" for i in range(num_depots))
    stations = tuple(f"f{i}" for i in range(num_stations))
    facilities = depots + stations
    positions: dict[str, tuple[float, float]] = {}
    facility_draws = 0
    for facility_draws in range(1, max_facility_draws + 1):
        positions = {}
        for index, depot in enumerate(depots):
            coordinate = rng.uniform(0.08 * side, 0.92 * side)
            positions[depot] = (
                (coordinate, 0.0),
                (side, coordinate),
                (coordinate, side),
                (0.0, coordinate),
            )[index % 4]
        for station in stations:
            positions[station] = (
                rng.uniform(0.04 * side, 0.96 * side),
                rng.uniform(0.04 * side, 0.96 * side),
            )
        if fixed_radius is None or _root_biconnected(
                facilities, depots, positions, fixed_radius):
            break
    else:
        raise ValueError(
            "Cannot sample a facility layout that is two-root-connected at "
            f"radius {fixed_radius} within {max_facility_draws} draws"
        )
    if fixed_radius is None:
        critical_radius = _critical_flight_radius(facilities, depots, positions)
        flight_radius = min(math.sqrt(2) * side, (1 + eta) * critical_radius)
    else:
        flight_radius = fixed_radius
        critical_radius = (
            _critical_flight_radius(facilities, depots, positions)
            if num_stations <= CRITICAL_RADIUS_MAX_STATIONS
            else None
        )
    service_radius = service_ratio * flight_radius

    requirements: dict[str, int] = {}
    for index in range(num_demands):
        if requirement_mode == "all1":
            value = 1
        elif requirement_mode == "all2":
            value = 2
        else:
            value = 2 if rng.random() < mixed_two_probability else 1
        requirements[f"p{index}"] = value
    if requirement_mode == "mixed" and num_demands >= 2:
        requirements["p0"] = 1
        requirements["p1"] = 2

    demand_positions: dict[str, tuple[float, float]] = {}
    draws = 0
    for demand, requirement in requirements.items():
        while draws < max_demand_draws:
            draws += 1
            point = (rng.uniform(0, side), rng.uniform(0, side))
            neighbors = [
                facility for facility in facilities
                if math.dist(point, positions[facility]) <= service_radius + 1e-12
            ]
            if len(neighbors) >= requirement:
                demand_positions[demand] = point
                break
        else:
            raise ValueError("Cannot place demands with the requested coverage")

    # Draw every raw cost for both variants before overriding preinstalled
    # depot costs.  This keeps station costs paired across the two variants.
    raw_costs = {
        facility: rng.randint(
            *(depot_cost if facility in depots else station_cost)
        )
        for facility in facilities
    }
    costs = dict(raw_costs)
    if variant == "preinstalled":
        costs.update({depot: 0 for depot in depots})
    instance = TwoRadiusInstance(
        variant=variant,
        facility_positions=positions,
        demand_positions=demand_positions,
        depots=depots,
        stations=stations,
        requirements=requirements,
        costs=costs,
        flight_radius=flight_radius,
        service_radius=service_radius,
        seed=seed,
    )
    paired_payload = {
        "facility_positions": positions,
        "demand_positions": demand_positions,
        "requirements": requirements,
        "station_costs": {station: costs[station] for station in stations},
        "flight_radius": flight_radius,
        "service_radius": service_radius,
    }
    paired_instance_sha256 = hashlib.sha256(
        json.dumps(
            paired_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "critical_flight_radius": critical_radius,
        "critical_flight_radius_computed": critical_radius is not None,
        "fixed_flight_radius": fixed_radius,
        "facility_layout_draws": facility_draws,
        "eta": eta,
        "service_ratio_r_over_R": service_ratio,
        "two_r_over_R": 2 * service_radius / flight_radius,
        "facility_graph_edges": len(instance.physical_edges),
        "facility_graph_average_degree": (
            2 * len(instance.physical_edges) / len(instance.facilities)
        ),
        "coverage_degree_min": min(map(len, instance.coverage.values())),
        "coverage_degree_mean": (
            sum(map(len, instance.coverage.values())) / len(instance.coverage)
        ),
        "coverage_degree_max": max(map(len, instance.coverage.values())),
        "demand_draws": draws,
        "demand_minimum_separation_imposed": False,
        "paired_instance_id": (
            f"D{num_depots}_F{num_stations}_P{num_demands}_seed{seed}"
        ),
        "paired_instance_sha256": paired_instance_sha256,
    }
    return instance, metadata
