"""Two-radius computational experiments for the revised ORL manuscript.

Facilities form the backbone with flight radius R. Demands are not backbone
vertices and are served only through terminal facilities within service radius r.
Stage 1 uses deterministic weighted greedy set cover; Stage 2 performs cutting-plane
iterative rounding on the rooted undirected cut (UCUT) linear program LP(J).
Exact node-cost benchmarking uses one run of the compact node-capacitated
multi-commodity FLOW-IP from Section 2 of the manuscript.  The initial root
relaxation value is captured from that same MIP run; no standalone FLOW-LP is
solved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

import cplex
import networkx as nx
# numpy/scipy are imported once here rather than lazily inside the separator.
# Importing them on first use charged the whole library load (~1 s on this
# machine) to whichever case ran first, which is always the algorithm path.
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph as csgraph_module


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


def _new_model(
    time_limit: float,
    mip: bool = False,
    strict_feasibility: bool = False,
    lp_method: str = "dual",
    capture_log: bool = False,
) -> cplex.Cplex:
    model = cplex.Cplex()
    model.objective.set_sense(model.objective.sense.minimize)
    # On this CPLEX build the log channel shares the results channel: calling
    # set_results_stream(None) silently disables the whole CPLEX log, so the
    # MIP node log cannot be captured any more.  The FLOW-IP solve therefore
    # keeps both streams live and isolates the console output through an
    # os.dup2 redirect instead (see _CapturedSolverLog).
    if not capture_log:
        model.set_log_stream(None)
        model.set_results_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.parameters.threads.set(1)
    model.parameters.parallel.set(1)
    model.parameters.timelimit.set(max(0.01, time_limit))
    model.parameters.advance.set(1)
    if mip:
        model.parameters.mip.tolerances.mipgap.set(1e-7)
        model.parameters.mip.tolerances.absmipgap.set(1e-7)
    else:
        if lp_method == "dual":
            model.parameters.lpmethod.set(model.parameters.lpmethod.values.dual)
        # lp_method="default" keeps the CPLEX setting, which is what the MIP
        # branch (and therefore every IP benchmark) uses.
        if strict_feasibility:
            # UCUT rows count as violated only beyond 1e-8. The master must
            # satisfy previously generated rows more accurately than that.
            # FLOW-IP keeps the CPLEX default feasibility tolerance.
            model.parameters.simplex.tolerances.feasibility.set(1e-9)
    return model


def _remaining(deadline: float) -> float:
    value = deadline - time.perf_counter()
    if value <= 0:
        raise TimeoutError("Case time limit exceeded")
    return value


def _add_rows(
    model: cplex.Cplex,
    rows: Iterable[tuple[dict[int, float], str, float]],
) -> None:
    materialized = list(rows)
    if not materialized:
        return
    model.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(ind=list(coefficients), val=list(coefficients.values()))
            for coefficients, _, _ in materialized
        ],
        senses="".join(sense for _, sense, _ in materialized),
        rhs=[rhs for _, _, rhs in materialized],
    )


class _CutOracle:
    """Small undirected min-cut oracle with reusable topology."""

    def __init__(
        self,
        nodes: Iterable[str],
        edges: Iterable[tuple[str, str, int]],
    ) -> None:
        self.nodes = tuple(dict.fromkeys(nodes))
        self.node_index = {
            node: index for index, node in enumerate(self.nodes)
        }
        self.edges = tuple(
            (self.node_index[u], self.node_index[v], index)
            for u, v, index in edges
        )
        self.calls = 0

    def minimum_cut(
        self,
        values: list[float],
        source: str,
        sink: str,
        threshold: float,
        removed: str | None = None,
    ) -> tuple[float, dict[int, float] | None]:
        """Return one violated cut, stopping once the threshold is certified."""
        self.calls += 1
        if source == removed or sink == removed:
            return 0.0, {}
        source_index = self.node_index.get(source)
        sink_index = self.node_index.get(sink)
        removed_index = self.node_index.get(removed) if removed is not None else -1
        if source_index is None or sink_index is None:
            return 0.0, {}

        size = len(self.nodes)
        adjacency: list[list[list[float | int]]] = [[] for _ in range(size)]

        def add_arc(u: int, v: int, capacity: float) -> None:
            forward: list[float | int] = [v, len(adjacency[v]), capacity]
            reverse: list[float | int] = [u, len(adjacency[u]), 0.0]
            adjacency[u].append(forward)
            adjacency[v].append(reverse)

        for u, v, index in self.edges:
            if u == removed_index or v == removed_index:
                continue
            capacity = max(0.0, values[index])
            if capacity <= TOL:
                continue
            add_arc(u, v, capacity)
            add_arc(v, u, capacity)

        flow = 0.0
        while flow < threshold - TOL:
            level = [-1] * size
            level[source_index] = 0
            queue = deque([source_index])
            while queue:
                u = queue.popleft()
                for edge in adjacency[u]:
                    v, _, capacity = edge
                    if (
                        v != removed_index
                        and capacity > TOL
                        and level[v] < 0
                    ):
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink_index] < 0:
                break

            cursor = [0] * size

            def send(u: int, pushed: float) -> float:
                if u == sink_index:
                    return pushed
                while cursor[u] < len(adjacency[u]):
                    edge = adjacency[u][cursor[u]]
                    v, reverse_index, capacity = edge
                    if capacity > TOL and level[v] == level[u] + 1:
                        sent = send(v, min(pushed, float(capacity)))
                        if sent > TOL:
                            edge[2] = float(capacity) - sent
                            reverse = adjacency[v][int(reverse_index)]
                            reverse[2] = float(reverse[2]) + sent
                            return sent
                    cursor[u] += 1
                return 0.0

            while flow < threshold - TOL:
                sent = send(source_index, threshold - flow)
                if sent <= TOL:
                    break
                flow += sent

        if flow >= threshold - TOL:
            return flow, None

        reachable = {source_index}
        queue = deque([source_index])
        while queue:
            u = queue.popleft()
            for v, _, capacity in adjacency[u]:
                if (
                    v != removed_index
                    and capacity > TOL
                    and v not in reachable
                ):
                    reachable.add(int(v))
                    queue.append(int(v))
        coefficients = {
            index: 1.0
            for u, v, index in self.edges
            if u != removed_index
            and v != removed_index
            and ((u in reachable) != (v in reachable))
        }
        return flow, coefficients



class _SplitCutOracle:
    """Exact rooted UCUT separator: one capacitated split graph per terminal.

    A cut of value below k_t contains either zero or one unit-capacity split
    node, yielding an ordinary cut or a deleted-vertex cut respectively. Cut
    values at or above k_t need not match the original UCUT expression.

    Both backends share the split-graph semantics and optimisation A1 (only the
    current positive support of ``x`` carries capacity, so zero-capacity arcs
    never cost anything on the Python side):

    * ``"python"`` -- A1 + A3: compact per-call arc arrays feeding a flat-array
      iterative Dinic that never recurses and uses level version stamps instead
      of wiping the level array at every phase.
    * ``"scipy"`` -- A1 + A2 + A3: the same split graph is handed to the C
      routine ``scipy.sparse.csgraph.maximum_flow`` on integer-scaled
      capacities, and the minimum cut is recovered from the residual graph.

    ``cut_mode`` controls how many UCUT rows a single max flow yields:
    ``"single"`` (only the smallest minimum cut), ``"double"`` (the smallest
    and the largest minimum cut) and ``"nested"`` (additional intermediate
    minimum cuts taken from the residual lattice).
    """

    # Integer scaling for the scipy backend. scipy.sparse.csgraph.maximum_flow
    # stores capacities as int32, so any value >= 2**31 wraps silently and
    # returns a wrong flow. The target's split arc carries value 2*SCALE (for
    # k=2 terminals), hence SCALE must satisfy 2*SCALE <= 2**31 - 1; 10**9
    # leaves margin. Per-arc flooring is less than 1/SCALE, so a cut crossing
    # many arcs can accumulate error above TOL=1e-8. Flooring gives a
    # one-sided certificate; ambiguous sub-threshold cuts use exact Dinic.
    SCALE = 10 ** 9

    def __init__(
        self,
        nodes: Iterable[str],
        edges: Iterable[tuple[str, str, int]],
        backend: str = "scipy",
        cut_mode: str = "double",
        nested_levels: int = 4,
    ) -> None:
        self.backend = backend
        self.cut_mode = cut_mode
        self.nested_levels = max(1, int(nested_levels))
        self.nodes = tuple(dict.fromkeys(nodes))
        self.facilities = tuple(v for v in self.nodes if v != ROOT)
        self.number = {ROOT: 0}
        for vertex in self.facilities:
            self.number[vertex, "in"] = len(self.number)
            self.number[vertex, "out"] = len(self.number)
        self.size = len(self.number)
        self.vertex_arcs = tuple(
            (self.number[v, "in"], self.number[v, "out"], v)
            for v in self.facilities
        )
        self.edge_arcs: list[tuple[int, int, int]] = []
        self.edges = []
        for u, v, index in edges:
            uout = 0 if u == ROOT else self.number[u, "out"]
            vout = 0 if v == ROOT else self.number[v, "out"]
            uin = 0 if u == ROOT else self.number[u, "in"]
            vin = 0 if v == ROOT else self.number[v, "in"]
            self.edge_arcs.append((uout, vin, index))
            self.edge_arcs.append((vout, uin, index))
            self.edges.append((u, v, index))
        total_arcs = len(self.vertex_arcs) + len(self.edge_arcs)
        # Scratch buffers for the flat-array Dinic (A3), reused across calls.
        self.buffer_head = [-1] * self.size
        self.buffer_to = [0] * (2 * total_arcs)
        self.buffer_nxt = [0] * (2 * total_arcs)
        self.buffer_capacity = [0.0] * (2 * total_arcs)
        self.level = [-1] * self.size
        self.level_stamp = [0] * self.size
        self.reachable_stamp = [0] * self.size
        self.stamp = 0
        self.calls = 0
        self.secondary_side: set[int] | None = None
        # Residual state of the most recent flow, reused by the nested mode.
        self.last_residual = None
        self.last_source_side: set[int] | None = None
        self.secondary_rows_added = 0
        self.nested_sides_tried = 0
        self.nested_rows_added = 0
        # numpy/scipy are imported at module level; only the scipy backend
        # materialises the scaled split graph here.
        self.numpy = None
        self.sparse = None
        self.csgraph = None
        if backend == "scipy":
            self.numpy = np
            self.sparse = sparse
            self.csgraph = csgraph_module
            self._prepare_scaled_graph()

    def _prepare_scaled_graph(self) -> None:
        """Materialise the full split-graph CSR once (A2); only data changes."""
        np = self.numpy
        arc_from = [u for u, _, _ in self.vertex_arcs]
        arc_to = [v for _, v, _ in self.vertex_arcs]
        base = [self.SCALE] * len(self.vertex_arcs)
        for u, v, _ in self.edge_arcs:
            arc_from.append(u)
            arc_to.append(v)
            base.append(0)
        matrix = self.sparse.csr_matrix(
            (
                np.asarray(base, dtype=np.int64),
                (
                    np.asarray(arc_from, dtype=np.int64),
                    np.asarray(arc_to, dtype=np.int64),
                ),
            ),
            shape=(self.size, self.size),
        ).tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        indptr = matrix.indptr
        indices = matrix.indices
        position: dict[tuple[int, int], int] = {}
        for row in range(self.size):
            for slot in range(int(indptr[row]), int(indptr[row + 1])):
                position[(row, int(indices[slot]))] = slot
        self.scaled_indptr = indptr
        self.scaled_indices = indices
        self.scaled_base = matrix.data.astype(np.int64)
        self.scaled_position = position
        self.scaled_tolerance = max(1, int(math.ceil(TOL * self.SCALE)))

    def minimum_cut(
        self,
        values: list[float],
        target: str,
        threshold: float,
    ) -> tuple[float, dict[int, float] | None, str | None]:
        """Certify threshold or return a violated UCUT row and deleted node."""
        self.calls += 1
        self.secondary_side = None
        self.last_residual = None
        self.last_source_side = None
        if self.backend == "scipy":
            flow_value, source_side = self._flow_scaled(values, target, threshold)
        else:
            flow_value, source_side = self._flow_dinic(values, target, threshold)
        if source_side is None:
            return flow_value, None, None
        value, row, removed = self._violated_row(
            source_side, values, target, threshold, flow_value)
        if row is None and self.backend == "scipy":
            # Integer capacities are rounded DOWN. A passing scaled flow is
            # a safe certificate, but a sub-threshold scaled cut that is not
            # violated in floating point can conceal a different violated
            # cut. Recheck this ambiguous terminal with unscaled Dinic.
            self.calls += 1
            exact_value, exact_side = self._flow_dinic(values, target, threshold)
            if exact_side is None:
                return exact_value, None, None
            return self._violated_row(
                exact_side, values, target, threshold, exact_value)
        return value, row, removed

    def minimum_cut_rows(
        self, values: list[float], target: str, threshold: float
    ) -> tuple[float, list[tuple[dict[int, float], str | None]]]:
        """Obtain several minimum cuts from one max-flow.

        The smallest and the largest minimum cut are both valid UCUT cuts when
        the flow value is below the threshold, and they often give distinct
        violated rows at almost no additional separation cost. In nested mode
        intermediate minimum cuts of the residual lattice are added on top.
        """
        flow_value, first, removed = self.minimum_cut(values, target, threshold)
        if first is None:
            return flow_value, []
        rows = [(first, removed)]
        if self.cut_mode == "single":
            return flow_value, rows
        if self.secondary_side is not None:
            _, other, other_removed = self._violated_row(
                set(self.secondary_side), values, target, threshold, flow_value
            )
            if other is not None and (other, other_removed) != (first, removed):
                rows.append((other, other_removed))
                self.secondary_rows_added += 1
        for side in self._nested_sides():
            _, extra, extra_removed = self._violated_row(
                side, values, target, threshold, flow_value)
            if extra is None:
                continue
            if any(
                extra == known and extra_removed == known_removed
                for known, known_removed in rows
            ):
                continue
            rows.append((extra, extra_removed))
            self.nested_rows_added += 1
        return flow_value, rows

    def _nested_sides(self) -> list[set[int]]:
        """Intermediate minimum source sides of the most recent max-flow.

        For a maximum flow ``f``, a set ``S`` with the source in ``S`` and the
        sink outside is a minimum cut exactly when no residual arc leaves
        ``S``, that is, when ``S`` is closed under residual reachability. The
        two extremes are the source side of the smallest cut and the
        complement of the nodes that still reach the sink; every set in
        between is ``source_side`` unioned with the residual reachability of
        nodes that reach neither extreme. Because the flow value is already
        below the threshold, each of those sets yields a violated UCUT row
        without any additional max-flow call.
        """
        if self.cut_mode != "nested" or self.numpy is None:
            return []
        residual = self.last_residual
        smallest = self.last_source_side
        largest = self.secondary_side
        if residual is None or smallest is None or largest is None:
            return []
        if self.nested_levels <= 1:
            return []
        middle = sorted(set(largest) - set(smallest))
        if not middle:
            return []
        step = max(1, len(middle) // max(1, self.nested_levels - 1))
        candidates = middle[::step][: self.nested_levels - 1]
        sides: list[set[int]] = []
        accumulated = set(smallest)
        for node in candidates:
            self.nested_sides_tried += 1
            order = self.csgraph.breadth_first_order(
                residual, node, directed=True, return_predecessors=False)
            reach = {int(vertex) for vertex in order}
            if reach <= accumulated:
                continue
            accumulated |= reach
            sides.append(set(accumulated))
        return sides

    def _flow_dinic(
        self, values: list[float], target: str, threshold: float
    ) -> tuple[float, set[int] | None]:
        """Dinic on the positive-support split graph (A1 + A3)."""
        size = self.size
        # The exact fallback must retain arcs below the LP feasibility
        # tolerance: many tiny capacities can add up across a cut.
        flow_eps = 1e-12
        head = self.buffer_head
        to = self.buffer_to
        nxt = self.buffer_nxt
        capacity = self.buffer_capacity
        for index in range(size):
            head[index] = -1
        count = 0
        for u, v, facility in self.vertex_arcs:
            arc = 2 * count
            twin = arc + 1
            to[arc] = v
            capacity[arc] = threshold if facility == target else 1.0
            nxt[arc] = head[u]
            head[u] = arc
            to[twin] = u
            capacity[twin] = 0.0
            nxt[twin] = head[v]
            head[v] = twin
            count += 1
        for u, v, index in self.edge_arcs:
            value = values[index]
            if value <= flow_eps:
                continue
            arc = 2 * count
            twin = arc + 1
            to[arc] = v
            capacity[arc] = value
            nxt[arc] = head[u]
            head[u] = arc
            to[twin] = u
            capacity[twin] = 0.0
            nxt[twin] = head[v]
            head[v] = twin
            count += 1

        sink = self.number[target, "out"]
        level = self.level
        level_stamp = self.level_stamp
        queue = [0] * size
        cursor = [-1] * size
        node_stack = [0] * (size + 1)
        arc_stack = [0] * (size + 1)
        stamp = self.stamp
        flow = 0.0
        while flow < threshold - TOL:
            # Level graph; a fresh stamp replaces the O(V) level wipe.
            stamp += 1
            level[0] = 0
            level_stamp[0] = stamp
            queue[0] = 0
            tail = 1
            position = 0
            while position < tail:
                u = queue[position]
                position += 1
                next_level = level[u] + 1
                arc = head[u]
                while arc != -1:
                    v = to[arc]
                    if capacity[arc] > flow_eps and level_stamp[v] != stamp:
                        level_stamp[v] = stamp
                        level[v] = next_level
                        queue[tail] = v
                        tail += 1
                    arc = nxt[arc]
            if level_stamp[sink] != stamp:
                break
            for node in range(size):
                cursor[node] = head[node]
            # Blocking flow by an explicit stack (no recursion).
            while flow < threshold - TOL:
                depth = 0
                node_stack[0] = 0
                while node_stack[depth] != sink:
                    u = node_stack[depth]
                    arc = cursor[u]
                    while arc != -1:
                        v = to[arc]
                        if (
                            capacity[arc] > flow_eps
                            and level_stamp[v] == stamp
                            and level[v] == level[u] + 1
                        ):
                            break
                        arc = nxt[arc]
                    cursor[u] = arc
                    if arc == -1:
                        if depth == 0:
                            break
                        depth -= 1
                        cursor[node_stack[depth]] = nxt[cursor[node_stack[depth]]]
                        continue
                    arc_stack[depth] = arc
                    depth += 1
                    node_stack[depth] = to[arc]
                if node_stack[depth] != sink:
                    break
                bottleneck = threshold - flow
                for step in range(depth):
                    residual = capacity[arc_stack[step]]
                    if residual < bottleneck:
                        bottleneck = residual
                for step in range(depth):
                    arc = arc_stack[step]
                    capacity[arc] -= bottleneck
                    capacity[arc ^ 1] += bottleneck
                flow += bottleneck
        self.stamp = stamp
        if flow >= threshold - TOL:
            return flow, None
        # Residual reachability from the root defines the source side.
        stamp += 1
        self.stamp = stamp
        reachable_stamp = self.reachable_stamp
        queue[0] = 0
        reachable_stamp[0] = stamp
        tail = 1
        position = 0
        while position < tail:
            u = queue[position]
            position += 1
            arc = head[u]
            while arc != -1:
                v = to[arc]
                if capacity[arc] > flow_eps and reachable_stamp[v] != stamp:
                    reachable_stamp[v] = stamp
                    queue[tail] = v
                    tail += 1
                arc = nxt[arc]
        source_side = set(queue[:tail])
        reverse_seen = {sink}
        queue[0] = sink
        tail = 1
        position = 0
        while position < tail:
            v = queue[position]
            position += 1
            arc = head[v]
            while arc != -1:
                u = to[arc]
                if capacity[arc ^ 1] > flow_eps and u not in reverse_seen:
                    reverse_seen.add(u)
                    queue[tail] = u
                    tail += 1
                arc = nxt[arc]
        self.secondary_side = set(range(size)) - reverse_seen
        return flow, source_side

    def _flow_scaled(
        self, values: list[float], target: str, threshold: float
    ) -> tuple[float, set[int] | None]:
        """C max-flow on the integer-scaled split graph (A2), with A1 and A3."""
        scale = self.SCALE
        data = self.scaled_base.copy()
        inside = self.number[target, "in"]
        sink = self.number[target, "out"]
        position = self.scaled_position
        scaled_threshold = int(round(threshold * scale))
        data[position[(inside, sink)]] = scaled_threshold
        for u, v, index in self.edge_arcs:
            value = values[index]
            if value > TOL:
                data[position[(u, v)]] = int(math.floor(value * scale))
        graph = self.sparse.csr_matrix(
            (data, self.scaled_indices, self.scaled_indptr),
            shape=(self.size, self.size),
        )
        result = self.csgraph.maximum_flow(graph, 0, sink)
        flow_scaled = int(result.flow_value)
        if flow_scaled >= scaled_threshold - self.scaled_tolerance:
            return flow_scaled / scale, None
        residual = graph - result.flow
        residual.sum_duplicates()
        residual.sort_indices()
        residual.data[residual.data <= 0] = 0
        residual.eliminate_zeros()
        order = self.csgraph.breadth_first_order(
            residual, 0, directed=True, return_predecessors=False)
        reversed_order = self.csgraph.breadth_first_order(
            residual.transpose().tocsr(), sink,
            directed=True, return_predecessors=False)
        self.last_residual = residual
        self.last_source_side = {int(node) for node in order}
        self.secondary_side = set(range(self.size)) - {
            int(node) for node in reversed_order
        }
        return flow_scaled / scale, set(self.last_source_side)

    def _violated_row(
        self,
        reachable: set[int],
        values: list[float],
        target: str,
        threshold: float,
        flow_value: float,
    ) -> tuple[float, dict[int, float], str | None]:
        """Turn the source side of a sub-threshold cut into a UCUT row."""
        # A node with in on the sink side and out on the source side can be
        # moved entirely to the source side without increasing the cut
        # capacity. This gives a canonical A/Z/B partition.
        for facility in self.facilities:
            inside = self.number[facility, "in"]
            outside = self.number[facility, "out"]
            if outside in reachable and inside not in reachable:
                reachable.add(inside)
        deleted = [
            facility for facility in self.facilities
            if self.number[facility, "in"] in reachable
            and self.number[facility, "out"] not in reachable
        ]
        if len(deleted) > 1 or target in deleted:
            raise RuntimeError("Split cut below threshold has invalid boundary")
        removed = deleted[0] if deleted else None
        source_side = {ROOT} | {
            facility for facility in self.facilities
            if facility != removed
            and self.number[facility, "out"] in reachable
        }
        row = {
            index: 1.0 for u, v, index in self.edges
            if u != removed and v != removed
            and ((u in source_side) != (v in source_side))
        }
        rhs = threshold - len(deleted)
        row_capacity = sum(values[index] for index in row)
        if row_capacity >= rhs - TOL:
            # A rounded or tolerance-truncated cut may cease to be violated
            # in original units. The caller must use an exact fallback;
            # it cannot certify the terminal from this candidate alone.
            return flow_value, None, None
        return flow_value, row, removed

def _minimum_cut_row(
    nodes: Iterable[str],
    edges: Iterable[tuple[str, str, int, float]],
    source: str,
    sink: str,
    threshold: float,
    removed: str | None = None,
) -> tuple[float, dict[int, float] | None]:
    """Use the stable NetworkX oracle for independent LP/IP benchmarks."""
    graph = nx.Graph()
    graph.add_nodes_from(node for node in nodes if node != removed)
    indexed_edges = []
    for u, v, index, value in edges:
        if u == removed or v == removed:
            continue
        graph.add_edge(u, v, capacity=max(0.0, value))
        indexed_edges.append((u, v, index))
    if source not in graph or sink not in graph:
        return 0.0, {}
    cut_value, partition = nx.minimum_cut(
        graph, source, sink, capacity="capacity",
        flow_func=nx.algorithms.flow.preflow_push,
    )
    if cut_value >= threshold - TOL:
        return cut_value, None
    source_side, sink_side = partition
    coefficients = {
        index: 1.0
        for u, v, index in indexed_edges
        if (u in source_side and v in sink_side)
        or (v in source_side and u in sink_side)
    }
    return cut_value, coefficients


def _design_cut_rows(
    instance: TwoRadiusInstance,
    edge_indices: dict[tuple[str, str], int],
    service_indices: dict[tuple[str, str], int],
    values: list[float],
) -> tuple[list[tuple[dict[int, float], str, float]], float]:
    ordinary_rows = []
    minimum_value = math.inf
    nodes = (ROOT,) + instance.facilities + tuple(instance.demand_positions)
    for demand, requirement in instance.requirements.items():
        edges = [
            (u, v, index, values[index])
            for (u, v), index in edge_indices.items()
        ] + [
            (facility, demand, service_indices[demand, facility],
             values[service_indices[demand, facility]])
            for facility in instance.coverage[demand]
            if (demand, facility) in service_indices
        ]
        cut_value, coefficients = _minimum_cut_row(
            nodes, edges, ROOT, demand, requirement)
        minimum_value = min(minimum_value, cut_value)
        if coefficients is not None:
            ordinary_rows.append((coefficients, "G", float(requirement)))
    if ordinary_rows:
        return ordinary_rows, minimum_value

    deleted_vertex_rows = []
    for demand, requirement in instance.requirements.items():
        if requirement == 2:
            edges = [
                (u, v, index, values[index])
                for (u, v), index in edge_indices.items()
            ] + [
                (facility, demand, service_indices[demand, facility],
                 values[service_indices[demand, facility]])
                for facility in instance.coverage[demand]
                if (demand, facility) in service_indices
            ]
            incident_load = {
                facility: sum(
                    value for u, v, _, value in edges
                    if facility in (u, v)
                )
                for facility in instance.facilities
            }
            # Only facilities carrying positive capacity can reduce the cut
            # below 1 when deleted (ordinary cut is already >= 2).
            active_candidates = [
                facility
                for facility in instance.facilities
                if incident_load[facility] > TOL
            ]
            removed_order = sorted(
                active_candidates,
                key=lambda facility: (
                    incident_load[facility] <= 1.0 + TOL,
                    -incident_load[facility],
                    facility,
                ),
            )
            for removed in removed_order:
                cut_value, coefficients = _minimum_cut_row(
                    nodes, edges, ROOT, demand, 1.0, removed)
                minimum_value = min(minimum_value, cut_value)
                if coefficients is not None:
                    deleted_vertex_rows.append((coefficients, "G", 1.0))
    return deleted_vertex_rows, minimum_value


def verify_design(instance: TwoRadiusInstance, opened: set[str]) -> bool:
    """Check demand connectivity directly on the opened undirected graph.

    Attaching one demand at a time keeps other demands out of backbone paths.
    A demand has two internally vertex-disjoint root paths precisely when it
    shares a biconnected block with the root (Menger's theorem).
    """
    opened = set(opened)
    if instance.variant == "preinstalled":
        opened |= set(instance.depots)
    if not opened <= set(instance.facilities):
        return False
    graph = nx.Graph()
    graph.add_nodes_from((ROOT,) + tuple(opened))
    graph.add_edges_from(
        (u, v) for u, v in instance.backbone_edges
        if (u == ROOT and v in opened) or (u in opened and v in opened)
    )
    reachable = nx.node_connected_component(graph, ROOT)
    for demand, requirement in instance.requirements.items():
        neighbors = set(instance.coverage[demand]) & opened
        if len(neighbors) < requirement:
            return False
        if requirement == 1:
            if not neighbors & reachable:
                return False
            continue
        if len(neighbors & reachable) < 2:
            return False
        graph.add_node(demand)
        graph.add_edges_from((demand, facility) for facility in neighbors)
        valid = any(
            ROOT in block and demand in block and len(block) >= 3
            for block in nx.biconnected_components(graph)
        )
        graph.remove_node(demand)
        if not valid:
            return False
    return True


def solve_cover_pass_mip(
    instance: TwoRadiusInstance,
    demands: list[str],
    candidates: list[str],
    deadline: float,
    relative_gap: float,
) -> dict:
    start = time.perf_counter()
    if not demands:
        return {"status": "empty", "selected": [], "objective": 0.0, "seconds": 0.0}
    model = _new_model(_remaining(deadline), mip=True)
    model.parameters.mip.tolerances.mipgap.set(relative_gap)
    indices = {facility: index for index, facility in enumerate(candidates)}
    model.variables.add(
        names=[f"cover_{facility}" for facility in candidates],
        obj=[instance.costs[facility] for facility in candidates],
        lb=[0.0] * len(candidates),
        ub=[1.0] * len(candidates),
        types="B" * len(candidates),
    )
    rows = []
    for demand in demands:
        coefficients = {
            indices[facility]: 1.0
            for facility in instance.coverage[demand]
            if facility in indices
        }
        if not coefficients:
            raise ValueError(f"Cover pass cannot cover demand {demand}")
        rows.append((coefficients, "G", 1.0))
    _add_rows(model, rows)
    model.parameters.timelimit.set(_remaining(deadline))
    model.solve()
    status = model.solution.get_status_string()
    if not model.solution.is_primal_feasible():
        raise RuntimeError(f"Cover MIP has no incumbent: {status}")
    values = model.solution.get_values()
    selected = [
        facility for facility, index in indices.items() if values[index] > 0.5
    ]
    return {
        "status": status,
        "selected": selected,
        "objective": model.solution.get_objective_value(),
        "best_bound": model.solution.MIP.get_best_objective(),
        "mip_relative_gap": model.solution.MIP.get_mip_relative_gap(),
        "variables": model.variables.get_num(),
        "constraints": model.linear_constraints.get_num(),
        "seconds": time.perf_counter() - start,
    }


def solve_cover_pass_greedy(
    instance: TwoRadiusInstance,
    demands: list[str],
    candidates: list[str],
    deadline: float,
) -> dict:
    """Run deterministic weighted greedy set cover for one coverage pass."""
    start = time.perf_counter()
    pending = set(demands)
    available = set(candidates)
    selected: list[str] = []
    candidate_evaluations = 0
    while pending:
        _remaining(deadline)
        best_facility = None
        best_key = None
        for facility in sorted(available):
            gain = sum(
                facility in instance.coverage[demand] for demand in pending)
            candidate_evaluations += 1
            if gain == 0:
                continue
            cost = instance.costs[facility]
            score = math.inf if cost == 0 else gain / cost
            key = (score, gain, -cost, facility)
            if best_key is None or key > best_key:
                best_key = key
                best_facility = facility
        if best_facility is None:
            missing = sorted(pending)
            raise ValueError(
                f"Greedy cover cannot cover demand(s): {', '.join(missing)}")
        selected.append(best_facility)
        available.remove(best_facility)
        pending = {
            demand for demand in pending
            if best_facility not in instance.coverage[demand]
        }
    return {
        "status": "greedy feasible",
        "selected": selected,
        "objective": sum(instance.costs[facility] for facility in selected),
        "candidate_evaluations": candidate_evaluations,
        "seconds": time.perf_counter() - start,
    }


def stage_one_cover(
    instance: TwoRadiusInstance,
    epsilon: float,
    deadline: float,
    cover_method: str = "greedy",
) -> dict:
    if cover_method not in ("greedy", "mip"):
        raise ValueError("Cover method must be greedy or mip")
    start = time.perf_counter()
    coverage = instance.coverage
    cover = {
        facility
        for facility in instance.facilities
        if instance.costs[facility] == 0
        and any(facility in coverage[demand] for demand in coverage)
    }
    passes = []
    for pass_index in (1, 2):
        demands = [
            demand
            for demand, requirement in instance.requirements.items()
            if requirement >= pass_index
            and len(set(coverage[demand]) & cover) == pass_index - 1
        ]
        candidates = [
            facility for facility in instance.facilities if facility not in cover
        ]
        if cover_method == "greedy":
            result = solve_cover_pass_greedy(
                instance, demands, candidates, deadline)
        else:
            result = solve_cover_pass_mip(
                instance, demands, candidates, deadline, epsilon / 2)
        cover.update(result["selected"])
        result["pass"] = pass_index
        result["demands"] = demands
        passes.append(result)
    if any(
        len(set(coverage[demand]) & cover) < requirement
        for demand, requirement in instance.requirements.items()
    ):
        raise RuntimeError("Stage 1 returned an invalid service cover")
    terminals_two = {
        facility
        for facility in cover
        if any(
            instance.requirements[demand] == 2
            and facility in coverage[demand]
            for demand in coverage
        )
    }
    return {
        "status": "feasible",
        "cover": sorted(cover),
        "T1": sorted(cover - terminals_two),
        "T2": sorted(terminals_two),
        "cost": sum(instance.costs[v] for v in cover),
        "cover_method": cover_method,
        "passes": passes,
        "seconds": time.perf_counter() - start,
    }


class UcutWorkspace:
    """Retained CPLEX master for residual UCUT LPs."""

    def __init__(
        self,
        instance: TwoRadiusInstance,
        cover: set[str],
        terminals_one: set[str],
        terminals_two: set[str],
        separator_backend: str = "scipy",
        cut_mode: str = "double",
        nested_levels: int = 4,
        cut_cap_per_terminal: int = 0,
        cut_cap_per_round: int = 0,
    ) -> None:
        self.instance = instance
        self.cover = cover
        self.terminals_one = terminals_one
        self.terminals_two = terminals_two
        self.edges = instance.backbone_edges
        self.edge_index = {edge: index for index, edge in enumerate(self.edges)}
        self.separator_backend = separator_backend
        self.cut_mode = cut_mode
        self.nested_levels = nested_levels
        self.cut_cap_per_terminal = cut_cap_per_terminal
        self.cut_cap_per_round = cut_cap_per_round
        self.cut_oracle = _SplitCutOracle(
            (ROOT,) + instance.facilities,
            ((u, v, index) for (u, v), index in self.edge_index.items()),
            backend=separator_backend,
            cut_mode=cut_mode,
            nested_levels=nested_levels,
        )
        reduced = {
            facility: 0 if facility in cover else instance.costs[facility]
            for facility in instance.facilities
        }
        reduced[ROOT] = 0
        self.edge_cost = {
            edge: (reduced[edge[0]] + reduced[edge[1]]) / 2
            for edge in self.edges
        }
        self.model = _new_model(1.0, strict_feasibility=True)
        self.model.variables.add(
            names=[f"x_{index}" for index in range(len(self.edges))],
            obj=[self.edge_cost[edge] for edge in self.edges],
            lb=[0.0] * len(self.edges),
            ub=[1.0] * len(self.edges),
        )
        self.fixed: set[tuple[str, str]] = set()
        self.cut_signatures: set[tuple] = set()
        self.trace: list[dict] = []
        self.total_cuts = 0
        self.last_separation_phase = "ordinary"
        self.last_removed_phase = None
        self.integer_test_calls = 0
        self.integer_test_seconds = 0.0
        singleton_rows = []
        for terminal in terminals_one | terminals_two:
            requirement = 2.0 if terminal in terminals_two else 1.0
            coefficients = {
                index: 1.0
                for edge, index in self.edge_index.items()
                if terminal in edge
            }
            singleton_rows.append((coefficients, "G", requirement))
        self._add_unique_rows(singleton_rows)

    @staticmethod
    def _signature(row: tuple[dict[int, float], str, float]) -> tuple:
        coefficients, sense, rhs = row
        return tuple(sorted(coefficients.items())), sense, rhs

    def _add_unique_rows(
        self, rows: list[tuple[dict[int, float], str, float]]
    ) -> int:
        unique = []
        for row in rows:
            signature = self._signature(row)
            if row[0] and signature not in self.cut_signatures:
                self.cut_signatures.add(signature)
                unique.append(row)
        _add_rows(self.model, unique)
        self.total_cuts += len(unique)
        return len(unique)

    def separate(
        self,
        values: list[float],
    ) -> tuple[list[tuple[dict[int, float], str, float, float]], float]:
        """One complete split-graph threshold min-cut per terminal.

        Each candidate row carries its violation ``rhs - sum(x_e)`` in the
        current LP solution so that a capped selection can prefer strongly
        violated and short rows.
        """
        rows = []
        minimum_value = math.inf
        self.last_separation_phase = "split"
        self.last_removed_phase = None
        for terminal in sorted(self.terminals_one | self.terminals_two):
            requirement = 2.0 if terminal in self.terminals_two else 1.0
            cut_value, cut_rows = self.cut_oracle.minimum_cut_rows(
                values, terminal, requirement)
            minimum_value = min(minimum_value, cut_value)
            candidates = []
            for coefficients, removed in cut_rows:
                rhs = requirement - (removed is not None)
                violation = rhs - sum(
                    values[index] for index in coefficients)
                candidates.append((coefficients, "G", float(rhs), violation))
            if 0 < self.cut_cap_per_terminal < len(candidates):
                candidates.sort(key=lambda item: (-item[3], len(item[0])))
                candidates = candidates[: self.cut_cap_per_terminal]
            rows.extend(candidates)
        return rows, minimum_value

    def is_feasible(self, selected: set[tuple[str, str]]) -> bool:
        start = time.perf_counter()
        self.integer_test_calls += 1
        graph = nx.Graph()
        graph.add_nodes_from((ROOT,) + self.instance.facilities)
        graph.add_edges_from(selected)
        connected = (
            nx.node_connected_component(graph, ROOT)
            if ROOT in graph else set()
        )
        feasible = self.terminals_one <= connected
        if feasible and self.terminals_two:
            protected: set[str] = set()
            for block in nx.biconnected_components(graph):
                if ROOT in block and len(block) >= 3:
                    protected.update(block)
            feasible = self.terminals_two <= protected
        self.integer_test_seconds += time.perf_counter() - start
        return feasible

    def solve(self, fixed: set[tuple[str, str]], deadline: float) -> dict:
        for edge in fixed - self.fixed:
            index = self.edge_index[edge]
            self.model.variables.set_lower_bounds(index, 1.0)
            self.model.objective.set_linear(index, 0.0)
        if not self.fixed <= fixed:
            raise ValueError("Rounded edges cannot be unfixed")
        self.fixed = set(fixed)
        start = time.perf_counter()
        solver_seconds = 0.0
        separation_seconds = 0.0
        cuts_added = 0
        solve_flow_calls = 0
        while True:
            self.model.parameters.timelimit.set(_remaining(deadline))
            solve_start = time.perf_counter()
            self.model.solve()
            solver_seconds += time.perf_counter() - solve_start
            status = self.model.solution.get_status_string()
            if self.model.solution.get_status() != self.model.solution.status.optimal:
                raise RuntimeError(f"UCUT master is not optimal: {status}")
            values = self.model.solution.get_values()
            objective = self.model.solution.get_objective_value()
            solution_type = self.model.solution.get_solution_type()
            separation_start = time.perf_counter()
            flow_calls_before = self.cut_oracle.calls
            rows, minimum_value = self.separate(values)
            separation_seconds += time.perf_counter() - separation_start
            flow_calls = self.cut_oracle.calls - flow_calls_before
            solve_flow_calls += flow_calls
            if self.cut_cap_per_round > 0:
                # Only the admission of rows is capped. Every round still runs
                # the complete separation below, so the loop stops exactly when
                # an unlimited separation finds no fresh violated row.
                fresh = [
                    row for row in rows
                    if self._signature((row[0], row[1], row[2]))
                    not in self.cut_signatures
                ]
                fresh.sort(key=lambda item: (-item[3], len(item[0])))
                selected = fresh[: self.cut_cap_per_round]
            else:
                selected = sorted(rows, key=lambda item: len(item[0]))
            new_cuts = self._add_unique_rows(
                [(row[0], row[1], row[2]) for row in selected])
            cuts_added += new_cuts
            self.trace.append({
                "status": status,
                "objective": objective,
                "solver_seconds": solver_seconds,
                "separation_seconds": separation_seconds,
                "new_cuts": new_cuts,
                "total_cuts": self.total_cuts,
                "separation_phase": self.last_separation_phase,
                "removed_phase": self.last_removed_phase,
                "max_flow_calls": flow_calls,
                "minimum_cut_value": minimum_value,
                "variables": self.model.variables.get_num(),
                "constraints": self.model.linear_constraints.get_num(),
                "max_unfixed_x": max(
                    (values[index] for edge, index in self.edge_index.items()
                     if edge not in fixed),
                    default=0.0,
                ),
            })
            if rows and new_cuts == 0:
                raise RuntimeError(
                    "UCUT separator found only previously added violated "
                    "rows; check LP/separation feasibility tolerances"
                )
            if new_cuts == 0:
                if solution_type != self.model.solution.type.basic:
                    raise RuntimeError("UCUT solution is not a basic solution")
                return {
                    "status": status,
                    "objective": objective,
                    "x": dict(zip(self.edges, values)),
                    "solver_seconds": solver_seconds,
                    "separation_seconds": separation_seconds,
                    "max_flow_calls": solve_flow_calls,
                    "cuts_added": cuts_added,
                    "optimal_basic": True,
                    "seconds": time.perf_counter() - start,
                }


def iterative_round_stage_two(
    instance: TwoRadiusInstance,
    stage_one: dict,
    deadline: float,
    separator_backend: str = "scipy",
    cut_mode: str = "double",
    nested_levels: int = 4,
    cut_cap_per_terminal: int = 0,
    cut_cap_per_round: int = 0,
) -> dict:
    start = time.perf_counter()
    cover = set(stage_one["cover"])
    terminals_one = set(stage_one["T1"])
    terminals_two = set(stage_one["T2"])
    workspace = UcutWorkspace(
        instance,
        cover,
        terminals_one,
        terminals_two,
        separator_backend,
        cut_mode=cut_mode,
        nested_levels=nested_levels,
        cut_cap_per_terminal=cut_cap_per_terminal,
        cut_cap_per_round=cut_cap_per_round,
    )
    full_test_before = workspace.integer_test_seconds
    if not workspace.is_feasible(set(instance.backbone_edges)):
        return {"status": "infeasible", "seconds": time.perf_counter() - start}
    full_test_seconds = workspace.integer_test_seconds - full_test_before
    fixed: set[tuple[str, str]] = set()
    rounds = []
    initial_lp = None
    initial_positive_support_edges = None
    rounding_test_seconds = 0.0
    while not workspace.is_feasible(fixed):
        rounding_test_seconds = (
            workspace.integer_test_seconds - full_test_seconds)
        result = workspace.solve(fixed, deadline)
        if initial_lp is None:
            initial_lp = result["objective"]
            initial_positive_support_edges = sum(
                value > TOL for value in result["x"].values()
            )
        eligible = [
            edge
            for edge in workspace.edges
            if edge not in fixed and result["x"][edge] >= 0.5 - 1e-9
        ]
        if not eligible:
            raise RuntimeError("Optimal basic UCUT solution has no x >= 1/2")
        fixed.update(eligible)
        rounds.append({
            "round": len(rounds) + 1,
            "residual_objective": result["objective"],
            "rounded_edges": [list(edge) for edge in eligible],
            "num_rounded_edges": len(eligible),
            "fixed_edges_after": len(fixed),
            "solver_seconds": result["solver_seconds"],
            "separation_seconds": result["separation_seconds"],
            "max_flow_calls": result["max_flow_calls"],
            "cuts_added": result["cuts_added"],
            "optimal_basic": result["optimal_basic"],
        })
        _remaining(deadline)

    before_pruning = len(fixed)
    rounding_test_seconds = workspace.integer_test_seconds - full_test_seconds
    pruning_start = time.perf_counter()
    pruning_test_calls_before = workspace.integer_test_calls
    pruning_test_seconds_before = workspace.integer_test_seconds
    for edge in sorted(fixed):
        _remaining(deadline)
        candidate = fixed - {edge}
        if workspace.is_feasible(candidate):
            fixed = candidate
    pruning_seconds = time.perf_counter() - pruning_start
    pruning_test_calls = (
        workspace.integer_test_calls - pruning_test_calls_before)
    pruning_test_seconds = (
        workspace.integer_test_seconds - pruning_test_seconds_before)
    endpoints = {
        vertex for edge in fixed for vertex in edge if vertex != ROOT
    }
    opened = cover | endpoints
    if instance.variant == "preinstalled":
        opened |= set(instance.depots)
    verification_start = time.perf_counter()
    verified = verify_design(instance, set(opened))
    verification_seconds = time.perf_counter() - verification_start
    if not verified:
        raise RuntimeError("Rounded design failed independent demand verification")
    stage_two_added = opened - cover
    stage_two_cost = sum(instance.costs[v] for v in stage_two_added)
    return {
        "status": "feasible",
        "selected": sorted(opened),
        "cost": sum(instance.costs[v] for v in opened),
        "stage_two_added_cost": stage_two_cost,
        "separator_backend": workspace.separator_backend,
        "cut_mode": workspace.cut_mode,
        "nested_levels": workspace.nested_levels,
        "cut_cap_per_terminal": workspace.cut_cap_per_terminal,
        "cut_cap_per_round": workspace.cut_cap_per_round,
        "secondary_rows_added": workspace.cut_oracle.secondary_rows_added,
        "nested_rows_added": workspace.cut_oracle.nested_rows_added,
        "nested_sides_tried": workspace.cut_oracle.nested_sides_tried,
        "stage_two_over_edge_lp": (
            stage_two_cost / initial_lp
            if initial_lp is not None and initial_lp > TOL else 0.0
        ),
        "support_edges": [list(edge) for edge in sorted(fixed)],
        "initial_edge_lp": initial_lp,
        "initial_positive_support_edges": initial_positive_support_edges,
        "rounds": rounds,
        "num_rounds": len(rounds),
        "master_lp_solves": len(workspace.trace),
        "edges_before_pruning": before_pruning,
        "edges_after_pruning": len(fixed),
        "edges_pruned": before_pruning - len(fixed),
        "cuts_retained": workspace.total_cuts,
        "master_trace": workspace.trace,
        "lp_solver_seconds": sum(
            round_data["solver_seconds"] for round_data in rounds),
        "lp_separation_seconds": sum(
            round_data["separation_seconds"] for round_data in rounds),
        "max_flow_calls": sum(
            round_data["max_flow_calls"] for round_data in rounds),
        "full_graph_test_seconds": full_test_seconds,
        "rounding_test_seconds": rounding_test_seconds,
        "pruning_seconds": pruning_seconds,
        "pruning_test_seconds": pruning_test_seconds,
        "verification_seconds": verification_seconds,
        "integer_test_calls": workspace.integer_test_calls,
        "pruning_test_calls": pruning_test_calls,
        "verified": True,
        "seconds": time.perf_counter() - start,
    }


_ROOT_NODE_LINE_PATTERN = re.compile(r"^\s*(?:[*!]\s*)?0(?!\+)\s+0\s+")
# Emitted by CPLEX once the root LP relaxation of a MIP has been solved.
_ROOT_LP_MARKER_PATTERN = re.compile(r"root relaxation", re.IGNORECASE)
# CPLEX prints the pre-cut root LP value on its own line, e.g.
# "Root relaxation: objective 1.234567e+03, 12345 iterations, 12.34 seconds".
# That line survives even when the time limit stops the solve before the node
# table is printed, which is exactly what happens on the largest tiers.
_ROOT_RELAXATION_VALUE_PATTERN = re.compile(
    r"root relaxation:\s*objective\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
    re.IGNORECASE)


class _CapturedSolverLog:
    """Capture the CPLEX log by redirecting the process stdout descriptors.

    On this CPLEX build ``Cplex.set_log_stream`` is tied to the results channel:
    the usual ``set_results_stream(None)`` used to silence every other model
    also silences the MIP log, so the log has to be captured outside the Python
    stream API.  CPLEX writes it to the process stdout handle, therefore the
    FLOW-IP model keeps its results stream live and this context manager redirects
    file descriptors 1 and 2 to a temporary file for the duration of the solve.
    A file keeps the captured text out of memory until the solve is finished.
    """

    def __init__(self) -> None:
        self.text = ""
        self._path: str | None = None
        self._stream = None
        self._saved: tuple[int, int] | None = None

    def __enter__(self) -> "_CapturedSolverLog":
        sys.stdout.flush()
        sys.stderr.flush()
        handle, path = tempfile.mkstemp(prefix="orl_cplex_log_", suffix=".txt")
        os.close(handle)
        self._path = path
        self._stream = open(path, "w+", encoding="utf-8", errors="replace")
        self._saved = (os.dup(1), os.dup(2))
        os.dup2(self._stream.fileno(), 1)
        os.dup2(self._stream.fileno(), 2)
        return self

    def __exit__(self, *_exc: object) -> bool:
        assert self._saved is not None
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._saved[0])
        os.close(self._saved[1])
        sys.stdout.flush()
        sys.stderr.flush()
        self._stream.close()
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as fh:
                self.text = fh.read()
        finally:
            if self._path is not None and os.path.exists(self._path):
                os.remove(self._path)
        return False


def _extract_first_root_relaxation(log_text: str) -> float | None:
    """Extract the first pre-cut root relaxation objective from a CPLEX node log.

    Incumbent-only lines use ``0+`` and are deliberately ignored.  Later root
    lines can reflect CPLEX cuts, so only the first node-0 line after the
    node-log header is admissible, and lines whose objective column is the word
    ``cutoff`` carry no relaxation value.  When the root LP relaxation is
    already integral CPLEX prints the literal word ``integral`` instead of a
    number, and the value then sits in the incumbent/bound columns of the same
    line.  The CPLEX infinity sentinels printed for an infeasible root LP are
    rejected.
    """
    in_node_log = False
    for line in log_text.splitlines():
        if "Node" in line and "Left" in line and "Objective" in line:
            in_node_log = True
            continue
        if not in_node_log:
            continue
        prefix = _ROOT_NODE_LINE_PATTERN.match(line)
        if prefix is None:
            continue
        tokens = line[prefix.end():].split()
        if not tokens:
            continue
        if tokens[0].startswith("integral"):
            # Columns: Objective='integral', IInf, Best Integer, Best Bound.
            # The Best Integer column holds the root relaxation value.
            candidates = tokens[2:4] if len(tokens) > 2 else []
            for candidate in candidates:
                try:
                    value = float(candidate)
                except ValueError:
                    continue
                return None if abs(value) >= ROOT_VALUE_SENTINEL else value
            return None
        try:
            value = float(tokens[0])
        except ValueError:
            # e.g. the 'cutoff' line, which has no objective value.
            continue
        return None if abs(value) >= ROOT_VALUE_SENTINEL else value
    # The node table is only printed once CPLEX has finished the root LP, so a
    # truncated log can carry the pre-cut root value on the dedicated
    # "Root relaxation: objective ..." line and nothing else.
    match = _ROOT_RELAXATION_VALUE_PATTERN.search(log_text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return None if abs(value) >= ROOT_VALUE_SENTINEL else value


def _root_lp_completed(log_text: str) -> bool:
    """Report whether CPLEX finished the root LP relaxation of the FLOW-IP.

    CPLEX prints a ``Root relaxation: objective ...`` / ``Root relaxation
    solution time = ...`` line as soon as the root LP relaxation has been
    solved, immediately before the node table.  When the time limit stops the
    solve during presolve or during the root LP itself, that line never appears,
    so its absence is used as evidence that no root relaxation value exists.
    """
    return bool(_ROOT_LP_MARKER_PATTERN.search(log_text))


def estimate_flow_ip_size(instance: TwoRadiusInstance) -> tuple[int, int]:
    """Estimate the compact FLOW-IP dimensions before building the model.

    Columns: one binary opening variable per facility plus one flow variable per
    directed arc of every demand commodity.  Rows: two supply rows per
    commodity plus one conservation row and at most one capacity row per
    facility.  The estimate is used as a memory guard so that an oversized
    compact model is never handed to CPLEX.
    """
    directed_backbone = len(instance.depots) + 2 * len(instance.physical_edges)
    columns = len(instance.facilities)
    rows = 0
    for coverage in instance.coverage.values():
        columns += directed_backbone + len(coverage)
        rows += 2 + 2 * len(instance.facilities)
    return columns, rows


def _build_flow_model(
    instance: TwoRadiusInstance,
    model: cplex.Cplex,
    facility_types: str | None = "B",
) -> dict[str, int]:
    """Add the compact node-capacitated flow formulation to ``model``.

    ``facility_types`` is the CPLEX variable type string of the opening
    variables: "B" builds FLOW-IP, while ``None`` builds its LP relaxation over
    the very same rows, columns and coefficients, so the relaxation objective
    is a valid lower bound on the FLOW-IP optimum.  ``None`` rather than "C" is
    required because handing any type string to ``variables.add`` turns the
    model into a MIP, which would make CPLEX run its MIP machinery (and report
    an integer status) instead of solving a plain LP.
    """
    facilities = instance.facilities
    y = {facility: index for index, facility in enumerate(facilities)}
    lower = [
        1.0 if instance.variant == "preinstalled" and v in instance.depots else 0.0
        for v in facilities
    ]
    if facility_types is None:
        model.variables.add(
            names=[f"y_{v}" for v in facilities],
            obj=[instance.costs[v] for v in facilities],
            lb=lower,
            ub=[1.0] * len(facilities),
        )
    else:
        model.variables.add(
            names=[f"y_{v}" for v in facilities],
            obj=[instance.costs[v] for v in facilities],
            lb=lower,
            ub=[1.0] * len(facilities),
            types=facility_types * len(facilities),
        )

    # Build directed arcs per demand p:
    # 1. Root-to-depot arcs (ROOT, d) for d in depots
    # 2. Bidirectional physical arcs (u, v) and (v, u) for {u, v} in physical_edges
    # 3. Service arcs (v, p) for v in coverage[p]
    directed_backbone: list[tuple[str, str]] = [
        (ROOT, depot) for depot in instance.depots
    ]
    for u, v in instance.physical_edges:
        directed_backbone.append((u, v))
        directed_backbone.append((v, u))

    rows: list[tuple[dict[int, float], str, float]] = []
    for demand, requirement in instance.requirements.items():
        demand_arcs = directed_backbone + [
            (facility, demand) for facility in instance.coverage[demand]
        ]
        base_idx = model.variables.get_num()
        model.variables.add(
            names=[f"f_{demand}_{u}_{v}" for u, v in demand_arcs],
            lb=[0.0] * len(demand_arcs),
            ub=[1.0] * len(demand_arcs),
        )
        inflow: dict[str, list[int]] = {v: [] for v in facilities}
        outflow: dict[str, list[int]] = {v: [] for v in facilities}
        root_out: list[int] = []
        demand_in: list[int] = []
        for offset, (u, v) in enumerate(demand_arcs):
            var_idx = base_idx + offset
            if u == ROOT:
                root_out.append(var_idx)
            elif u in outflow:
                outflow[u].append(var_idx)
            if v == demand:
                demand_in.append(var_idx)
            elif v in inflow:
                inflow[v].append(var_idx)

        # Supply at ROOT and demand at p: sum_{d in D} f^p_{rho,d} = k_p, sum_{v in N(p)} f^p_{v,p} = k_p
        rows.append(({idx: 1.0 for idx in root_out}, "E", float(requirement)))
        rows.append(({idx: 1.0 for idx in demand_in}, "E", float(requirement)))

        # Flow conservation and node capacity at each facility v in D u F:
        # sum_u f^p_{u,v} - sum_u f^p_{v,u} = 0  and  sum_u f^p_{u,v} - y_v <= 0
        for facility in facilities:
            cons_coeff = {idx: 1.0 for idx in inflow[facility]}
            for idx in outflow[facility]:
                cons_coeff[idx] = -1.0
            if cons_coeff:
                rows.append((cons_coeff, "E", 0.0))
            if inflow[facility]:
                cap_coeff = {idx: 1.0 for idx in inflow[facility]}
                cap_coeff[y[facility]] = -1.0
                rows.append((cap_coeff, "L", 0.0))

    _add_rows(model, rows)
    return y


def solve_flow_lp_relaxation(
    instance: TwoRadiusInstance,
    deadline: float,
) -> dict:
    """Solve the LP relaxation of the compact FLOW-IP on its own.

    The value is used as the certified lower bound for every case in which the
    FLOW-IP was stopped before optimality, because CPLEX reports no usable MIP
    best bound when its time limit expires inside the root cut loop.  Barrier
    with crossover is used to obtain an optimal basic solution without the
    dual-simplex long tail observed on these degenerate UDG instances.
    """
    start = time.perf_counter()
    estimated_variables, estimated_constraints = estimate_flow_ip_size(instance)
    if estimated_variables > MAX_FLOW_IP_VARIABLES:
        return {
            "status": "skipped_too_large",
            "objective": None,
            "seconds": time.perf_counter() - start,
            "variables": estimated_variables,
            "constraints": estimated_constraints,
            "error": (
                "the LP relaxation was skipped because the estimated compact "
                f"size ({estimated_variables} variables) exceeds the "
                f"{MAX_FLOW_IP_VARIABLES}-variable memory guard"
            ),
        }
    model = _new_model(_remaining(deadline), mip=False, lp_method="default")
    model.parameters.lpmethod.set(model.parameters.lpmethod.values.barrier)
    model.parameters.barrier.crossover.set(
        model.parameters.barrier.crossover.values.auto)
    _build_flow_model(instance, model, facility_types=None)
    model.parameters.timelimit.set(_remaining(deadline))
    model.solve()
    status = model.solution.get_status_string()
    status_code = model.solution.get_status()
    optimum_codes = {model.solution.status.optimal}
    tolerance_code = getattr(
        model.solution.status, "optimal_tolerance", None)
    if tolerance_code is not None:
        optimum_codes.add(tolerance_code)
    # Barrier with crossover can also terminate with an "optimal" status string
    # that carries a different code, so the string is checked as well; the
    # primal feasibility flag guards against reporting a value from a run that
    # merely stopped near optimality.
    optimal = (
        status_code in optimum_codes or "optimal" in status.lower()
    ) and model.solution.is_primal_feasible()
    objective = model.solution.get_objective_value() if optimal else None
    return {
        "status": status,
        "proven_optimal": optimal,
        "objective": objective,
        "seconds": time.perf_counter() - start,
        "variables": model.variables.get_num(),
        "constraints": model.linear_constraints.get_num(),
        "method": "barrier with crossover",
    }


def solve_flow_benchmark(
    instance: TwoRadiusInstance,
    deadline: float,
) -> dict:
    """Solve FLOW-IP once and capture its first root relaxation objective."""
    start = time.perf_counter()
    estimated_variables, estimated_constraints = estimate_flow_ip_size(instance)
    if estimated_variables > MAX_FLOW_IP_VARIABLES:
        # The compact model would exceed the memory budget, so no IP optimum is
        # produced; the reason is recorded instead of being silently dropped.
        return {
            "status": FLOW_IP_TOO_LARGE_STATUS,
            "method": "flow",
            "proven_optimal": False,
            "objective": None,
            "best_bound": None,
            "root_relaxation": None,
            "root_relaxation_source": None,
            "root_gap_status": "unavailable",
            "root_gap_ratio": None,
            "root_gap_percent": None,
            "root_lp_solved": False,
            "root_lp_marker_found": False,
            "root_lp_note": (
                "the compact FLOW-IP was skipped by the memory guard, so no "
                "root LP relaxation was computed"
            ),
            "selected_values": None,
            "verified": None,
            "verification_seconds": 0.0,
            "mip_relative_gap": None,
            "nodes": None,
            "variables": estimated_variables,
            "constraints": estimated_constraints,
            "estimated_variables": estimated_variables,
            "estimated_constraints": estimated_constraints,
            "skipped_reason": (
                "estimated compact FLOW-IP size ("
                f"{estimated_variables} variables, {estimated_constraints} rows) "
                f"exceeds the {MAX_FLOW_IP_VARIABLES}-variable memory guard; "
                "the model was not built and no IP optimum is available"
            ),
            "seconds": time.perf_counter() - start,
        }
    model = _new_model(
        _remaining(deadline),
        mip=True,
        lp_method="default",
        capture_log=True,
    )
    # Level 2 emits the node log (including the first root relaxation line)
    # without verbose per-node LP output.  Periodic node lines are suppressed
    # because only the root line is parsed.
    model.parameters.mip.display.set(2)
    model.parameters.mip.interval.set(MIP_NODE_LOG_INTERVAL)
    y = _build_flow_model(instance, model, facility_types="B")
    model.parameters.timelimit.set(_remaining(deadline))
    # On this CPLEX build `set_log_stream` delivers nothing to a Python object;
    # the node log is written to the process stdout descriptor instead, so the
    # root relaxation value is captured by redirecting that descriptor.
    with _CapturedSolverLog() as captured:
        model.solve()
        status = model.solution.get_status_string()
        status_code = model.solution.get_status()
    root_relaxation = _extract_first_root_relaxation(captured.text)
    root_lp_marker_found = _root_lp_completed(captured.text)
    root_lp_solved = root_relaxation is not None or root_lp_marker_found
    if root_relaxation is not None:
        root_lp_note = None
    elif root_lp_marker_found:
        root_lp_note = (
            "CPLEX printed the 'Root relaxation' marker but never the node "
            "table line, so the numeric root LP value could not be parsed; the "
            "time limit expired inside the root cut loop and the separately "
            "solved LP relaxation is used as the certified lower bound"
        )
    else:
        root_lp_note = (
            "CPLEX printed no root relaxation line, so the FLOW-IP root LP "
            "relaxation was not solved before the solve stopped "
            "(time limit hit during presolve or during the root LP)"
        )

    # Safely extract best bound and incumbent even if the time limit is hit.
    try:
        best_bound = model.solution.MIP.get_best_objective()
    except Exception:
        best_bound = None
    has_incumbent = model.solution.is_primal_feasible()
    objective = (
        model.solution.get_objective_value() if has_incumbent else None
    )
    selected_values = None
    verified = None
    verification_seconds = 0.0
    if has_incumbent:
        values = model.solution.get_values()
        selected_values = {
            facility: values[index] for facility, index in y.items()
        }
        selected = {
            facility for facility, value in selected_values.items() if value > 0.5
        }
        verification_start = time.perf_counter()
        verified = verify_design(instance, set(selected))
        verification_seconds = time.perf_counter() - verification_start
    try:
        mip_gap = model.solution.MIP.get_mip_relative_gap() if has_incumbent else None
    except Exception:
        mip_gap = None
    optimal_codes = {model.solution.status.MIP_optimal}
    optimal_tolerance_code = getattr(
        model.solution.status, "MIP_optimal_tolerance", None)
    if optimal_tolerance_code is not None:
        optimal_codes.add(optimal_tolerance_code)
    proven_optimal = status_code in optimal_codes

    root_gap_ratio = None
    root_gap_percent = None
    root_gap_status = "unavailable"
    if not root_lp_solved:
        root_gap_status = "root_lp_not_solved"
    if proven_optimal and verified is True and root_relaxation is not None:
        difference = objective - root_relaxation
        root_tolerance = ROOT_GAP_TOL * max(1.0, abs(objective))
        if difference < -root_tolerance:
            root_gap_status = "inconsistent_root_bound"
        elif abs(difference) <= root_tolerance:
            root_gap_status = "root_tight"
        else:
            root_gap_status = "positive_root_gap"
        if objective > TOL and difference >= -root_tolerance:
            root_gap_ratio = max(0.0, difference / objective)
            root_gap_percent = 100.0 * root_gap_ratio
    return {
        "status": status,
        "method": "flow",
        "proven_optimal": proven_optimal,
        "objective": objective,
        "best_bound": best_bound,
        "root_relaxation": root_relaxation,
        "root_relaxation_source": (
            "first node-0 line of the CPLEX MIP node log captured from the same "
            "FLOW-IP solve; an 'integral' objective column is read from the "
            "incumbent/bound columns of that line"
        ),
        "root_gap_status": root_gap_status,
        "root_gap_ratio": root_gap_ratio,
        "root_gap_percent": root_gap_percent,
        "root_lp_solved": root_lp_solved,
        "root_lp_marker_found": root_lp_marker_found,
        "root_lp_note": root_lp_note,
        "selected_values": selected_values,
        "verified": verified,
        "verification_seconds": verification_seconds,
        "mip_relative_gap": mip_gap,
        "nodes": model.solution.progress.get_num_nodes_processed(),
        "variables": model.variables.get_num(),
        "constraints": model.linear_constraints.get_num(),
        "seconds": time.perf_counter() - start,
    }


def solve_node_benchmark(
    instance: TwoRadiusInstance,
    deadline: float,
    *,
    relaxed: bool,
) -> dict:
    """Solve the exact node model or its LP by repeated exact cut separation."""
    start = time.perf_counter()
    model = _new_model(_remaining(deadline), mip=not relaxed, lp_method="default")
    facilities = instance.facilities
    y = {facility: index for index, facility in enumerate(facilities)}
    lower = [
        1.0 if instance.variant == "preinstalled" and v in instance.depots else 0.0
        for v in facilities
    ]
    types = "" if relaxed else "B" * len(facilities)
    kwargs = {} if relaxed else {"types": types}
    model.variables.add(
        names=[f"y_{v}" for v in facilities],
        obj=[instance.costs[v] for v in facilities],
        lb=lower,
        ub=[1.0] * len(facilities),
        **kwargs,
    )
    edge_indices = {}
    for edge in instance.backbone_edges:
        edge_indices[edge] = model.variables.get_num()
        model.variables.add(names=[f"a_{edge[0]}_{edge[1]}"], lb=[0.0], ub=[1.0])
    service_indices = {}
    for demand in instance.demand_positions:
        for facility in instance.coverage[demand]:
            service_indices[demand, facility] = model.variables.get_num()
            model.variables.add(
                names=[f"b_{facility}_{demand}"], lb=[0.0], ub=[1.0])
    linking_rows = []
    for (u, v), index in edge_indices.items():
        if u != ROOT:
            linking_rows.append(({index: 1.0, y[u]: -1.0}, "L", 0.0))
        if v != ROOT:
            linking_rows.append(({index: 1.0, y[v]: -1.0}, "L", 0.0))
    for (demand, facility), index in service_indices.items():
        linking_rows.append(({index: 1.0, y[facility]: -1.0}, "L", 0.0))
    for demand, requirement in instance.requirements.items():
        linking_rows.append((
            {
                service_indices[demand, facility]: 1.0
                for facility in instance.coverage[demand]
            },
            "G",
            float(requirement),
        ))
    _add_rows(model, linking_rows)

    signatures = set()
    trace = []
    total_cuts = 0
    separated_incumbent = False
    last_objective = None
    last_values = None
    while True:
        model.parameters.timelimit.set(_remaining(deadline))
        model.solve()
        status = model.solution.get_status_string()
        if (
            relaxed
            and model.solution.get_status() != model.solution.status.optimal
        ):
            return {
                "status": status,
                "method": "cut",
                "objective": None,
                "best_bound": None,
                "seconds": time.perf_counter() - start,
                "cuts_generated": total_cuts,
                "trace": trace,
                "variables": model.variables.get_num(),
                "constraints": model.linear_constraints.get_num(),
                "error": "Node LP did not reach a certified optimum",
            }
        if not model.solution.is_primal_feasible():
            return {
                "status": status,
                "method": "cut",
                "objective": None,
                "best_bound": (
                    model.solution.MIP.get_best_objective() if not relaxed else None
                ),
                "seconds": time.perf_counter() - start,
                "cuts_generated": total_cuts,
                "trace": trace,
                "variables": model.variables.get_num(),
                "constraints": model.linear_constraints.get_num(),
            }
        values = model.solution.get_values()
        last_values = values
        last_objective = model.solution.get_objective_value()
        separation_start = time.perf_counter()
        rows, minimum_value = _design_cut_rows(
            instance, edge_indices, service_indices, values)
        unique_rows = []
        for row in rows:
            if not row[0]:
                return {
                    "status": "infeasible",
                    "method": "cut",
                    "objective": None,
                    "best_bound": (
                        model.solution.MIP.get_best_objective()
                        if not relaxed else None
                    ),
                    "seconds": time.perf_counter() - start,
                    "cuts_generated": total_cuts,
                    "trace": trace,
                    "variables": model.variables.get_num(),
                    "constraints": model.linear_constraints.get_num(),
                    "error": "A required cut has no available edge",
                }
            signature = (tuple(sorted(row[0].items())), row[1], row[2])
            if row[0] and signature not in signatures:
                signatures.add(signature)
                unique_rows.append(row)
        separation_seconds = time.perf_counter() - separation_start
        trace.append({
            "solve": len(trace) + 1,
            "status": status,
            "objective": last_objective,
            "best_bound": (
                model.solution.MIP.get_best_objective() if not relaxed else None
            ),
            "new_cuts": len(unique_rows),
            "minimum_cut_value": minimum_value,
            "separation_seconds": separation_seconds,
        })
        if not unique_rows:
            separated_incumbent = True
            break
        _add_rows(model, unique_rows)
        total_cuts += len(unique_rows)
        _remaining(deadline)

    selected_values = {
        facility: last_values[index] for facility, index in y.items()
    }
    selected = {
        facility for facility, value in selected_values.items() if value > 0.5
    }
    verified = (
        verify_design(instance, set(selected))
        if not relaxed and separated_incumbent else None
    )
    best_bound = (
        model.solution.MIP.get_best_objective() if not relaxed else last_objective
    )
    return {
        "status": model.solution.get_status_string(),
        "method": "cut",
        "proven_optimal": (
            not relaxed
            and separated_incumbent
            and model.solution.get_status() == model.solution.status.MIP_optimal
        ),
        "objective": last_objective,
        "best_bound": best_bound,
        "selected_values": selected_values,
        "verified": verified,
        "mip_relative_gap": (
            model.solution.MIP.get_mip_relative_gap() if not relaxed else None
        ),
        "nodes": (
            model.solution.progress.get_num_nodes_processed() if not relaxed else None
        ),
        "optimal_basic": (
            model.solution.get_solution_type() == model.solution.type.basic
            if relaxed else None
        ),
        "cuts_generated": total_cuts,
        "trace": trace,
        "variables": model.variables.get_num(),
        "constraints": model.linear_constraints.get_num(),
        "seconds": time.perf_counter() - start,
    }


def run_case(
    instance: TwoRadiusInstance,
    time_limit: float = 1800.0,
    epsilon: float = 0.20,
    cover_method: str = "greedy",
    separator_backend: str = "scipy",
    cut_mode: str = "double",
    nested_levels: int = 4,
    cut_cap_per_terminal: int = 0,
    cut_cap_per_round: int = 0,
    skip_ip_stations: int = 0,
    lp_fill_max_stations: int = 0,
) -> dict:
    start = time.perf_counter()
    algorithm_deadline = start + time_limit
    record = {
        "instance": instance.as_dict(),
        "two_radius_semantics": {
            "backbone_rule": "facility-facility distance <= flight_radius",
            "service_rule": "facility-demand distance <= service_radius",
            "demands_are_backbone_vertices": False,
            "condition_2r_le_R": (
                2 * instance.service_radius <= instance.flight_radius + 1e-10
            ),
        },
    }
    try:
        stage_one = stage_one_cover(
            instance, epsilon, algorithm_deadline, cover_method)
        algorithm = iterative_round_stage_two(
            instance,
            stage_one,
            algorithm_deadline,
            separator_backend,
            cut_mode=cut_mode,
            nested_levels=nested_levels,
            cut_cap_per_terminal=cut_cap_per_terminal,
            cut_cap_per_round=cut_cap_per_round,
        )
        algorithm["stage_one"] = stage_one
        algorithm["seconds_total"] = time.perf_counter() - start
        record["algorithm"] = algorithm
    except Exception as error:
        record["algorithm"] = {
            "status": (
                "time_limit" if isinstance(error, TimeoutError) else "failed"
            ),
            "error": f"{type(error).__name__}: {error}",
            "seconds_total": time.perf_counter() - start,
        }
    # FLOW-IP receives its own time limit.  Its initial root relaxation is
    # captured from this same solve; there is deliberately no standalone LP.
    if skip_ip_stations > 0 and len(instance.stations) >= skip_ip_stations:
        record["ip"] = {
            "status": FLOW_IP_SKIPPED_STATUS,
            "method": "flow",
            "proven_optimal": False,
            "objective": None,
            "seconds": 0.0,
            "stations": len(instance.stations),
            "error": (
                "the FLOW-IP benchmark was not attempted because this tier "
                f"({len(instance.stations)} stations) is at or above the "
                f"--skip-ip-stations threshold of {skip_ip_stations}; only the "
                "algorithm and the LP relaxation lower bound are reported"
            ),
        }
    else:
        try:
            record["ip"] = solve_flow_benchmark(
                instance, time.perf_counter() + time_limit)
        except Exception as error:
            record["ip"] = {
                "status": "time_limit" if isinstance(error, TimeoutError) else "failed",
                "method": "flow",
                "error": f"{type(error).__name__}: {error}",
                "seconds": None,
            }

    # At the largest tiers CPLEX burns its whole limit inside the root cut loop,
    # so it reports neither an incumbent nor a usable best bound.  The LP
    # relaxation of the identical compact model is then solved on its own so
    # every case still exposes a certified lower bound for the error report.
    fill_lp_relaxation(
        instance, record["ip"], time_limit, lp_fill_max_stations)

    apply_quality_fields(record)
    record["case_seconds"] = time.perf_counter() - start
    return record


def fill_lp_relaxation(
    instance: TwoRadiusInstance,
    ip: dict,
    time_limit: float,
    max_stations: int = 0,
) -> None:
    """Attach a separately solved LP relaxation to ``ip`` when one is missing.

    CPLEX's final best bound survives the root cut loop and, whenever it is
    meaningful, is at least as tight as the plain LP relaxation.  It is however
    reported as a tiny value when the time limit expires inside the root cut
    loop, so for every case the FLOW-IP did not prove optimal the LP relaxation
    of the same rows, columns and coefficients is solved as well; taking the
    larger of the two can never overstate the certified quality.
    """
    if ip.get("status") == FLOW_IP_TOO_LARGE_STATUS:
        return
    if ip.get("proven_optimal") is True:
        return
    if max_stations > 0 and len(instance.stations) > max_stations:
        ip["lp_relaxation_status"] = "skipped_station_guard"
        ip["lp_relaxation"] = None
        ip["lp_relaxation_error"] = (
            "the standalone LP relaxation was skipped because this tier has "
            f"{len(instance.stations)} stations, above the "
            f"--lp-fill-max-stations threshold of {max_stations}"
        )
        return
    existing = ip.get("lp_relaxation")
    if (
        isinstance(existing, (int, float))
        and not isinstance(existing, bool)
        and existing > 0
    ):
        return
    try:
        relaxation = solve_flow_lp_relaxation(
            instance, time.perf_counter() + time_limit)
    except Exception as error:
        relaxation = {
            "status": "failed",
            "objective": None,
            "seconds": None,
            "method": "barrier with crossover",
            "error": f"{type(error).__name__}: {error}",
        }
    for key, value in (
        ("lp_relaxation", relaxation.get("objective")),
        ("lp_relaxation_status", relaxation.get("status")),
        ("lp_relaxation_seconds", relaxation.get("seconds")),
        ("lp_relaxation_method", relaxation.get("method")),
        ("lp_relaxation_variables", relaxation.get("variables")),
        ("lp_relaxation_error", relaxation.get("error")),
    ):
        ip[key] = value


def strongest_lower_bound(ip: dict) -> tuple[float | None, str | None]:
    """Return the strongest usable lower bound on the FLOW-IP optimum.

    CPLEX's final MIP best bound survives the root cut loop and is usually the
    tighter of the two, but it degenerates to a tiny value when the time limit
    expires inside that loop, so a separately solved LP relaxation of the same
    compact model is kept alongside it.  Both are valid lower bounds, hence
    taking the maximum can never overstate the certified quality.
    """
    candidates: list[tuple[float, str]] = []
    for key, basis in (
        ("best_bound", "final_ip_lower_bound"),
        ("lp_relaxation", "lp_relaxation_lower_bound"),
    ):
        value = ip.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            candidates.append((float(value), basis))
    if not candidates:
        return None, None
    return max(candidates)


def apply_quality_fields(record: dict) -> None:
    """Classify the FLOW-IP outcome and derive the quality gaps of one case.

    The algorithm cost is compared against the exact IP optimum when CPLEX
    proved it, and otherwise against the strongest available lower bound, so
    that every case reports either an exact relative error or a certified error
    upper bound.
    """
    algorithm = record.get("algorithm", {})
    ip = record.get("ip", {})
    for stale in (
        "algorithm_over_ip",
        "algorithm_exact_gap_percent",
        "algorithm_over_ip_bound",
        "algorithm_certified_gap_upper_percent",
        "quality_gap_percent",
        "quality_gap_basis",
    ):
        record.pop(stale, None)

    if ip.get("status") == FLOW_IP_TOO_LARGE_STATUS:
        ip["status_class"] = "skipped_too_large"
    elif ip.get("status") == FLOW_IP_SKIPPED_STATUS:
        ip["status_class"] = "skipped_by_request"
    elif ip.get("objective") is not None and ip.get("verified") is False:
        ip["status_class"] = "invalid_incumbent"
    elif ip.get("proven_optimal") is True:
        ip["status_class"] = "optimal"
    elif ip.get("objective") is not None:
        ip["status_class"] = "open_gap"
    else:
        ip["status_class"] = "no_incumbent"

    algorithm_is_valid = (
        algorithm.get("status") == "feasible"
        and algorithm.get("verified") is True
    )
    if (
        algorithm_is_valid
        and ip.get("objective") is not None
        and ip.get("verified") is True
        and ip.get("proven_optimal") is True
        and ip["objective"] > 0
    ):
        record["algorithm_over_ip"] = algorithm["cost"] / ip["objective"]
        record["algorithm_exact_gap_percent"] = 100.0 * (
            record["algorithm_over_ip"] - 1.0
        )
        record["quality_gap_percent"] = record["algorithm_exact_gap_percent"]
        record["quality_gap_basis"] = "exact_ip_optimum"
        return
    lower_bound, basis = strongest_lower_bound(ip)
    if algorithm_is_valid and lower_bound is not None:
        record["algorithm_over_ip_bound"] = algorithm["cost"] / lower_bound
        record["algorithm_certified_gap_upper_percent"] = 100.0 * (
            record["algorithm_over_ip_bound"] - 1.0
        )
        record["quality_gap_percent"] = (
            record["algorithm_certified_gap_upper_percent"]
        )
        record["quality_gap_basis"] = basis


def summary_row(record: dict) -> dict:
    instance = record["instance"]
    generator = record.get("generator", {})
    algorithm = record.get("algorithm", {})
    stage_one = algorithm.get("stage_one", {})
    ip = record.get("ip", {})
    num_k1 = sum(value == 1 for value in instance["requirements"].values())
    num_k2 = sum(value == 2 for value in instance["requirements"].values())
    req_mode = (
        "all1" if num_k2 == 0 else ("all2" if num_k1 == 0 else "mixed")
    )
    depots = set(instance["depots"])
    stations = set(instance["stations"])
    costs = instance["costs"]
    algorithm_selected = (
        set(algorithm["selected"]) if algorithm.get("selected") is not None else None
    )
    ip_selected = (
        {
            facility
            for facility, value in ip["selected_values"].items()
            if value > 0.5
        }
        if ip.get("selected_values") is not None else None
    )

    def _selected_count(
        selected: set[str] | None, candidates: set[str]
    ) -> int | None:
        return len(selected & candidates) if selected is not None else None

    def _selected_cost(
        selected: set[str] | None, candidates: set[str]
    ) -> int | None:
        if selected is None:
            return None
        return sum(costs[facility] for facility in selected & candidates)

    lower_bound_used, lower_bound_basis = strongest_lower_bound(ip)

    return {
        "variant": instance["variant"],
        "paired_instance_id": generator.get("paired_instance_id"),
        "paired_instance_sha256": generator.get("paired_instance_sha256"),
        "requirement_mode": req_mode,
        "seed": instance["seed"],
        "num_demands": len(instance["demand_positions"]),
        "num_stations": len(instance["stations"]),
        "num_depots": len(instance["depots"]),
        "flight_radius_R": instance["flight_radius"],
        "service_radius_r": instance["service_radius"],
        "two_r_over_R": 2 * instance["service_radius"] / instance["flight_radius"],
        "critical_flight_radius": generator.get("critical_flight_radius"),
        "critical_flight_radius_computed": generator.get(
            "critical_flight_radius_computed"),
        "facility_layout_draws": generator.get("facility_layout_draws"),
        "eta": generator.get("eta"),
        "physical_edges": generator.get("facility_graph_edges"),
        "facility_average_degree": generator.get("facility_graph_average_degree"),
        "coverage_degree_min": generator.get("coverage_degree_min"),
        "coverage_degree_mean": generator.get("coverage_degree_mean"),
        "coverage_degree_max": generator.get("coverage_degree_max"),
        "num_k1": num_k1,
        "num_k2": num_k2,
        "algorithm_status": algorithm.get("status"),
        "algorithm_cost_U": algorithm.get("cost"),
        "algorithm_opened_depots": _selected_count(algorithm_selected, depots),
        "algorithm_opened_stations": _selected_count(algorithm_selected, stations),
        "algorithm_depot_cost": _selected_cost(algorithm_selected, depots),
        "algorithm_station_cost": _selected_cost(algorithm_selected, stations),
        "algorithm_depot_cost_share": (
            _selected_cost(algorithm_selected, depots) / algorithm["cost"]
            if algorithm.get("cost") is not None and algorithm["cost"] > 0
            else None
        ),
        "separator_backend": algorithm.get("separator_backend"),
        "cut_mode": algorithm.get("cut_mode"),
        "nested_levels": algorithm.get("nested_levels"),
        "cut_cap_per_terminal": algorithm.get("cut_cap_per_terminal"),
        "cut_cap_per_round": algorithm.get("cut_cap_per_round"),
        "stage_one_cost": stage_one.get("cost"),
        "stage_two_added_cost": algorithm.get("stage_two_added_cost"),
        "stage_two_over_edge_lp": algorithm.get("stage_two_over_edge_lp"),
        "cover_method": stage_one.get("cover_method"),
        "edge_lp_initial_value": algorithm.get("initial_edge_lp"),
        "initial_positive_support_edges": algorithm.get(
            "initial_positive_support_edges"),
        "benchmark_method": ip.get("method"),
        "ip_status": ip.get("status"),
        "ip_status_class": ip.get("status_class"),
        "ip_estimated_variables": ip.get("estimated_variables"),
        "ip_skipped_reason": ip.get("skipped_reason"),
        "ip_proven_optimal": ip.get("proven_optimal", False),
        "ip_objective": ip.get("objective"),
        "ip_opened_depots": _selected_count(ip_selected, depots),
        "ip_opened_stations": _selected_count(ip_selected, stations),
        "ip_depot_cost": _selected_cost(ip_selected, depots),
        "ip_station_cost": _selected_cost(ip_selected, stations),
        "ip_depot_cost_share": (
            _selected_cost(ip_selected, depots) / ip["objective"]
            if ip.get("objective") is not None and ip["objective"] > 0
            else None
        ),
        "ip_best_bound": ip.get("best_bound"),
        "ip_mip_gap": ip.get("mip_relative_gap"),
        "ip_root_relaxation": ip.get("root_relaxation"),
        "ip_root_lp_solved": ip.get("root_lp_solved"),
        "ip_root_lp_marker_found": ip.get("root_lp_marker_found"),
        "ip_root_lp_note": ip.get("root_lp_note"),
        "ip_root_gap_status": ip.get("root_gap_status"),
        "ip_root_gap_percent": ip.get("root_gap_percent"),
        "ip_lp_relaxation": ip.get("lp_relaxation"),
        "ip_lp_relaxation_status": ip.get("lp_relaxation_status"),
        "ip_lp_relaxation_seconds": ip.get("lp_relaxation_seconds"),
        "ip_lower_bound_used": lower_bound_used,
        "ip_lower_bound_basis": lower_bound_basis,
        "algorithm_over_ip": record.get("algorithm_over_ip"),
        "algorithm_over_ip_bound": record.get("algorithm_over_ip_bound"),
        "algorithm_exact_gap_percent": record.get(
            "algorithm_exact_gap_percent"),
        "algorithm_certified_gap_upper_percent": record.get(
            "algorithm_certified_gap_upper_percent"),
        "quality_gap_percent": record.get("quality_gap_percent"),
        "quality_gap_basis": record.get("quality_gap_basis"),
        "algorithm_seconds": algorithm.get("seconds_total"),
        "stage_one_seconds": stage_one.get("seconds"),
        "stage_two_seconds": algorithm.get("seconds"),
        "stage_two_lp_solver_seconds": algorithm.get("lp_solver_seconds"),
        "stage_two_separation_seconds": algorithm.get("lp_separation_seconds"),
        "stage_two_max_flow_calls": algorithm.get("max_flow_calls"),
        "stage_two_full_test_seconds": algorithm.get("full_graph_test_seconds"),
        "stage_two_rounding_test_seconds": algorithm.get("rounding_test_seconds"),
        "stage_two_pruning_seconds": algorithm.get("pruning_seconds"),
        "stage_two_verification_seconds": algorithm.get("verification_seconds"),
        "stage_two_integer_test_calls": algorithm.get("integer_test_calls"),
        "stage_two_pruning_test_calls": algorithm.get("pruning_test_calls"),
        "edges_before_pruning": algorithm.get("edges_before_pruning"),
        "edges_after_pruning": algorithm.get("edges_after_pruning"),
        "edges_pruned": algorithm.get("edges_pruned"),
        "ip_seconds": ip.get("seconds"),
        "ip_nodes": ip.get("nodes"),
        "ip_verification_seconds": ip.get("verification_seconds"),
        "case_seconds": record.get("case_seconds"),
        "rounds": algorithm.get("num_rounds"),
        "master_lp_solves": algorithm.get("master_lp_solves"),
        "stage_two_cuts": algorithm.get("cuts_retained"),
        "secondary_rows_added": algorithm.get("secondary_rows_added"),
        "nested_rows_added": algorithm.get("nested_rows_added"),
        "nested_sides_tried": algorithm.get("nested_sides_tried"),
        "algorithm_verified": algorithm.get("verified"),
        "ip_verified": ip.get("verified"),
        "error": algorithm.get("error") or ip.get("error"),
    }


def build_table_summary(rows: list[dict]) -> list[dict]:
    """Aggregate the pre-registered main-table and diagnostic quantities."""
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        if "num_stations" not in row:
            continue
        key = (
            row["num_depots"],
            row["num_stations"],
            row["num_demands"],
            row["variant"],
            row.get("requirement_mode", "all2"),
            row.get("cut_mode") or "double",
        )
        grouped.setdefault(key, []).append(row)

    def _mean(vals: list[float]) -> float | None:
        return statistics.mean(vals) if vals else None

    def _max(vals: list[float]) -> float | None:
        return max(vals) if vals else None

    table_rows = []
    for (d, f, p, variant, req_mode, cut_mode), group in grouped.items():
        quality_gaps = [
            r["quality_gap_percent"]
            for r in group
            if r.get("quality_gap_percent") is not None
        ]
        exact_gaps = [
            r["algorithm_exact_gap_percent"]
            for r in group
            if r.get("algorithm_exact_gap_percent") is not None
        ]
        certified_gaps = [
            r["algorithm_certified_gap_upper_percent"]
            for r in group
            if r.get("algorithm_certified_gap_upper_percent") is not None
        ]
        rounds_list = [r["rounds"] for r in group if r.get("rounds") is not None]
        pruned_list = [
            r["edges_pruned"] for r in group if r.get("edges_pruned") is not None
        ]
        t_alg = [
            r["algorithm_seconds"]
            for r in group
            if r.get("algorithm_seconds") is not None
        ]
        t_ip = [
            r["ip_seconds"] for r in group if r.get("ip_seconds") is not None
        ]
        lp_solves = [
            r["master_lp_solves"]
            for r in group
            if r.get("master_lp_solves") is not None
        ]
        cuts_list = [
            r["stage_two_cuts"]
            for r in group
            if r.get("stage_two_cuts") is not None
        ]
        t_lp_solver = [
            r["stage_two_lp_solver_seconds"]
            for r in group
            if r.get("stage_two_lp_solver_seconds") is not None
        ]
        t_separation = [
            r["stage_two_separation_seconds"]
            for r in group
            if r.get("stage_two_separation_seconds") is not None
        ]
        t_verification = [
            r["stage_two_verification_seconds"]
            for r in group
            if r.get("stage_two_verification_seconds") is not None
        ]
        secondary_rows = [
            r["secondary_rows_added"]
            for r in group
            if r.get("secondary_rows_added") is not None
        ]
        nested_rows = [
            r["nested_rows_added"]
            for r in group
            if r.get("nested_rows_added") is not None
        ]
        table_rows.append({
            "instance_class_D_F_P": f"({d},{f},{p}) {variant}",
            "num_depots": d,
            "num_stations": f,
            "num_demands": p,
            "variant": variant,
            "requirement_mode": req_mode,
            "cut_mode": cut_mode,
            "num_seeds": len(group),
            "num_algorithm_feasible": sum(
                r.get("algorithm_verified") is True for r in group
            ),
            "num_ip_optimal": sum(
                r.get("ip_status_class") == "optimal" for r in group
            ),
            "num_ip_open_gap": sum(
                r.get("ip_status_class") == "open_gap" for r in group
            ),
            "num_ip_no_incumbent": sum(
                r.get("ip_status_class") == "no_incumbent" for r in group
            ),
            "num_ip_invalid_incumbent": sum(
                r.get("ip_status_class") == "invalid_incumbent" for r in group
            ),
            "num_ip_skipped_too_large": sum(
                r.get("ip_status_class") == "skipped_too_large" for r in group
            ),
            "num_ip_skipped_by_request": sum(
                r.get("ip_status_class") == "skipped_by_request" for r in group
            ),
            "num_root_relaxation_available": sum(
                r.get("ip_root_relaxation") is not None for r in group
            ),
            "num_root_lp_solved": sum(
                r.get("ip_root_lp_solved") is True for r in group
            ),
            "num_root_lp_not_solved": sum(
                r.get("ip_root_lp_solved") is False for r in group
            ),
            "num_bound_from_final_ip": sum(
                r.get("ip_lower_bound_basis") == "final_ip_lower_bound"
                for r in group
            ),
            "num_bound_from_lp_relaxation": sum(
                r.get("ip_lower_bound_basis") == "lp_relaxation_lower_bound"
                for r in group
            ),
            "num_cases_without_bound": sum(
                r.get("ip_lower_bound_basis") is None
                and r.get("ip_status_class") not in (None, "skipped_too_large")
                for r in group
            ),
            "num_root_tight_among_optimal": sum(
                r.get("ip_root_gap_status") == "root_tight" for r in group
            ),
            "num_positive_root_gap_among_optimal": sum(
                r.get("ip_root_gap_status") == "positive_root_gap"
                for r in group
            ),
            "num_inconsistent_root_bounds": sum(
                r.get("ip_root_gap_status") == "inconsistent_root_bound"
                for r in group
            ),
            "num_exact_quality_gaps": len(exact_gaps),
            "num_certified_gap_upper_bounds": len(certified_gaps),
            "quality_gap_mean_percent": _mean(quality_gaps),
            "quality_gap_max_percent": _max(quality_gaps),
            "exact_gap_mean_percent": _mean(exact_gaps),
            "exact_gap_max_percent": _max(exact_gaps),
            "certified_gap_upper_mean_percent": _mean(certified_gaps),
            "certified_gap_upper_max_percent": _max(certified_gaps),
            "rounds_mean": _mean(rounds_list),
            "master_lp_solves_mean": _mean(lp_solves),
            "stage_two_cuts_mean": _mean(cuts_list),
            "secondary_rows_added_mean": _mean(secondary_rows),
            "nested_rows_added_mean": _mean(nested_rows),
            "pruned_edges_mean": _mean(pruned_list),
            "t_alg_mean_s": _mean(t_alg),
            "t_stage_two_lp_solver_mean_s": _mean(t_lp_solver),
            "t_separation_mean_s": _mean(t_separation),
            "t_verification_mean_s": _mean(t_verification),
            "t_IP_mean_s": _mean(t_ip),
        })
    return table_rows


def _write_summaries(
    csv_path: Path,
    table_csv_path: Path,
    rows: list[dict],
) -> list[dict]:
    """Write the per-case CSV and the aggregated main-table CSV."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table_rows = build_table_summary(rows)
    if table_rows:
        table_fields = list(table_rows[0].keys())
        with table_csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=table_fields)
            writer.writeheader()
            writer.writerows(table_rows)
    return table_rows


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse_scale(spec: str) -> tuple[int, int, int]:
    """Parse 'D,F,P' (depots,stations,demands) into a 3-tuple of ints."""
    parts = [int(item.strip()) for item in spec.split(",")]
    if len(parts) != 3 or parts[0] < 2 or parts[1] < 1 or parts[2] < 1:
        raise ValueError(
            f"Invalid scale '{spec}'; expected 'depots,stations,demands' with depots>=2"
        )
    return parts[0], parts[1], parts[2]


def backfill_lp_relaxations(
    jsonl_path: Path,
    csv_path: Path,
    table_csv_path: Path,
    max_stations: int = 0,
) -> None:
    """Add missing LP relaxations to an existing JSONL and rebuild summaries.

    Only cases whose FLOW-IP did not prove optimality are touched.  Because the
    generator is deterministic, the instance is regenerated from the recorded
    settings and checked against the stored pair fingerprint, so the LP
    relaxation is solved on exactly the same rows, columns and coefficients as
    the FLOW-IP that ran before.
    """
    settings_path = jsonl_path.with_name(jsonl_path.stem + "_settings.json")
    if not jsonl_path.exists() or not settings_path.exists():
        raise SystemExit(
            "--fill-lp requires the existing JSONL and settings files")
    parameters = json.loads(
        settings_path.read_text(encoding="utf-8"))["parameters"]
    generation = {
        "side": float(parameters["side"]),
        "eta": float(parameters["eta"]),
        "fixed_radius": float(parameters["flight_radius"]),
        "service_ratio": float(parameters["service_ratio"]),
        "depot_cost": tuple(parameters["depot_cost"]),
        "station_cost": tuple(parameters["station_cost"]),
        "max_facility_draws": int(parameters["max_facility_draws"]),
    }
    time_limit = float(parameters["time_limit"])

    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSONL at line {line_number}: {error.msg}")

    filled = 0
    for record in records:
        instance_data = record.get("instance")
        ip = record.get("ip")
        if instance_data is None or not isinstance(ip, dict):
            continue
        if ip.get("status") == FLOW_IP_TOO_LARGE_STATUS:
            continue
        if ip.get("proven_optimal") is True:
            continue
        instance, metadata = generate_two_radius_instance(
            len(instance_data["depots"]),
            len(instance_data["stations"]),
            len(instance_data["demand_positions"]),
            instance_data["variant"],
            _requirement_mode(instance_data["requirements"]),
            instance_data["seed"],
            **generation,
        )
        expected = record.get("generator", {}).get("paired_instance_sha256")
        if expected is not None and metadata["paired_instance_sha256"] != expected:
            print(json.dumps({
                "lp_backfill_skipped": metadata["paired_instance_id"],
                "reason": "regenerated instance fingerprint does not match",
            }, ensure_ascii=True), flush=True)
            continue
        before = ip.get("lp_relaxation")
        print(json.dumps({
            "lp_backfill_start": metadata["paired_instance_id"],
            "variant": instance_data["variant"],
        }, ensure_ascii=True), flush=True)
        fill_lp_relaxation(instance, ip, time_limit, max_stations)
        apply_quality_fields(record)
        if ip.get("lp_relaxation") != before:
            filled += 1
        print(json.dumps({
            "lp_backfill_done": metadata["paired_instance_id"],
            "lp_relaxation": ip.get("lp_relaxation"),
            "lp_relaxation_status": ip.get("lp_relaxation_status"),
            "lp_relaxation_seconds": ip.get("lp_relaxation_seconds"),
        }, ensure_ascii=True), flush=True)

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    rows = [summary_row(record) for record in records if "instance" in record]
    _write_summaries(csv_path, table_csv_path, rows)
    print(json.dumps({
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "table_summary_csv": str(table_csv_path),
        "cases": len(rows),
        "lp_relaxations_filled": filled,
    }, ensure_ascii=True), flush=True)


def _requirement_mode(requirements: dict) -> str:
    """Recover the generation mode from the stored per-demand requirement."""
    values = set(requirements.values())
    if values == {1}:
        return "all1"
    if values == {2}:
        return "all2"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-radius ORL experiment: Stage 1 greedy cover + Stage 2 UCUT "
            "iterative rounding vs. compact node-capacitated FLOW-IP."
        )
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=[",".join(map(str, scale)) for scale in DEFAULT_SCALES],
        help=(
            "List of 'depots,stations,demands' triples. The default is the "
            "five-scale pre-registered experiment; pass one triple for a pilot."
        ),
    )
    parser.add_argument(
        "--variants", nargs="+", choices=["selectable", "preinstalled"],
        default=["selectable", "preinstalled"])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--side", type=float, default=100.0)
    parser.add_argument(
        "--flight-radius", type=float, default=DEFAULT_FLIGHT_RADIUS,
        help=(
            "Uniform facility-facility flight radius R for every tier. The "
            "facility layout is resampled until the facility graph is "
            "two-root-connected at this radius; the service radius is r=R/2."
        ))
    parser.add_argument(
        "--max-facility-draws", type=int, default=MAX_FACILITY_DRAWS,
        help="Rejection-sampling budget for a two-root-connected layout.")
    parser.add_argument(
        "--eta", type=float, default=0.10,
        help="Legacy critical-radius multiplier; unused with a fixed radius.")
    parser.add_argument("--service-ratio", type=float, default=0.50)
    parser.add_argument("--depot-cost", nargs=2, type=int, default=[100, 100])
    parser.add_argument("--station-cost", nargs=2, type=int, default=[10, 30])
    parser.add_argument(
        "--separator-backend", choices=["python", "scipy"], default="scipy",
        help=(
            "Stage 2 UCUT separator: 'scipy' (A1+A2+A3 scipy C maximum_flow, "
            "default) or 'python' (A1+A3 flat-array Dinic)."))
    parser.add_argument(
        "--cut-mode", choices=["single", "double", "nested"], default="double",
        help=(
            "Stage 2 UCUT rows per max-flow: 'single' (smallest minimum cut "
            "only), 'double' (smallest and largest minimum cut) or 'nested' "
            "(additional intermediate minimum cuts from the residual lattice)."))
    parser.add_argument(
        "--nested-levels", type=int, default=4,
        help="Maximum intermediate source sides tried per max-flow in 'nested' mode.")
    parser.add_argument(
        "--cut-cap-per-terminal", type=int, default=0,
        help="At most this many rows admitted per terminal per round (0 = unlimited).")
    parser.add_argument(
        "--cut-cap-per-round", type=int, default=0,
        help=(
            "At most this many fresh rows admitted per round, preferring the "
            "largest violation and shortest rows (0 = unlimited). Every round "
            "still runs the complete separation, so unlimited termination holds."))
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument(
        "--skip-ip-stations", type=int, default=0,
        help=(
            "Do not attempt the FLOW-IP benchmark for tiers with at least this "
            "many stations (0 disables the skip). Such cases still run the "
            "algorithm and the LP relaxation lower bound, and are recorded as "
            "ip.status_class='skipped_by_request'."
        ))
    parser.add_argument(
        "--lp-fill-max-stations", type=int, default=0,
        help=(
            "Only solve the standalone LP relaxation for tiers with at most "
            "this many stations (0 = no limit). Above the threshold the LP is "
            "skipped and the reason is recorded, which keeps the backfill from "
            "re-running an LP that cannot finish on this machine."
        ))
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume a matching partial JSONL run, skip completed cases, and "
            "rebuild the CSV summaries."
        ))
    parser.add_argument(
        "--fill-lp", action="store_true",
        help=(
            "Post-process an existing JSONL instead of running new cases: for "
            "every case whose FLOW-IP left no usable lower bound, solve the LP "
            "relaxation of the same compact model, record it, and rewrite the "
            "JSONL, CSV and table summaries."
        ))
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/orl_two_radius_results.jsonl"))
    args = parser.parse_args()
    scales = [_parse_scale(spec) for spec in args.scales]
    if (
        any(d < 2 or f < 1 or p < 1 for d, f, p in scales)
        or args.side <= 0
        or args.flight_radius <= 0
        or args.max_facility_draws < 1
        or args.eta < 0
        or not 0 < args.service_ratio <= 0.5
        or args.epsilon <= 0
        or args.time_limit <= 0
        or args.nested_levels < 1
        or args.cut_cap_per_terminal < 0
        or args.cut_cap_per_round < 0
        or args.depot_cost[0] < 0
        or args.depot_cost[0] > args.depot_cost[1]
        or args.station_cost[0] < 0
        or args.station_cost[0] > args.station_cost[1]
        or args.lp_fill_max_stations < 0
    ):
        parser.error("Invalid experiment parameters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    table_csv_path = args.output.with_name(args.output.stem + "_table_summary.csv")
    settings_path = args.output.with_name(args.output.stem + "_settings.json")
    cplex_probe = cplex.Cplex()
    experiment_signature = {
        key: value
        for key, value in vars(args).items()
        if key not in {
            "output", "resume", "scales", "fill_lp", "skip_ip_stations",
            "lp_fill_max_stations",
        }
    }
    experiment_signature["scales_depots_stations_demands"] = [
        list(scale) for scale in scales
    ]
    settings = {
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scales_depots_stations_demands": [list(scale) for scale in scales],
        "experiment_signature": experiment_signature,
        "expected_cases": len(scales) * len(args.variants) * len(args.seeds),
        "environment": {
            "python": sys.version.split()[0],
            "cplex": cplex_probe.get_version(),
            "networkx": nx.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "cplex_threads": 1,
        },
        "method": {
            "two_radius_model": True,
            "instance_generation": (
                "facilities placed by rejection sampling in a 100x100 region; a "
                "layout is accepted only when the facility graph, with the "
                "virtual root joined to every depot, is two-root-connected at "
                f"the uniform flight radius R={args.flight_radius:g}; demands "
                "are sampled by rejection until each demand has at least two "
                "covering facilities within r=R/2"
            ),
            "flow_ip_memory_guard": (
                "compact FLOW-IP is skipped with reason '"
                f"{FLOW_IP_TOO_LARGE_STATUS}' when the estimated variable count "
                f"exceeds {MAX_FLOW_IP_VARIABLES}; no IP optimum is produced "
                "for such instances"
            ),
            "stage_one": "two-pass deterministic weighted greedy set cover",
            "stage_two": (
                "UCUT row generation with one split-graph threshold min-cut per "
                "terminal (cut_mode="
                + args.cut_mode
                + ", nested_levels="
                + str(args.nested_levels)
                + ", cut_cap_per_terminal="
                + str(args.cut_cap_per_terminal)
                + ", cut_cap_per_round="
                + str(args.cut_cap_per_round)
                + "); violated cuts ordered by cardinality; retained rows; dual "
                "simplex; batch x>=1/2 rounding; biconnectivity Test(J) and "
                "reverse-delete; separator backend = "
                + args.separator_backend
            ),
            "ip": "compact node-capacitated multi-commodity FLOW-IP",
            "root_relaxation": (
                "first node-0 relaxation value captured from the node log of the "
                "same FLOW-IP solve (the word 'integral' in the objective column "
                "is read from the incumbent/bound columns of that line); no "
                "standalone FLOW-LP solve is needed while the FLOW-IP reaches "
                "optimality"
            ),
            "certified_lower_bound": (
                "whenever the FLOW-IP does not prove optimality the LP "
                "relaxation of the very same compact rows, columns and "
                "coefficients is solved separately with barrier plus crossover, "
                "and the certified lower bound is the larger of CPLEX's final "
                "MIP best bound and that LP relaxation value; both are valid "
                "lower bounds, so the maximum never overstates the quality"
            ),
            "ip_skipped": (
                f"tiers with at least {args.skip_ip_stations} stations do not "
                "attempt the FLOW-IP; those cases report only the algorithm and "
                "the LP relaxation bound and are recorded as "
                "ip.status_class='skipped_by_request'"
                if args.skip_ip_stations > 0
                else "the FLOW-IP is attempted for every tier"
            ),
            "time_limit_scope": (
                f"separate {args.time_limit:g}-second limits for ALG and FLOW-IP"
            ),
            "requirements": "k_p=2 for every demand",
            "pairing": (
                "selectable and preinstalled variants share geometry, demands, "
                "requirements, and station costs for each scale and seed"
            ),
        },
    }
    if args.fill_lp:
        backfill_lp_relaxations(
            args.output, csv_path, table_csv_path, args.lp_fill_max_stations)
        return
    existing_paths = [
        path for path in (args.output, csv_path, table_csv_path, settings_path)
        if path.exists()
    ]
    if existing_paths and not args.resume:
        parser.error("Output exists; choose a new output path or pass --resume")

    records_by_case: dict[tuple[int, int, int, str, int], dict] = {}
    completed_cases: set[tuple[int, int, int, str, int]] = set()
    if args.resume:
        if not args.output.exists() or not settings_path.exists():
            parser.error("--resume requires the existing JSONL and settings files")
        previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous_settings.get("experiment_signature") != experiment_signature:
            parser.error("Resume settings do not match the existing experiment")
        # Refresh the stored metadata so the settings file describes the run
        # that is about to continue (for example the FLOW-IP skip threshold).
        settings_path.write_text(
            json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8")
        with args.output.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    existing_record = json.loads(line)
                except json.JSONDecodeError as error:
                    parser.error(
                        f"Invalid JSONL at line {line_number}: {error.msg}"
                    )
                instance_data = existing_record.get("instance")
                case_data = existing_record.get("case")
                if instance_data is not None:
                    key = (
                        len(instance_data["depots"]),
                        len(instance_data["stations"]),
                        len(instance_data["demand_positions"]),
                        instance_data["variant"],
                        instance_data["seed"],
                    )
                elif case_data is not None:
                    key = (
                        case_data["depots"], case_data["stations"],
                        case_data["demands"], case_data["variant"],
                        case_data["seed"],
                    )
                else:
                    continue
                records_by_case[key] = existing_record
        for key, existing_record in records_by_case.items():
            algorithm_status = existing_record.get("algorithm", {}).get("status")
            ip_status = existing_record.get("ip", {}).get("status")
            if (
                "instance" in existing_record
                and algorithm_status in {"feasible", "time_limit", "infeasible"}
                and ip_status is not None
                and ip_status != "failed"
            ):
                completed_cases.add(key)
    else:
        settings_path.write_text(
            json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8")

    paired_fingerprints: dict[tuple[int, int, int, int], str] = {}
    for existing_record in records_by_case.values():
        generator = existing_record.get("generator", {})
        instance_data = existing_record.get("instance")
        fingerprint = generator.get("paired_instance_sha256")
        if instance_data is None or fingerprint is None:
            continue
        pair_key = (
            len(instance_data["depots"]),
            len(instance_data["stations"]),
            len(instance_data["demand_positions"]),
            instance_data["seed"],
        )
        previous = paired_fingerprints.setdefault(pair_key, fingerprint)
        if previous != fingerprint:
            parser.error("Existing selectable/preinstalled pair fingerprints differ")

    for num_depots, num_stations, num_demands in scales:
        for variant, seed in itertools.product(args.variants, args.seeds):
            requirement_mode = "all2"
            case_key = (
                num_depots, num_stations, num_demands, variant, seed
            )
            if case_key in completed_cases:
                print(json.dumps({
                    "skipped_completed_case": case_key,
                }, ensure_ascii=True), flush=True)
                continue
            try:
                instance, generator = generate_two_radius_instance(
                    num_depots,
                    num_stations,
                    num_demands,
                    variant,
                    requirement_mode,
                    seed,
                    side=args.side,
                    eta=args.eta,
                    fixed_radius=args.flight_radius,
                    max_facility_draws=args.max_facility_draws,
                    service_ratio=args.service_ratio,
                    depot_cost=tuple(args.depot_cost),
                    station_cost=tuple(args.station_cost),
                )
                pair_key = (
                    num_depots, num_stations, num_demands, seed
                )
                fingerprint = generator["paired_instance_sha256"]
                previous_fingerprint = paired_fingerprints.setdefault(
                    pair_key, fingerprint)
                if previous_fingerprint != fingerprint:
                    raise RuntimeError(
                        "Selectable/preinstalled paired instances differ in "
                        "geometry, demands, requirements, or station costs"
                    )
                record = run_case(
                    instance,
                    time_limit=args.time_limit,
                    epsilon=args.epsilon,
                    cover_method="greedy",
                    separator_backend=args.separator_backend,
                    cut_mode=args.cut_mode,
                    nested_levels=args.nested_levels,
                    cut_cap_per_terminal=args.cut_cap_per_terminal,
                    cut_cap_per_round=args.cut_cap_per_round,
                    skip_ip_stations=args.skip_ip_stations,
                    lp_fill_max_stations=args.lp_fill_max_stations,
                )
                record["generator"] = generator
            except Exception as error:
                record = {
                    "case": {
                        "depots": num_depots,
                        "stations": num_stations,
                        "demands": num_demands,
                        "variant": variant,
                        "requirement_mode": requirement_mode,
                        "seed": seed,
                    },
                    "error": f"{type(error).__name__}: {error}",
                }
            _write_jsonl(args.output, record)
            records_by_case[case_key] = record
            row = (
                summary_row(record)
                if "instance" in record
                else {
                    "variant": variant,
                    "requirement_mode": requirement_mode,
                    "seed": seed,
                    "num_demands": num_demands,
                    "num_stations": num_stations,
                    "num_depots": num_depots,
                    "error": record["error"],
                }
            )
            print(json.dumps(row, ensure_ascii=True), flush=True)

    rows = []
    for num_depots, num_stations, num_demands in scales:
        for variant, seed in itertools.product(args.variants, args.seeds):
            case_key = (
                num_depots, num_stations, num_demands, variant, seed
            )
            record = records_by_case.get(case_key)
            if record is None:
                continue
            if "instance" in record:
                rows.append(summary_row(record))
            else:
                rows.append({
                    "variant": variant,
                    "requirement_mode": "all2",
                    "seed": seed,
                    "num_demands": num_demands,
                    "num_stations": num_stations,
                    "num_depots": num_depots,
                    "error": record.get("error"),
                })

    _write_summaries(csv_path, table_csv_path, rows)

    print(json.dumps({
        "jsonl": str(args.output),
        "csv": str(csv_path),
        "table_summary_csv": str(table_csv_path),
        "settings": str(settings_path),
        "cases": len(rows),
    }, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
