"""The proposed algorithm: greedy terminal cover (Stage 1) plus rooted
set-pair residual-LP row generation with iterative rounding (Stage 2).

Stage 2 is implemented exactly as the rooted set-pair model of the paper.
The model data -- the rooted graph ``G_2 = (V_2, E_2)``, the labeled terminal
requirement set ``R_T`` and the node-to-edge costs ``c_e = (f_i + f_j) / 2`` --
live in :class:`RootedSetPairProblem`.  A disjoint pair of vertex sets is a
:class:`SetPair`, its omitted boundary is ``Z = V_2 \\ (U1 u U2)``, and
:func:`compile_setpair_row` is the single entry point that turns a set pair plus
the currently fixed edge set ``J`` into the paper's row

``sum_{e in E(U1,U2) \\ J} x_e >= g(U1,U2) - |J cap E(U1,U2)|``,

where ``q(U1,U2) = max{0} u {q_uv}`` and ``g(U1,U2) = q(U1,U2) - |Z|``.

The split-graph min-cut oracle supplies set-pair *evidence* only: it decodes a
source side into a :class:`SetPair` and never writes a CPLEX row itself.  Every
CPLEX row can therefore be traced back to ``(U1, U2, Z, q, g, J)`` and be
recomputed from the paper's formulas.  The production model keeps only the
unfixed edges as variables and subtracts the fixed crossing contribution from
every right-hand side (``residual_mode = eliminate_fixed_edges``); the master is
rebuilt whenever ``J`` grows, while the set-pair evidence is cached across
rounds and recompiled under the new ``J``.

This module also hosts the small CPLEX model helpers shared with the exact
benchmarks in :mod:`baseline`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cplex
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph as csgraph_module

from instance import ROOT, TOL, TwoRadiusInstance


# Formulation identity recorded in every result.  The version string is used to
# keep a resumed run from mixing records produced by a different model.
FORMULATION = "rooted_set_pair_lp"
FORMULATION_VERSION = "explicit_residual_v1"
RESIDUAL_MODE = "eliminate_fixed_edges"


# Separation tolerance: a compiled set-pair row counts as violated only beyond
# this value, and the master is solved with a stricter feasibility tolerance so
# a previously added row can never be re-reported as violated.
SEPARATION_TOL = TOL


# Batch rounding threshold of the paper (``x_e >= 1/2``), shifted by a purely
# engineering tolerance; it does not change the mathematical threshold.
ROUNDING_THRESHOLD = 0.5 - 1e-9


# Separation tail guard: the residual row generation stops once the master
# objective has not moved for this many consecutive iterations.  Instances
# whose bound keeps improving never reach the guard, so only the flat
# separation tails (a couple of rows per round for dozens of rounds) are cut
# short; the measured cost of the rounded design is unchanged on the
# pre-registered suite.  Zero disables the guard.
DEFAULT_MAX_STALLED_ITERATIONS = 3


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
            # Set-pair rows count as violated only beyond 1e-8.  The master
            # must satisfy previously generated rows more accurately than that.
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


# ---------------------------------------------------------------------------

# Model layer: the rooted set-pair LP of the paper.

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetPair:
    """A disjoint pair of vertex subsets of ``V_2``.

    The omitted boundary ``Z`` is not stored: it is always recomputed from the
    full vertex set so that two objects describe the same pair exactly when
    their two sides coincide.
    """

    U1: frozenset[str]
    U2: frozenset[str]


@dataclass(frozen=True)
class SetPairRow:
    """A compiled rooted set-pair row, in original and in residual units.

    ``original_rhs`` is ``g(U1,U2)``; ``residual_rhs`` is the same right-hand
    side with the contribution of the already fixed crossing edges removed.
    ``residual_ids`` lists the global edge ids that are still variables.
    """

    pair: SetPair
    boundary: frozenset[str]
    q: int
    original_rhs: int
    crossing_ids: tuple[int, ...]
    fixed_crossing_count: int
    residual_ids: tuple[int, ...]
    residual_rhs: int


def compile_setpair_row(
    universe: frozenset[str] | set[str],
    edges: tuple[tuple[str, str], ...],
    requirements: dict[str, int],
    pair: SetPair,
    fixed_ids: frozenset[int] | set[int],
    root: str = ROOT,
    incidence: dict[str, tuple[int, ...]] | None = None,
) -> SetPairRow:
    """Compile one set pair into the paper's row under the fixed set ``J``.

    This is the unique entry point: ``q`` is the maximum of the prescribed pair
    requirements across the two sides, ``g = q - |Z|``, and the residual right
    hand side subtracts the fixed crossing edges exactly once.  ``incidence``
    is an optional ``vertex -> global edge ids`` map that only speeds up the
    crossing-edge scan; it never changes the result.
    """
    V = universe if isinstance(universe, frozenset) else frozenset(universe)
    A, B = pair.U1, pair.U2
    if A & B or not (A | B) <= V:
        raise ValueError("Invalid set pair")
    if not A or not B:
        raise ValueError("A set pair needs two non-empty sides")

    Z = V - A - B
    # The only nonzero pair requirements are (root, terminal).
    if root in A:
        q = max((requirements.get(t, 0) for t in B), default=0)
    elif root in B:
        q = max((requirements.get(t, 0) for t in A), default=0)
    else:
        q = 0
    g = q - len(Z)

    if incidence is None:
        crossing = [
            eid for eid, (u, v) in enumerate(edges)
            if (u in A and v in B) or (u in B and v in A)
        ]
    else:
        scan, other = (A, B) if len(A) <= len(B) else (B, A)
        found: list[int] = []
        for vertex in scan:
            for eid in incidence.get(vertex, ()):
                u, v = edges[eid]
                endpoint = v if u == vertex else u
                if endpoint in other:
                    found.append(eid)
        crossing = sorted(set(found))
    crossing = tuple(crossing)
    fixed_count = sum(eid in fixed_ids for eid in crossing)
    active = tuple(eid for eid in crossing if eid not in fixed_ids)
    return SetPairRow(
        pair=pair,
        boundary=frozenset(Z),
        q=q,
        original_rhs=g,
        crossing_ids=crossing,
        fixed_crossing_count=fixed_count,
        residual_ids=active,
        residual_rhs=g - fixed_count,
    )


class RootedSetPairProblem:
    """Immutable model data of the paper's rooted set-pair LP.

    It fixes ``V_2 = F u {root}``, the stable global edge numbering of
    ``E_2 = E_F u {{root, w}}``, the labeled terminal requirement set
    ``R_T`` and the node-to-edge edge costs ``c_e``.  Two fixed depot variants
    only differ in the depot opening costs, which enter through ``c_e``.
    """

    def __init__(
        self,
        instance: TwoRadiusInstance,
        cover: Iterable[str],
        terminals_one: Iterable[str],
        terminals_two: Iterable[str],
    ) -> None:
        self.instance = instance
        self.root = ROOT
        self.nodes: tuple[str, ...] = (ROOT,) + tuple(instance.facilities)
        self.universe = frozenset(self.nodes)
        # The global edge numbering is exactly the backbone edge order, so the
        # physical and the logical root edges share one stable index space and
        # no edge is ever duplicated per direction.
        self.edges: tuple[tuple[str, str], ...] = tuple(instance.backbone_edges)
        self.edge_index = {edge: index for index, edge in enumerate(self.edges)}
        incidence: dict[str, list[int]] = {}
        for index, (u, v) in enumerate(self.edges):
            incidence.setdefault(u, []).append(index)
            incidence.setdefault(v, []).append(index)
        self.incidence = {
            vertex: tuple(indices) for vertex, indices in incidence.items()
        }

        requirements: dict[str, int] = {}
        for terminal in terminals_one:
            requirements[terminal] = 1
        for terminal in terminals_two:
            requirements[terminal] = 2
        self.requirements = requirements
        self.terminals = tuple(sorted(requirements))

        cover_set = set(cover)
        reduced = {
            facility: (0 if facility in cover_set else instance.costs[facility])
            for facility in instance.facilities
        }
        # The residual cost of the root is zero, but a logical root edge into a
        # paying, unselected depot keeps its full cost through the same formula.
        reduced[ROOT] = 0
        self.reduced_cost = reduced
        self.edge_costs: tuple[float, ...] = tuple(
            (reduced[u] + reduced[v]) / 2 for u, v in self.edges
        )

    def compile(self, pair: SetPair, fixed_ids: Iterable[int]) -> SetPairRow:
        return compile_setpair_row(
            self.universe,
            self.edges,
            self.requirements,
            pair,
            fixed_ids,
            self.root,
            self.incidence,
        )

    def terminal_singleton_pairs(self) -> list[SetPair]:
        """Terminal degree cuts as set pairs, ``Z`` empty by construction."""
        pairs = []
        for terminal in self.terminals:
            if self.requirements[terminal] <= 0:
                continue
            pairs.append(SetPair(self.universe - {terminal}, frozenset({terminal})))
        return pairs


class SplitGraphSetPairOracle:
    """Rooted separator: one capacitated split graph per terminal.

    A cut of value below ``r_t`` contains either zero or one unit-capacity split
    node, so it yields an ordinary cut or a deleted-vertex cut.  The oracle
    returns only the *source sides* of sub-threshold cuts plus a certification
    flag; :meth:`decode_source_side` turns a source side into a :class:`SetPair`
    and the workspace compiles the paper's row from it.

    Both backends share the split-graph semantics and the same optimisations:

    * ``"python"`` -- a flat-array iterative Dinic on the positive support of
      ``x`` that never recurses.
    * ``"scipy"`` -- ``scipy.sparse.csgraph.maximum_flow`` on the integer-scaled
      split graph, with the minimum cut recovered from the residual graph.

    ``cut_mode`` controls how many candidate sides one max flow yields:
    ``"single"`` (only the smallest minimum cut), ``"double"`` (the smallest and
    the largest minimum cut) and ``"nested"`` (additional intermediate minimum
    cuts taken from the residual lattice).
    """

    # Integer scaling for the scipy backend. scipy.sparse.csgraph.maximum_flow
    # stores capacities as int32, so any value >= 2**31 wraps silently and
    # returns a wrong flow. The target's split arc carries value 2*SCALE (for
    # r=2 terminals), hence SCALE must satisfy 2*SCALE <= 2**31 - 1; 10**9
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
        # Scratch buffers for the flat-array Dinic, reused across calls.
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
        self.numpy = None
        self.sparse = None
        self.csgraph = None
        if backend == "scipy":
            self.numpy = np
            self.sparse = sparse
            self.csgraph = csgraph_module
            self._prepare_scaled_graph()

    def _prepare_scaled_graph(self) -> None:
        """Materialise the full split-graph CSR once; only the data changes."""
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

    def cut_sides(
        self, values: list[float], target: str, threshold: float
    ) -> tuple[float, list[set[int]], bool]:
        """Return ``(flow_value, source_sides, certified)`` for one terminal.

        ``certified`` means the minimum cut reaches the threshold, so the
        terminal passes and no side is returned.  Otherwise the returned sides
        are the smallest minimum cut, the largest minimum cut (in ``double`` and
        ``nested`` modes) and, in ``nested`` mode, intermediate minimum cuts.
        """
        if self.backend == "scipy":
            return self._cut_sides(values, target, threshold, "scipy")
        return self._cut_sides(values, target, threshold, "python")

    def exact_cut_sides(
        self, values: list[float], target: str, threshold: float
    ) -> tuple[float, list[set[int]], bool]:
        """Same as :meth:`cut_sides` but always on the exact float Dinic.

        The scipy backend floors capacities to integers; a sub-threshold scaled
        cut that is not violated in floating point can conceal a different
        violated cut, so the caller rechecks such a terminal exactly.
        """
        return self._cut_sides(values, target, threshold, "python")

    def _cut_sides(
        self,
        values: list[float],
        target: str,
        threshold: float,
        backend: str,
    ) -> tuple[float, list[set[int]], bool]:
        self.calls += 1
        self.secondary_side = None
        self.last_residual = None
        self.last_source_side = None
        if backend == "scipy":
            flow_value, source_side = self._flow_scaled(values, target, threshold)
        else:
            flow_value, source_side = self._flow_dinic(values, target, threshold)
        if source_side is None:
            return flow_value, [], True
        sides = [set(source_side)]
        if self.cut_mode != "single" and self.secondary_side is not None:
            other = set(self.secondary_side)
            if other != sides[0]:
                sides.append(other)
                self.secondary_rows_added += 1
        for side in self._nested_sides():
            if any(side == known for known in sides):
                continue
            sides.append(side)
            self.nested_rows_added += 1
        return flow_value, sides, False

    def _nested_sides(self) -> list[set[int]]:
        """Intermediate minimum source sides of the most recent max-flow.

        For a maximum flow ``f``, a set ``S`` with the source in ``S`` and the
        sink outside is a minimum cut exactly when no residual arc leaves
        ``S``, that is, when ``S`` is closed under residual reachability.  The
        two extremes are the source side of the smallest cut and the
        complement of the nodes that still reach the sink; every set in
        between is ``source_side`` unioned with the residual reachability of
        nodes that reach neither extreme.
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

    def decode_source_side(self, reachable: Iterable[int], target: str) -> SetPair:
        """Decode a split-graph source side into a valid rooted :class:`SetPair`.

        A node with ``in`` on the sink side and ``out`` on the source side can be
        moved entirely to the source side without increasing the cut, which
        gives the canonical ``(U1, Z, U2)`` partition.  A cut strictly below
        ``r_t <= 2`` crosses at most one internal unit-capacity arc, so more than
        one omitted vertex is a numerical anomaly rather than a valid set pair.
        """
        side = set(reachable)
        for facility in self.facilities:
            inside = self.number[facility, "in"]
            outside = self.number[facility, "out"]
            if outside in side and inside not in side:
                side.add(inside)
        if 0 not in side:
            raise RuntimeError("Split cut source side does not contain the root")
        if self.number[target, "out"] in side:
            raise RuntimeError("Split cut source side contains the target")
        omitted = {
            facility for facility in self.facilities
            if self.number[facility, "in"] in side
            and self.number[facility, "out"] not in side
        }
        if target in omitted or len(omitted) > 1:
            raise RuntimeError("Split cut below threshold has invalid boundary")
        U1 = {ROOT} | {
            facility for facility in self.facilities
            if facility not in omitted and self.number[facility, "out"] in side
        }
        U2 = set(self.nodes) - U1 - omitted
        if target not in U2:
            raise RuntimeError("Decoded set pair does not separate the target")
        if frozenset(self.nodes) - frozenset(U1) - frozenset(U2) != frozenset(omitted):
            raise RuntimeError("Decoded set pair boundary is inconsistent")
        return SetPair(frozenset(U1), frozenset(U2))

    def _flow_dinic(
        self, values: list[float], target: str, threshold: float
    ) -> tuple[float, set[int] | None]:
        """Dinic on the positive-support split graph."""
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
        """C max-flow on the integer-scaled split graph."""
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


class SetPairWorkspace:
    """Builds, separates and solves the explicit rooted set-pair residual LP.

    The master contains exactly the unfixed edges ``E_2 \\ J`` as variables and
    is rebuilt whenever ``J`` grows.  Set-pair evidence is cached across rounds
    and recompiled under the current ``J``, so every row always carries the
    right residual right-hand side and the right local columns.
    """

    def __init__(
        self,
        problem: RootedSetPairProblem,
        separator_backend: str = "scipy",
        cut_mode: str = "double",
        nested_levels: int = 4,
        cut_cap_per_terminal: int = 0,
        cut_cap_per_round: int = 0,
        max_separation_iterations: int = 0,
        max_stalled_iterations: int = DEFAULT_MAX_STALLED_ITERATIONS,
        stall_improvement_tolerance: float = 1e-6,
    ) -> None:
        self.problem = problem
        self.instance = problem.instance
        self.terminals_one = {
            terminal for terminal in problem.terminals
            if problem.requirements[terminal] == 1
        }
        self.terminals_two = {
            terminal for terminal in problem.terminals
            if problem.requirements[terminal] == 2
        }
        self.separator_backend = separator_backend
        self.cut_mode = cut_mode
        self.nested_levels = nested_levels
        self.cut_cap_per_terminal = cut_cap_per_terminal
        self.cut_cap_per_round = cut_cap_per_round
        self.max_separation_iterations = max_separation_iterations
        self.max_stalled_iterations = max_stalled_iterations
        self.stall_improvement_tolerance = stall_improvement_tolerance
        self.oracle = SplitGraphSetPairOracle(
            (ROOT,) + self.instance.facilities,
            ((u, v, index) for index, (u, v) in enumerate(problem.edges)),
            backend=separator_backend,
            cut_mode=cut_mode,
            nested_levels=nested_levels,
        )
        # Evidence: raw set pairs found in any round, recompiled on every build.
        self.setpair_cache: list[SetPair] = []
        self.setpair_seen: set[SetPair] = set()
        # Terminal singleton set pairs of the initial master.
        self.terminal_pairs = problem.terminal_singleton_pairs()
        self.fixed: frozenset[int] = frozenset()
        self.model: cplex.Cplex | None = None
        self.local_of_global: dict[int, int] = {}
        self.active_edges: tuple[int, ...] = ()
        self.row_signatures: set[tuple] = set()
        self.trace: list[dict] = []
        self.total_rows_added = 0
        self.witnesses_added = 0
        self.integer_test_calls = 0
        self.integer_test_seconds = 0.0

    # -- master construction ------------------------------------------------

    def _build_residual_master(self, fixed_ids: frozenset[int]) -> None:
        """Create the residual master for ``E_2 \\ J`` and replay the evidence."""
        self.fixed = frozenset(fixed_ids)
        self.active_edges = tuple(
            index for index in range(len(self.problem.edges))
            if index not in self.fixed
        )
        self.model = _new_model(1.0, strict_feasibility=True)
        self.model.variables.add(
            names=[f"x_{index}" for index in self.active_edges],
            obj=[self.problem.edge_costs[index] for index in self.active_edges],
            lb=[0.0] * len(self.active_edges),
            ub=[1.0] * len(self.active_edges),
        )
        self.local_of_global = {
            global_id: local for local, global_id in enumerate(self.active_edges)
        }
        self.row_signatures = set()
        for pair in self.terminal_pairs:
            self._compile_and_add(pair)
        for pair in self.setpair_cache:
            self._compile_and_add(pair)

    def _compile_and_add(self, pair: SetPair) -> bool:
        """Compile one set pair under the current ``J`` and add its row."""
        assert self.model is not None
        row = self.problem.compile(pair, self.fixed)
        if row.residual_rhs <= 0:
            # Automatically satisfied by the nonnegative variables.
            return False
        if not row.residual_ids:
            raise RuntimeError(
                "Set-pair row has a positive residual right-hand side but no "
                "available edge, so the residual LP is infeasible: "
                f"U1={sorted(row.pair.U1)[:4]}..., U2={sorted(row.pair.U2)[:4]}..., "
                f"Z={sorted(row.boundary)}, g={row.original_rhs}, J={len(self.fixed)}"
            )
        signature = (tuple(sorted(row.residual_ids)), row.residual_rhs)
        if signature in self.row_signatures:
            return False
        coefficients = {
            self.local_of_global[global_id]: 1.0 for global_id in row.residual_ids
        }
        _add_rows(self.model, [(coefficients, "G", float(row.residual_rhs))])
        self.row_signatures.add(signature)
        self.total_rows_added += 1
        return True

    def _cache_pair(self, pair: SetPair) -> None:
        if pair not in self.setpair_seen:
            self.setpair_seen.add(pair)
            self.setpair_cache.append(pair)
            self.witnesses_added += 1

    # -- separation ---------------------------------------------------------

    def _compile_sides(
        self, sides: list[set[int]], target: str, design: list[float]
    ) -> list[tuple[SetPair, SetPairRow, float]]:
        """Decode sides into set pairs, compile and keep the violated rows."""
        rows = []
        seen_pairs: set[SetPair] = set()
        for side in sides:
            pair = self.oracle.decode_source_side(side, target)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            row = self.problem.compile(pair, self.fixed)
            if row.residual_rhs <= 0:
                continue
            if not row.residual_ids:
                raise RuntimeError(
                    "A sub-threshold split cut decoded into a set-pair row with "
                    "no available edge; the residual LP is infeasible"
                )
            violation = row.residual_rhs - sum(
                design[global_id] for global_id in row.residual_ids)
            if violation > SEPARATION_TOL:
                rows.append((pair, row, violation))
        return rows

    def separate(
        self, design: list[float]
    ) -> tuple[list[tuple[SetPair, SetPairRow, float]], float, bool, int, int]:
        """Complete set-pair separation at the current ``J``.

        Returns the violated rows together with their set-pair evidence, the
        minimum cut value over all terminals, a completeness flag and the counts
        of max-flow calls and exact fallbacks.
        """
        candidates: list[tuple[SetPair, SetPairRow, float]] = []
        minimum_cut_value = math.inf
        complete = True
        flow_calls = 0
        fallback_calls = 0
        for terminal in self.problem.terminals:
            requirement = self.problem.requirements[terminal]
            if requirement <= 0:
                continue
            threshold = float(requirement)
            calls_before = self.oracle.calls
            flow_value, sides, certified = self.oracle.cut_sides(
                design, terminal, threshold)
            flow_calls += self.oracle.calls - calls_before
            minimum_cut_value = min(minimum_cut_value, flow_value)
            terminal_rows: list[tuple[SetPair, SetPairRow, float]] = []
            if not certified:
                terminal_rows = self._compile_sides(sides, terminal, design)
                if not terminal_rows and self.oracle.backend == "scipy":
                    # Integer capacities are rounded DOWN: a sub-threshold
                    # scaled cut that is not violated in floating point can
                    # conceal a different violated cut, so recheck this
                    # ambiguous terminal on the exact float Dinic.
                    fallback_calls += 1
                    calls_before = self.oracle.calls
                    flow_value, sides, certified = self.oracle.exact_cut_sides(
                        design, terminal, threshold)
                    flow_calls += self.oracle.calls - calls_before
                    minimum_cut_value = min(minimum_cut_value, flow_value)
                    if not certified:
                        terminal_rows = self._compile_sides(
                            sides, terminal, design)
            if terminal_rows:
                complete = False
                if 0 < self.cut_cap_per_terminal < len(terminal_rows):
                    terminal_rows.sort(key=lambda item: (-item[2], len(item[1].residual_ids)))
                    terminal_rows = terminal_rows[: self.cut_cap_per_terminal]
                candidates.extend(terminal_rows)
        return candidates, minimum_cut_value, complete, flow_calls, fallback_calls

    # -- integer feasibility test ------------------------------------------

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

    # -- solve one residual master -----------------------------------------

    def _full_design(self, values: list[float]) -> list[float]:
        """Restore global design values: fixed edges are one, others solved."""
        design = [1.0 if index in self.fixed else 0.0
                  for index in range(len(self.problem.edges))]
        for global_id, local in self.local_of_global.items():
            design[global_id] = values[local]
        return design

    def solve_residual(
        self, fixed: set[int], deadline: float
    ) -> dict:
        """Solve ``L(J)`` by cut generation and return an optimal basic point.

        The residual master is rebuilt from scratch for every ``J``, so this is
        also the first call that creates ``self.model``.
        """
        rebuild_start = time.perf_counter()
        self._build_residual_master(frozenset(fixed))
        rebuild_seconds = time.perf_counter() - rebuild_start
        start = time.perf_counter()
        solver_seconds = 0.0
        separation_seconds = 0.0
        rows_added = 0
        witnesses = 0
        flow_calls = 0
        fallback_calls = 0
        max_violation = 0.0
        iteration = 0
        best_objective = -math.inf
        stalled_iterations = 0
        while True:
            iteration += 1
            self.model.parameters.timelimit.set(_remaining(deadline))
            solve_start = time.perf_counter()
            self.model.solve()
            solver_seconds += time.perf_counter() - solve_start
            status = self.model.solution.get_status_string()
            if self.model.solution.get_status() != self.model.solution.status.optimal:
                raise RuntimeError(f"Set-pair master is not optimal: {status}")
            values = self.model.solution.get_values()
            objective = self.model.solution.get_objective_value()
            # Cut generation raises the lower bound; an iteration that does not
            # move it is a "stall" and the caller may cap a long stall tail.
            if objective > best_objective + self.stall_improvement_tolerance:
                best_objective = objective
                stalled_iterations = 0
            else:
                stalled_iterations += 1
            solution_type = self.model.solution.get_solution_type()
            design = self._full_design(values)
            separation_start = time.perf_counter()
            candidates, minimum_cut_value, complete, calls, fallbacks = (
                self.separate(design))
            separation_seconds += time.perf_counter() - separation_start
            flow_calls += calls
            fallback_calls += fallbacks
            if self.cut_cap_per_round > 0 and len(candidates) > self.cut_cap_per_round:
                candidates = sorted(
                    candidates, key=lambda item: (-item[2], len(item[1].residual_ids))
                )[: self.cut_cap_per_round]
            new_rows = 0
            # Rows already present before this separation round must not come
            # back as violated, which would mean the master does not really
            # satisfy its own rows.  Duplicates *within* one round are expected
            # (the same set pair can be found from several terminals) and are
            # silently deduplicated by the row signature.
            known_before = set(self.row_signatures)
            for pair, row, violation in candidates:
                max_violation = max(max_violation, violation)
                signature = (tuple(sorted(row.residual_ids)), row.residual_rhs)
                if signature in known_before:
                    raise RuntimeError(
                        "The separator found a violated set-pair row that is "
                        "already in the master; check the LP and separation "
                        "feasibility tolerances"
                    )
                if self._compile_and_add(pair):
                    new_rows += 1
                    self._cache_pair(pair)
                    witnesses += 1
            rows_added += new_rows
            self.trace.append({
                "status": status,
                "objective": objective,
                "solver_seconds": solver_seconds,
                "separation_seconds": separation_seconds,
                "compiled_rows_added": new_rows,
                "total_rows": self.total_rows_added,
                "max_flow_calls": calls,
                "minimum_cut_value": minimum_cut_value,
                "variables": self.model.variables.get_num(),
                "constraints": self.model.linear_constraints.get_num(),
                "active_edge_count": len(self.active_edges),
                "fixed_edge_count": len(self.fixed),
                "max_violation": max_violation,
                "max_unfixed_x": max(
                    (values[local] for local in self.local_of_global.values()),
                    default=0.0,
                ),
            })
            if complete:
                separation_complete = True
            elif 0 < self.max_separation_iterations <= iteration:
                # Truncated row generation: the partially separated residual LP
                # is still a valid lower bound, but the separation tail is cut
                # short so the caller can round the current basic point instead
                # of spinning on rows that barely move the objective.
                separation_complete = False
            elif 0 < self.max_stalled_iterations <= stalled_iterations:
                # The bound has not moved for several iterations: stop the
                # crawl.  Instances that keep improving never reach this branch,
                # so only the long separation tails are affected.
                separation_complete = False
            else:
                separation_complete = None
            if separation_complete is not None:
                if solution_type != self.model.solution.type.basic:
                    raise RuntimeError("Set-pair solution is not a basic solution")
                return {
                    "status": status,
                    "objective": objective,
                    "x": dict(zip(self.problem.edges, design)),
                    "solver_seconds": solver_seconds,
                    "separation_seconds": separation_seconds,
                    "rebuild_seconds": rebuild_seconds,
                    "max_flow_calls": flow_calls,
                    "scaling_fallback_calls": fallback_calls,
                    "compiled_rows_added": rows_added,
                    "setpair_witnesses_added": witnesses,
                    "max_violation": max_violation,
                    "optimal_basic": True,
                    "separation_complete": separation_complete,
                    "separation_truncated": not separation_complete,
                    "separation_iterations": iteration,
                    "stalled_iterations": stalled_iterations,
                    "fixed_edge_count": len(self.fixed),
                    "active_edge_count": len(self.active_edges),
                    "seconds": time.perf_counter() - start,
                }
            if new_rows == 0:
                raise RuntimeError(
                    "The set-pair separator reported an incomplete separation "
                    "but no new row could be compiled; check the residual "
                    "right-hand sides and the column mapping"
                )


def iterative_round_stage_two(
    instance: TwoRadiusInstance,
    stage_one: dict,
    deadline: float,
    separator_backend: str = "scipy",
    cut_mode: str = "double",
    nested_levels: int = 4,
    cut_cap_per_terminal: int = 0,
    cut_cap_per_round: int = 0,
    max_separation_iterations: int = 0,
    max_stalled_iterations: int = DEFAULT_MAX_STALLED_ITERATIONS,
) -> dict:
    start = time.perf_counter()
    cover = set(stage_one["cover"])
    problem = RootedSetPairProblem(
        instance, cover, stage_one["T1"], stage_one["T2"])
    workspace = SetPairWorkspace(
        problem,
        separator_backend,
        cut_mode=cut_mode,
        nested_levels=nested_levels,
        cut_cap_per_terminal=cut_cap_per_terminal,
        cut_cap_per_round=cut_cap_per_round,
        max_separation_iterations=max_separation_iterations,
        max_stalled_iterations=max_stalled_iterations,
    )
    full_test_before = workspace.integer_test_seconds
    if not workspace.is_feasible(set(instance.backbone_edges)):
        return {"status": "infeasible", "seconds": time.perf_counter() - start}
    full_test_seconds = workspace.integer_test_seconds - full_test_before
    fixed: set[int] = set()
    rounds = []
    initial_lp = None
    initial_positive_support_edges = None
    rounding_test_seconds = 0.0
    edges_by_index = problem.edges
    while not workspace.is_feasible({edges_by_index[index] for index in fixed}):
        rounding_test_seconds = (
            workspace.integer_test_seconds - full_test_seconds)
        result = workspace.solve_residual(fixed, deadline)
        if initial_lp is None:
            initial_lp = result["objective"]
            initial_positive_support_edges = sum(
                value > TOL for value in result["x"].values()
            )
        eligible = [
            index for index in range(len(edges_by_index))
            if index not in fixed
            and result["x"][edges_by_index[index]] >= ROUNDING_THRESHOLD
        ]
        if not eligible:
            raise RuntimeError(
                "The optimal basic set-pair solution has no x_e >= 1/2, so the "
                "rounding interface of the residual LP is violated"
            )
        fixed.update(eligible)
        rounds.append({
            "round": len(rounds) + 1,
            "residual_objective": result["objective"],
            "rounded_edges": [list(edges_by_index[index]) for index in eligible],
            "num_rounded_edges": len(eligible),
            "fixed_edges_after": len(fixed),
            "active_edge_count": result["active_edge_count"],
            "solver_seconds": result["solver_seconds"],
            "separation_seconds": result["separation_seconds"],
            "rebuild_seconds": result["rebuild_seconds"],
            "max_flow_calls": result["max_flow_calls"],
            "scaling_fallback_calls": result["scaling_fallback_calls"],
            "compiled_rows_added": result["compiled_rows_added"],
            "setpair_witnesses_added": result["setpair_witnesses_added"],
            "max_violation": result["max_violation"],
            "optimal_basic": result["optimal_basic"],
            "separation_complete": result["separation_complete"],
            "separation_truncated": result["separation_truncated"],
            "separation_iterations": result["separation_iterations"],
            "stalled_iterations": result["stalled_iterations"],
        })
        _remaining(deadline)

    before_pruning = len(fixed)
    rounding_test_seconds = workspace.integer_test_seconds - full_test_seconds
    pruning_start = time.perf_counter()
    pruning_test_calls_before = workspace.integer_test_calls
    pruning_test_seconds_before = workspace.integer_test_seconds
    fixed_edges = sorted(edges_by_index[index] for index in fixed)
    for edge in fixed_edges:
        _remaining(deadline)
        candidate = fixed - {problem.edge_index[edge]}
        if workspace.is_feasible({edges_by_index[index] for index in candidate}):
            fixed = candidate
    pruning_seconds = time.perf_counter() - pruning_start
    pruning_test_calls = (
        workspace.integer_test_calls - pruning_test_calls_before)
    pruning_test_seconds = (
        workspace.integer_test_seconds - pruning_test_seconds_before)
    support_edges = sorted(edges_by_index[index] for index in fixed)
    endpoints = {
        vertex for edge in support_edges for vertex in edge if vertex != ROOT
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
        "formulation": FORMULATION,
        "formulation_version": FORMULATION_VERSION,
        "residual_mode": RESIDUAL_MODE,
        "separation_backend": f"split_graph_{separator_backend}",
        "separator_backend": workspace.separator_backend,
        "cut_mode": workspace.cut_mode,
        "nested_levels": workspace.nested_levels,
        "cut_cap_per_terminal": workspace.cut_cap_per_terminal,
        "cut_cap_per_round": workspace.cut_cap_per_round,
        "max_separation_iterations": workspace.max_separation_iterations,
        "max_stalled_iterations": workspace.max_stalled_iterations,
        "separation_truncated": any(
            round_data["separation_truncated"] for round_data in rounds),
        "terminals_total": len(problem.terminals),
        "terminal_requirements": dict(problem.requirements),
        "global_edge_count": len(problem.edges),
        "secondary_rows_added": workspace.oracle.secondary_rows_added,
        "nested_rows_added": workspace.oracle.nested_rows_added,
        "nested_sides_tried": workspace.oracle.nested_sides_tried,
        "stage_two_over_edge_lp": (
            stage_two_cost / initial_lp
            if initial_lp is not None and initial_lp > TOL else 0.0
        ),
        "support_edges": [list(edge) for edge in support_edges],
        "initial_edge_lp": initial_lp,
        "initial_positive_support_edges": initial_positive_support_edges,
        "rounds": rounds,
        "num_rounds": len(rounds),
        "master_lp_solves": len(workspace.trace),
        "edges_before_pruning": before_pruning,
        "edges_after_pruning": len(fixed),
        "edges_pruned": before_pruning - len(fixed),
        "cuts_retained": workspace.total_rows_added,
        "setpair_witnesses_added": workspace.witnesses_added,
        "max_violation": max(
            (round_data["max_violation"] for round_data in rounds), default=0.0),
        "master_trace": workspace.trace,
        "lp_solver_seconds": sum(
            round_data["solver_seconds"] for round_data in rounds),
        "lp_separation_seconds": sum(
            round_data["separation_seconds"] for round_data in rounds),
        "rebuild_seconds": sum(
            round_data["rebuild_seconds"] for round_data in rounds),
        "max_flow_calls": sum(
            round_data["max_flow_calls"] for round_data in rounds),
        "scaling_fallback_calls": sum(
            round_data["scaling_fallback_calls"] for round_data in rounds),
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
