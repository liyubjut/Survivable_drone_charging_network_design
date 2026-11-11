"""Exact and relaxation baselines solved with CPLEX.

``solve_flow_benchmark`` runs the compact node-capacitated multi-commodity
FLOW-IP and captures the root relaxation from its own node log, so no separate
FLOW-LP solve is needed.  ``solve_flow_lp_relaxation`` solves the LP relaxation
of the identical model when the IP cannot be closed within its time limit, and
``solve_node_benchmark`` provides the cut-based node formulation used by the
unit tests.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time

import cplex

from algorithm import _add_rows, _design_cut_rows, _new_model, _remaining, verify_design
from instance import FLOW_IP_TOO_LARGE_STATUS, MAX_FLOW_IP_VARIABLES, MIP_NODE_LOG_INTERVAL, ROOT, ROOT_GAP_TOL, ROOT_VALUE_SENTINEL, TOL, TwoRadiusInstance


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
    """Capture the CPLEX log by redirecting both the Python and the OS streams.

    ``Cplex.__init__`` binds its log channel to the ``sys.stdout`` object that is
    current when the model is created, and on this CPLEX build the node log also
    reaches the process stdout descriptor.  A host that replaces ``sys.stdout``
    (a test runner, a notebook) therefore swallows the whole log unless the
    Python level is redirected too, so the model has to be built inside this
    context manager and both levels are redirected for the duration of the
    solve.  A file keeps the captured text out of memory until the solve is
    finished.
    """

    def __init__(self) -> None:
        self.text = ""
        self._path: str | None = None
        self._stream = None
        self._saved: tuple[int, int] | None = None
        self._saved_streams: tuple = (None, None)

    def __enter__(self) -> "_CapturedSolverLog":
        sys.stdout.flush()
        sys.stderr.flush()
        handle, path = tempfile.mkstemp(prefix="orl_cplex_log_", suffix=".txt")
        os.close(handle)
        self._path = path
        self._stream = open(path, "w+", encoding="utf-8", errors="replace")
        self._saved_streams = (sys.stdout, sys.stderr)
        sys.stdout = self._stream
        sys.stderr = self._stream
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
        sys.stdout, sys.stderr = self._saved_streams
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
    # The model is built inside the capture because CPLEX binds its log channel
    # to the ``sys.stdout`` object that is current at construction time (see
    # _CapturedSolverLog); the node log then lands in the captured file whether
    # the host writes it through the Python stream or the stdout descriptor.
    with _CapturedSolverLog() as captured:
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
