"""Tests for the rooted set-pair model, the split-graph separator and the
two-stage algorithm.

The suite follows the acceptance plan of the migration note: exhaustive
set-pair LPs on small graphs, the residual transformation identity, the
separator and double-cut evidence, the per-round rounding cost and an
independent integer optimum.
"""

from __future__ import annotations

import itertools
import random
import time

import numpy as np
import pytest
from scipy.optimize import linprog

from algorithm import (
    DEFAULT_MAX_STALLED_ITERATIONS,
    FORMULATION,
    FORMULATION_VERSION,
    RESIDUAL_MODE,
    RootedSetPairProblem,
    SetPair,
    SetPairWorkspace,
    SplitGraphSetPairOracle,
    _CutOracle,
    iterative_round_stage_two,
    stage_one_cover,
    verify_design,
)
from instance import ROOT, TOL, TwoRadiusInstance

# One shared deadline per test: every solver call inside the tested function
# receives it, and the small instances never come close to it.
TIME_LIMIT = 60.0


def tiny_instance(**overrides) -> TwoRadiusInstance:
    data = dict(
        variant="selectable",
        facility_positions={"d0": (0.0, 0.0), "d1": (2.0, 0.0), "f0": (1.0, 0.0)},
        demand_positions={"p0": (1.0, 0.0)},
        depots=("d0", "d1"),
        stations=("f0",),
        requirements={"p0": 2},
        costs={"d0": 3, "d1": 4, "f0": 2},
        flight_radius=2.0,
        service_radius=1.0,
        seed=4,
    )
    data.update(overrides)
    return TwoRadiusInstance(**data)


def model_instance(**overrides) -> TwoRadiusInstance:
    """Four facilities: a zero-cost station, a depot, and mixed requirements."""
    data = dict(
        variant="selectable",
        facility_positions={
            "d0": (0.0, 0.0), "d1": (2.0, 0.0),
            "f0": (1.0, 0.0), "f1": (3.0, 0.0),
        },
        demand_positions={"p0": (1.0, 0.0), "p1": (3.0, 0.0)},
        depots=("d0", "d1"),
        stations=("f0", "f1"),
        requirements={"p0": 2, "p1": 1},
        costs={"d0": 3, "d1": 4, "f0": 0, "f1": 2},
        flight_radius=2.0,
        service_radius=1.0,
        seed=11,
    )
    data.update(overrides)
    return TwoRadiusInstance(**data)


def model_problem(instance: TwoRadiusInstance | None = None) -> RootedSetPairProblem:
    """The rooted set-pair problem of ``model_instance``.

    ``f0`` (requirement 2) and ``d1`` (requirement 1) are the terminals, so the
    model exercises mixed requirements, a root-adjacent terminal, a zero-cost
    facility and two paying non-terminals.
    """
    return RootedSetPairProblem(
        instance if instance is not None else model_instance(),
        cover={"f0", "d1"},
        terminals_one={"d1"},
        terminals_two={"f0"},
    )


def enumerate_set_pairs(problem: RootedSetPairProblem) -> list[SetPair]:
    """Every rooted set pair: the root is in ``U1``, the rest is labeled.

    Rooted rows with the root in ``U2`` are the same constraint with the sides
    exchanged, so this enumeration already covers every non-redundant row.
    """
    others = sorted(problem.universe - {problem.root})
    pairs = []
    for labels in itertools.product(("U1", "U2", "Z"), repeat=len(others)):
        U1 = {problem.root}
        U2: set[str] = set()
        for vertex, label in zip(others, labels):
            if label == "U1":
                U1.add(vertex)
            elif label == "U2":
                U2.add(vertex)
        if U2:
            pairs.append(SetPair(frozenset(U1), frozenset(U2)))
    return pairs


def reference_residual_lp(
    problem: RootedSetPairProblem,
    pairs: list[SetPair],
    fixed_ids: frozenset[int] = frozenset(),
) -> tuple[float | None, list[float] | None]:
    """Full explicit set-pair LP: fixed edges are one, their cost is zero."""
    edges = problem.edges
    rows = []
    for pair in pairs:
        row = problem.compile(pair, fixed_ids)
        if row.original_rhs > 0:
            rows.append(row)
    coefficients = np.zeros((len(rows), len(edges)))
    for position, row in enumerate(rows):
        for edge_id in row.crossing_ids:
            coefficients[position, edge_id] = 1.0
    objective = [
        0.0 if index in fixed_ids else problem.edge_costs[index]
        for index in range(len(edges))
    ]
    bounds = [
        (1.0 if index in fixed_ids else 0.0, 1.0)
        for index in range(len(edges))
    ]
    result = linprog(
        objective,
        A_ub=-coefficients,
        b_ub=[-row.original_rhs for row in rows],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None, None
    return float(result.fun), [float(value) for value in result.x]


def max_violation(
    problem: RootedSetPairProblem,
    pairs: list[SetPair],
    fixed_ids: frozenset[int],
    values: list[float],
) -> float:
    """Largest shortfall of ``values`` against every enumerated set-pair row."""
    worst = 0.0
    for pair in pairs:
        row = problem.compile(pair, fixed_ids)
        if row.original_rhs <= 0:
            continue
        slack = row.original_rhs - sum(values[index] for index in row.crossing_ids)
        worst = max(worst, slack)
    return worst


def brute_force_optimum(instance: TwoRadiusInstance) -> float:
    return min(
        sum(instance.costs[facility] for facility in subset)
        for size in range(len(instance.facilities) + 1)
        for subset in itertools.combinations(instance.facilities, size)
        if verify_design(instance, set(subset))
    )


def brute_force_augmentation(instance: TwoRadiusInstance, cover: set[str]) -> float:
    """Cheapest facility superset of ``cover`` that verifies as a design."""
    others = [v for v in instance.facilities if v not in cover]
    best = None
    for size in range(len(others) + 1):
        for extra in itertools.combinations(others, size):
            opened = cover | set(extra)
            if not verify_design(instance, opened):
                continue
            cost = sum(instance.costs[v] for v in extra)
            best = cost if best is None else min(best, cost)
    return best


def stage_two(instance: TwoRadiusInstance, **kwargs) -> tuple[dict, dict]:
    deadline = time.perf_counter() + TIME_LIMIT
    stage_one = stage_one_cover(instance, 0.2, deadline, "greedy")
    result = iterative_round_stage_two(instance, stage_one, deadline, **kwargs)
    return stage_one, result


# -- model layer: exhaustive set pairs and the residual LP ------------------


def test_compiled_row_matches_the_paper_formulas() -> None:
    problem = model_problem()
    fixed = frozenset({0, 2})
    for pair in enumerate_set_pairs(problem):
        row = problem.compile(pair, fixed)
        assert row.boundary == problem.universe - pair.U1 - pair.U2
        assert row.q == max(
            (problem.requirements.get(v, 0) for v in pair.U2), default=0)
        assert row.original_rhs == row.q - len(row.boundary)
        expected_crossing = tuple(
            index for index, (u, v) in enumerate(problem.edges)
            if (u in pair.U1 and v in pair.U2) or (v in pair.U1 and u in pair.U2)
        )
        assert row.crossing_ids == expected_crossing
        assert row.fixed_crossing_count == sum(
            index in fixed for index in expected_crossing)
        assert row.residual_ids == tuple(
            index for index in expected_crossing if index not in fixed)
        assert row.residual_rhs == row.original_rhs - row.fixed_crossing_count


def test_q_is_the_largest_requirement_of_the_separated_side() -> None:
    problem = model_problem()
    # f0 needs two root paths, d1 only one: separating both at once must use 2.
    both = SetPair(
        frozenset({ROOT}) | (problem.universe - {"f0", "d1"}),
        frozenset({"f0", "d1"}),
    )
    assert problem.compile(both, ()).q == 2
    # A pair whose sides avoid the root carries no requirement at all.
    side = sorted(problem.universe - {ROOT, "f0"})[0]
    unrooted = SetPair(frozenset({side}), frozenset({"f0"}))
    assert problem.compile(unrooted, ()).q == 0
    assert problem.compile(unrooted, ()).original_rhs <= 0


def test_exhaustive_set_pairs_match_the_separated_lp() -> None:
    problem = model_problem()
    pairs = enumerate_set_pairs(problem)
    reference, _ = reference_residual_lp(problem, pairs)
    assert reference is not None

    workspace = SetPairWorkspace(problem, "scipy", cut_mode="double")
    result = workspace.solve_residual(set(), time.perf_counter() + TIME_LIMIT)

    assert result["optimal_basic"] is True
    assert result["separation_complete"] is True
    assert result["objective"] == pytest.approx(reference, rel=1e-7, abs=1e-7)
    values = [result["x"][edge] for edge in problem.edges]
    assert max_violation(problem, pairs, frozenset(), values) <= 1e-7


@pytest.mark.parametrize("cut_mode", ["single", "double", "nested"])
def test_every_cut_mode_reaches_the_same_lp_optimum(cut_mode: str) -> None:
    problem = model_problem()
    pairs = enumerate_set_pairs(problem)
    reference, _ = reference_residual_lp(problem, pairs)
    workspace = SetPairWorkspace(problem, "scipy", cut_mode=cut_mode)
    result = workspace.solve_residual(set(), time.perf_counter() + TIME_LIMIT)
    assert result["objective"] == pytest.approx(reference, rel=1e-7, abs=1e-7)
    values = [result["x"][edge] for edge in problem.edges]
    assert max_violation(problem, pairs, frozenset(), values) <= 1e-7


def test_residual_lp_matches_the_fixed_edge_model() -> None:
    """Level 2: the residual LP equals the full model with ``J`` fixed to one."""
    problem = model_problem()
    pairs = enumerate_set_pairs(problem)
    rng = random.Random(20261025)
    all_ids = list(range(len(problem.edges)))
    fixed_sets = [
        set(),
        set(rng.sample(all_ids, 1)),
        set(rng.sample(all_ids, 2)),
        # A full terminal boundary: f0 has degree two, so this is still feasible.
        {index for index, (u, v) in enumerate(problem.edges)
         if "f0" in (u, v)},
        set(all_ids),
    ]
    for fixed in fixed_sets:
        frozen = frozenset(fixed)
        reference, _ = reference_residual_lp(problem, pairs, frozen)
        workspace = SetPairWorkspace(problem, "scipy", cut_mode="double")
        if reference is None:
            with pytest.raises(RuntimeError):
                workspace.solve_residual(fixed, time.perf_counter() + TIME_LIMIT)
            continue
        result = workspace.solve_residual(fixed, time.perf_counter() + TIME_LIMIT)
        assert result["objective"] == pytest.approx(
            reference, rel=1e-7, abs=1e-7), sorted(fixed)
        assert result["fixed_edge_count"] == len(fixed)
        assert result["active_edge_count"] == len(problem.edges) - len(fixed)


def test_fixed_contribution_identity_holds_for_random_solutions() -> None:
    problem = model_problem()
    pairs = enumerate_set_pairs(problem)
    rng = random.Random(7)
    fixed = frozenset(rng.sample(range(len(problem.edges)), 2))
    values = [1.0 if index in fixed else rng.random() for index in range(len(problem.edges))]
    for pair in pairs:
        row = problem.compile(pair, fixed)
        full_sum = sum(values[index] for index in row.crossing_ids)
        active_sum = sum(values[index] for index in row.residual_ids)
        assert full_sum - row.original_rhs == pytest.approx(
            active_sum - row.residual_rhs)


def test_positive_residual_rhs_without_an_edge_is_infeasible() -> None:
    """Counter-example 1: an unsatisfiable row must fail loudly, not be kept."""
    instance = TwoRadiusInstance(
        variant="selectable",
        facility_positions={
            "d0": (0.0, 0.0), "d1": (5.0, 0.0), "f0": (1.0, 0.0),
        },
        demand_positions={"p0": (1.0, 0.0)},
        depots=("d0", "d1"),
        stations=("f0",),
        requirements={"p0": 2},
        costs={"d0": 1, "d1": 1, "f0": 1},
        flight_radius=2.0,
        service_radius=1.0,
        seed=3,
    )
    problem = RootedSetPairProblem(
        instance, cover={"f0"}, terminals_one=(), terminals_two={"f0"})
    crossing = {
        index for index, (u, v) in enumerate(problem.edges) if "f0" in (u, v)
    }
    assert len(crossing) == 1
    row = problem.compile(problem.terminal_singleton_pairs()[0], crossing)
    assert row.original_rhs == 2
    assert row.residual_rhs == 1
    assert row.residual_ids == ()

    workspace = SetPairWorkspace(problem, "scipy", cut_mode="double")
    with pytest.raises(RuntimeError):
        workspace.solve_residual(set(crossing), time.perf_counter() + TIME_LIMIT)


# -- separator -------------------------------------------------------------


def test_separator_evidence_is_valid_and_complete() -> None:
    """Level 3: the returned set pairs are valid and completeness is exact."""
    problem = model_problem()
    pairs = enumerate_set_pairs(problem)
    rng = random.Random(5)
    for trial in range(6):
        values = [
            0.0 if rng.random() < 0.3 else rng.choice([0.25, 0.5, 0.75, 1.0])
            for _ in problem.edges
        ]
        workspace = SetPairWorkspace(problem, "scipy", cut_mode="double")
        candidates, _, complete, _, _ = workspace.separate(values)
        for pair, row, violation in candidates:
            assert pair.U1 & pair.U2 == frozenset()
            assert pair.U1 and pair.U2
            assert ROOT in pair.U1
            assert (pair.U1 | pair.U2) <= problem.universe
            assert row.boundary == problem.universe - pair.U1 - pair.U2
            separated = [
                terminal for terminal in problem.terminals
                if terminal in pair.U2 and problem.requirements[terminal] > 0
            ]
            assert separated
            assert row.q == max(
                problem.requirements[terminal] for terminal in separated)
            assert violation == pytest.approx(
                row.residual_rhs
                - sum(values[index] for index in row.residual_ids))
            assert violation > TOL
        worst = max_violation(problem, pairs, frozenset(), values)
        if complete:
            assert worst <= 1e-7, f"trial {trial} reported completeness"
        else:
            assert worst > 1e-9


def test_decoded_source_side_is_a_canonical_set_pair() -> None:
    oracle = SplitGraphSetPairOracle(
        (ROOT,) + model_problem().nodes[1:],
        [(u, v, index) for index, (u, v) in enumerate(model_problem().edges)],
        backend="python",
        cut_mode="double",
    )
    values = [0.0] * len(model_problem().edges)
    _, sides, certified = oracle.cut_sides(values, "f0", 2.0)
    assert certified is False
    for side in sides:
        pair = oracle.decode_source_side(side, "f0")
        assert ROOT in pair.U1
        assert "f0" not in pair.U1
        assert len(frozenset(oracle.nodes) - pair.U1 - pair.U2) <= 1


def test_single_and_double_modes_agree_on_feasibility() -> None:
    problem = model_problem()
    rng = random.Random(99)
    for _ in range(4):
        values = [rng.choice([0.0, 0.5, 1.0]) for _ in problem.edges]
        single = SetPairWorkspace(problem, "scipy", cut_mode="single")
        double = SetPairWorkspace(problem, "scipy", cut_mode="double")
        _, _, single_complete, _, _ = single.separate(values)
        _, _, double_complete, _, _ = double.separate(values)
        assert single_complete == double_complete
        assert single.oracle.secondary_rows_added == 0


# -- integer feasibility test ---------------------------------------------


def test_workspace_integer_test_is_root_biconnectivity() -> None:
    instance = tiny_instance()
    problem = RootedSetPairProblem(
        instance, {"f0"}, set(), {"f0"})
    workspace = SetPairWorkspace(problem, "scipy", cut_mode="double")
    assert workspace.is_feasible({
        ("rho", "d0"), ("rho", "d1"), ("d0", "f0"), ("d1", "f0"),
    })
    assert not workspace.is_feasible({("rho", "d0"), ("d0", "f0")})


def test_cut_oracle_finds_known_minimum_cut() -> None:
    edges = (
        ("s", "a", 0),
        ("a", "t", 1),
        ("s", "b", 2),
        ("b", "t", 3),
    )
    values = [0.4, 0.7, 0.6, 0.2]
    oracle = _CutOracle(("s", "a", "b", "t"), edges)
    cut_value, row = oracle.minimum_cut(values, "s", "t", 2.0)
    assert cut_value == pytest.approx(0.6)
    assert row is not None
    assert sum(values[index] for index in row) == pytest.approx(cut_value)


def test_verify_design_needs_two_internally_disjoint_paths() -> None:
    instance = tiny_instance()
    assert verify_design(instance, {"d0", "d1"})
    assert verify_design(instance, {"d0", "d1", "f0"})
    assert not verify_design(instance, {"f0"})
    assert not verify_design(instance, {"d0", "f0"})


# -- full stage two --------------------------------------------------------


def test_stage_two_returns_a_verified_feasible_design() -> None:
    instance = tiny_instance()
    optimum = brute_force_optimum(instance)
    assert optimum == pytest.approx(7.0)

    stage_one, result = stage_two(instance, cut_mode="double")
    assert result["status"] == "feasible"
    assert result["verified"] is True
    assert result["cost"] >= optimum
    assert verify_design(instance, set(result["selected"]))
    assert result["master_lp_solves"] >= 1
    assert result["formulation"] == FORMULATION
    assert result["formulation_version"] == FORMULATION_VERSION
    assert result["residual_mode"] == RESIDUAL_MODE
    assert result["separation_backend"] == "split_graph_scipy"
    assert result["cost"] == pytest.approx(
        stage_one["cost"] + result["stage_two_added_cost"])


def test_default_separation_tail_guard_is_enabled() -> None:
    """The shipped default guards the flat row-generation tail."""
    assert DEFAULT_MAX_STALLED_ITERATIONS == 3
    _, result = stage_two(tiny_instance(), cut_mode="double")
    assert result["max_stalled_iterations"] == DEFAULT_MAX_STALLED_ITERATIONS


def test_stalled_iteration_guard_keeps_the_rounded_cost() -> None:
    """The guard truncates the tail without paying for it in design quality."""
    instance = model_instance()
    _, unguarded = stage_two(
        instance, cut_mode="double", max_stalled_iterations=0)
    _, guarded = stage_two(
        instance, cut_mode="double", max_stalled_iterations=1)
    assert guarded["status"] == "feasible"
    assert guarded["verified"] is True
    assert guarded["cost"] == unguarded["cost"]
    assert guarded["max_stalled_iterations"] == 1
    assert guarded["master_lp_solves"] <= unguarded["master_lp_solves"]


def test_stage_two_rounds_replay_the_logged_residual_lp() -> None:
    """Level 4: monotone ``J``, threshold rounding and replayed ``L(J)``."""
    instance = model_instance()
    stage_one, result = stage_two(instance, cut_mode="double")
    assert result["status"] == "feasible"
    problem = RootedSetPairProblem(
        instance, set(stage_one["cover"]), stage_one["T1"], stage_one["T2"])
    edge_index = problem.edge_index
    deadline = time.perf_counter() + TIME_LIMIT
    fixed: set[int] = set()
    previous_objective = None
    for round_data in result["rounds"]:
        delta = [edge_index[tuple(edge)] for edge in round_data["rounded_edges"]]
        assert round_data["optimal_basic"] is True
        assert round_data["separation_complete"] is True
        assert delta
        assert all(index not in fixed for index in delta)
        assert round_data["fixed_edges_after"] == len(fixed) + len(delta)
        # Rounding never strengthens its own cost bound: the residual LP value
        # can only fall as edges are fixed, and every rounded edge is paid for
        # at most twice by that drop.
        replay = SetPairWorkspace(problem, "scipy", cut_mode="double")
        next_value = replay.solve_residual(set(fixed) | set(delta), deadline)
        cost = sum(problem.edge_costs[index] for index in delta)
        drop = round_data["residual_objective"] - next_value["objective"]
        assert cost <= 2.0 * drop + 1e-7
        if previous_objective is not None:
            assert round_data["residual_objective"] <= previous_objective + 1e-7
        previous_objective = round_data["residual_objective"]
        fixed = set(fixed) | set(delta)

    # The paper's global bound of Lemma "Stage 2 edge rounding".
    clamped = sum(problem.edge_costs[index] for index in fixed)
    assert clamped <= 2.0 * result["initial_edge_lp"] + 1e-7
    assert result["stage_two_over_edge_lp"] <= 2.0 + 1e-7
    # Pruning never reopens a facility that stage one had already paid for.
    assert set(result["selected"]) >= set(stage_one["cover"])


@pytest.mark.parametrize("variant", ["selectable", "preinstalled"])
def test_stage_two_respects_the_facility_augmentation_bound(
    variant: str,
) -> None:
    """Level 5: compare the added facilities with the enumerated optimum."""
    overrides = {"variant": variant}
    if variant == "preinstalled":
        overrides["costs"] = {"d0": 0, "d1": 0, "f0": 0, "f1": 2}
    instance = model_instance(**overrides)
    stage_one, result = stage_two(instance, cut_mode="double")
    cover = set(stage_one["cover"])
    optimum_augmentation = brute_force_augmentation(instance, cover)
    assert result["status"] == "feasible"
    assert result["verified"] is True
    assert result["stage_two_added_cost"] <= 6.0 * optimum_augmentation + 1e-7
    if variant == "preinstalled":
        assert result["stage_two_added_cost"] <= 5.0 * optimum_augmentation + 1e-7
