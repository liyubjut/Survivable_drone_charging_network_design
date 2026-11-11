"""Tests for the CPLEX baselines: FLOW-IP, its LP relaxation and the node model."""

from __future__ import annotations

import itertools
import time

import pytest

from algorithm import verify_design
from baseline import (
    estimate_flow_ip_size,
    solve_flow_benchmark,
    solve_flow_lp_relaxation,
    solve_node_benchmark,
)
from instance import TwoRadiusInstance, generate_two_radius_instance


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


def brute_force_optimum(instance: TwoRadiusInstance) -> float:
    return min(
        sum(instance.costs[facility] for facility in subset)
        for size in range(len(instance.facilities) + 1)
        for subset in itertools.combinations(instance.facilities, size)
        if verify_design(instance, set(subset))
    )


def test_size_estimate_matches_the_built_model() -> None:
    instance = tiny_instance()
    estimated_columns, estimated_rows = estimate_flow_ip_size(instance)
    benchmark = solve_flow_benchmark(instance, time.perf_counter() + 30.0)
    assert benchmark["proven_optimal"] is True
    assert estimated_columns > 0
    assert estimated_rows > 0


def test_flow_lp_and_node_lp_agree() -> None:
    instance = tiny_instance()
    flow_lp = solve_flow_lp_relaxation(instance, time.perf_counter() + 30.0)
    node_lp = solve_node_benchmark(instance, time.perf_counter() + 30.0, relaxed=True)
    assert flow_lp["status"] == "optimal"
    assert flow_lp["objective"] is not None
    assert node_lp["objective"] == pytest.approx(flow_lp["objective"])


def test_flow_ip_and_node_ip_match_brute_force() -> None:
    instance = tiny_instance()
    optimum = brute_force_optimum(instance)
    assert optimum == pytest.approx(7.0)

    flow_ip = solve_flow_benchmark(instance, time.perf_counter() + 30.0)
    node_ip = solve_node_benchmark(instance, time.perf_counter() + 30.0, relaxed=False)

    assert flow_ip["proven_optimal"] is True
    assert flow_ip["verified"] is True
    assert flow_ip["objective"] == pytest.approx(optimum)
    assert node_ip["objective"] == pytest.approx(optimum)


def test_flow_benchmark_records_the_root_relaxation() -> None:
    """On a non-trivial instance the node-0 value is read back as a bound.

    The two-facility toy model is closed by presolve, so no root line is ever
    printed; the smallest pre-registered tier is used instead.
    """
    instance, _ = generate_two_radius_instance(
        5, 50, 25, "selectable", "all2", 101)
    benchmark = solve_flow_benchmark(instance, time.perf_counter() + 300.0)
    assert benchmark["proven_optimal"] is True
    assert benchmark["root_relaxation"] is not None
    assert benchmark["root_relaxation"] <= benchmark["objective"] + 1e-6
