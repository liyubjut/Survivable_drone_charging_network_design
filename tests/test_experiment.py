"""Tests for the per-case metrics, the CSV rows and the aggregated table."""

from __future__ import annotations

import time

import pytest

from experiment import (
    _parse_scale,
    _requirement_mode,
    apply_quality_fields,
    build_table_summary,
    fill_lp_relaxation,
    run_case,
    strongest_lower_bound,
    summary_row,
)
from instance import (
    FLOW_IP_SKIPPED_STATUS,
    FLOW_IP_TOO_LARGE_STATUS,
    TwoRadiusInstance,
)

# The pre-registered column count of one per-case CSV row.
SUMMARY_COLUMNS = 110


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


def feasible_algorithm(cost: float) -> dict:
    return {"status": "feasible", "verified": True, "cost": cost}


def solved_record() -> dict:
    """A record whose tiny FLOW-IP proves optimality at 7.0."""
    return run_case(tiny_instance(), time_limit=60.0, cover_method="greedy")


def test_parse_scale_round_trips_a_triple() -> None:
    assert _parse_scale("5,50,25") == (5, 50, 25)
    assert _parse_scale(" 5 , 50 , 25 ") == (5, 50, 25)


@pytest.mark.parametrize("spec", ["5,50", "1,50,25", "5,0,25", "a,b,c"])
def test_parse_scale_rejects_malformed_scales(spec: str) -> None:
    with pytest.raises(ValueError):
        _parse_scale(spec)


def test_requirement_mode_recovers_the_generation_mode() -> None:
    assert _requirement_mode({"a": 1, "b": 1}) == "all1"
    assert _requirement_mode({"a": 2, "b": 2}) == "all2"
    assert _requirement_mode({"a": 1, "b": 2}) == "mixed"


def test_strongest_lower_bound_prefers_the_tighter_bound() -> None:
    assert strongest_lower_bound(
        {"best_bound": 5.0, "lp_relaxation": 6.0}
    ) == (6.0, "lp_relaxation_lower_bound")
    assert strongest_lower_bound(
        {"best_bound": 7.0, "lp_relaxation": 6.0}
    ) == (7.0, "final_ip_lower_bound")


@pytest.mark.parametrize(
    "ip",
    [
        {},
        {"best_bound": 0.0, "lp_relaxation": -3.0},
        {"best_bound": None, "lp_relaxation": None},
        {"best_bound": True},
    ],
)
def test_strongest_lower_bound_ignores_unusable_values(ip: dict) -> None:
    assert strongest_lower_bound(ip) == (None, None)


def test_lp_backfill_is_disabled_by_default() -> None:
    """The comparison must rely on the IP, never on the slow standalone LP."""
    ip = {"status": "time_limit", "proven_optimal": False, "best_bound": 12.0}
    fill_lp_relaxation(tiny_instance(), ip, 60.0, -1)
    assert ip["lp_relaxation"] is None
    assert ip["lp_relaxation_status"] == "disabled_by_request"
    assert strongest_lower_bound(ip) == (12.0, "final_ip_lower_bound")


def test_lp_backfill_keeps_an_already_recorded_relaxation() -> None:
    ip = {"status": "time_limit", "proven_optimal": False, "lp_relaxation": 9.0}
    fill_lp_relaxation(tiny_instance(), ip, 60.0, -1)
    assert ip["lp_relaxation"] == 9.0
    assert "lp_relaxation_status" not in ip


def test_quality_fields_use_the_exact_optimum_when_the_ip_proves_it() -> None:
    record = {
        "algorithm": feasible_algorithm(9.0),
        "ip": {"objective": 7.0, "verified": True, "proven_optimal": True},
    }
    apply_quality_fields(record)
    assert record["ip"]["status_class"] == "optimal"
    assert record["quality_gap_basis"] == "exact_ip_optimum"
    assert record["algorithm_over_ip"] == pytest.approx(9.0 / 7.0)
    assert record["quality_gap_percent"] == pytest.approx(
        100.0 * (9.0 / 7.0 - 1.0)
    )
    assert "algorithm_certified_gap_upper_percent" not in record


def test_quality_fields_fall_back_to_a_certified_bound() -> None:
    record = {
        "algorithm": feasible_algorithm(10.0),
        "ip": {"objective": None, "verified": None, "best_bound": 8.0},
    }
    apply_quality_fields(record)
    assert record["ip"]["status_class"] == "no_incumbent"
    assert record["quality_gap_basis"] == "final_ip_lower_bound"
    assert record["algorithm_over_ip_bound"] == pytest.approx(1.25)
    assert record["quality_gap_percent"] == pytest.approx(25.0)
    assert "algorithm_exact_gap_percent" not in record


def test_quality_fields_report_nothing_without_a_lower_bound() -> None:
    record = {
        "algorithm": feasible_algorithm(10.0),
        "ip": {"objective": None, "verified": None},
    }
    apply_quality_fields(record)
    assert "quality_gap_percent" not in record
    assert "quality_gap_basis" not in record


def test_quality_fields_reject_an_unverified_incumbent() -> None:
    record = {
        "algorithm": feasible_algorithm(10.0),
        "ip": {
            "objective": 6.0,
            "verified": False,
            "proven_optimal": False,
            "best_bound": 5.0,
        },
    }
    apply_quality_fields(record)
    assert record["ip"]["status_class"] == "invalid_incumbent"
    # The algorithm is still measured against the valid lower bound.
    assert record["quality_gap_basis"] == "final_ip_lower_bound"


def test_quality_fields_clear_stale_metrics_before_recomputing() -> None:
    record = {
        "algorithm": feasible_algorithm(10.0),
        "ip": {"objective": 7.0, "verified": True, "proven_optimal": True},
        "quality_gap_percent": 12345.0,
        "quality_gap_basis": "stale",
    }
    apply_quality_fields(record)
    assert record["quality_gap_basis"] == "exact_ip_optimum"
    assert record["quality_gap_percent"] == pytest.approx(
        100.0 * (10.0 / 7.0 - 1.0)
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (FLOW_IP_SKIPPED_STATUS, "skipped_by_request"),
        (FLOW_IP_TOO_LARGE_STATUS, "skipped_too_large"),
    ],
)
def test_quality_fields_keep_the_requested_skip_reasons(
    status: str, expected: str
) -> None:
    record = {
        "algorithm": feasible_algorithm(9.0),
        "ip": {"status": status, "objective": None},
    }
    apply_quality_fields(record)
    assert record["ip"]["status_class"] == expected


def test_small_case_quality_and_summary_row_agree() -> None:
    record = solved_record()
    assert record["algorithm"]["verified"] is True
    assert record["ip"]["proven_optimal"] is True
    assert record["quality_gap_basis"] == "exact_ip_optimum"

    row = summary_row(record)
    assert len(row) == SUMMARY_COLUMNS
    assert row["num_depots"] == 2
    assert row["num_stations"] == 1
    assert row["num_demands"] == 1
    assert row["requirement_mode"] == "all2"
    assert row["algorithm_verified"] is True
    assert row["formulation"] == "rooted_set_pair_lp"
    assert row["formulation_version"] == "explicit_residual_v1"
    assert row["residual_mode"] == "eliminate_fixed_edges"
    assert row["separation_backend"] == "split_graph_scipy"
    assert row["setpair_witnesses_added"] >= 0
    assert row["ip_status_class"] == "optimal"
    assert row["algorithm_cost_U"] == record["algorithm"]["cost"]
    assert row["ip_objective"] == pytest.approx(7.0)
    assert row["quality_gap_percent"] == pytest.approx(
        record["quality_gap_percent"]
    )
    assert row["ip_lower_bound_used"] == pytest.approx(7.0)
    assert row["ip_lower_bound_basis"] == "final_ip_lower_bound"


def test_build_table_summary_aggregates_one_group() -> None:
    base = summary_row(solved_record())
    rows = [dict(base, seed=101), dict(base, seed=102)]
    table = build_table_summary(rows)

    assert len(table) == 1
    entry = table[0]
    assert entry["num_seeds"] == 2
    assert entry["num_ip_optimal"] == 2
    assert entry["num_algorithm_feasible"] == 2
    assert entry["num_exact_quality_gaps"] == 2
    assert entry["num_bound_from_final_ip"] == 2
    assert entry["quality_gap_mean_percent"] == pytest.approx(
        base["quality_gap_percent"]
    )
    assert entry["quality_gap_max_percent"] == pytest.approx(
        base["quality_gap_percent"]
    )
    assert entry["t_alg_mean_s"] == pytest.approx(base["algorithm_seconds"])


def test_build_table_summary_ignores_rows_without_a_scale() -> None:
    assert build_table_summary([{"variant": "selectable", "error": "boom"}]) == []
