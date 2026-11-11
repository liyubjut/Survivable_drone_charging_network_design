"""One experiment case, the quality metrics derived from it, and the writers.

``run_case`` runs the algorithm and the matching FLOW-IP baseline on one instance
and returns a self-contained record.  The metric helpers turn that record into the
per-case CSV row and the aggregated main table, always reporting exact gaps when
the IP is optimal and certified gap upper bounds otherwise.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

from algorithm import (
    DEFAULT_MAX_STALLED_ITERATIONS,
    iterative_round_stage_two,
    stage_one_cover,
)
from baseline import solve_flow_benchmark, solve_flow_lp_relaxation
from instance import FLOW_IP_SKIPPED_STATUS, FLOW_IP_TOO_LARGE_STATUS, TwoRadiusInstance, generate_two_radius_instance


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
    max_stalled_iterations: int = DEFAULT_MAX_STALLED_ITERATIONS,
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
            max_stalled_iterations=max_stalled_iterations,
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

    The reported comparison uses the FLOW-IP itself: its optimum when it proves
    optimality and otherwise CPLEX's final best bound.  The standalone LP
    relaxation of the same compact model is disabled by default
    (``max_stations < 0``) because that model is far too large for the backfill
    to finish: it is the same rows, columns and coefficients but without the
    integrality and the root cut loop that let the IP close the root gap.  A
    non-negative ``max_stations`` re-enables it for tiers up to that size, where
    the final best bound degenerates to a tiny value when the time limit expires
    inside the root cut loop; taking the larger of the two bounds can never
    overstate the certified quality.
    """
    if ip.get("status") == FLOW_IP_TOO_LARGE_STATUS:
        return
    if ip.get("proven_optimal") is True:
        return
    existing = ip.get("lp_relaxation")
    if (
        isinstance(existing, (int, float))
        and not isinstance(existing, bool)
        and existing > 0
    ):
        return
    if max_stations < 0:
        ip["lp_relaxation_status"] = "disabled_by_request"
        ip["lp_relaxation"] = None
        ip["lp_relaxation_error"] = (
            "the standalone LP relaxation is disabled, so the comparison uses "
            "the FLOW-IP optimum or, when the IP does not prove optimality, "
            "CPLEX's final best bound"
        )
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
        "formulation": algorithm.get("formulation"),
        "formulation_version": algorithm.get("formulation_version"),
        "residual_mode": algorithm.get("residual_mode"),
        "separation_backend": algorithm.get("separation_backend"),
        "terminals_total": algorithm.get("terminals_total"),
        "global_edge_count": algorithm.get("global_edge_count"),
        "setpair_witnesses_added": algorithm.get("setpair_witnesses_added"),
        "stage_two_max_violation": algorithm.get("max_violation"),
        "stage_two_rebuild_seconds": algorithm.get("rebuild_seconds"),
        "scaling_fallback_calls": algorithm.get("scaling_fallback_calls"),
        "separator_backend": algorithm.get("separator_backend"),
        "cut_mode": algorithm.get("cut_mode"),
        "nested_levels": algorithm.get("nested_levels"),
        "cut_cap_per_terminal": algorithm.get("cut_cap_per_terminal"),
        "cut_cap_per_round": algorithm.get("cut_cap_per_round"),
        "max_stalled_iterations": algorithm.get("max_stalled_iterations"),
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
