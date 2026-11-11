"""Command-line entry point of the two-radius experiment.

Use this file to choose the instance scales, the number of repetitions (seeds),
the depot variant and the solver options, then to write the JSONL, CSV and
aggregated summary of the run.

Example
-------
    python main.py --scales 5,50,25 10,100,50 --seeds 101 102
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
from pathlib import Path

import cplex
import networkx as nx

from algorithm import (
    DEFAULT_MAX_STALLED_ITERATIONS,
    FORMULATION,
    FORMULATION_VERSION,
    RESIDUAL_MODE,
)
from experiment import _parse_scale, _write_jsonl, _write_summaries, backfill_lp_relaxations, run_case, summary_row
from instance import DEFAULT_FLIGHT_RADIUS, DEFAULT_SCALES, DEFAULT_SEEDS, FLOW_IP_TOO_LARGE_STATUS, MAX_FACILITY_DRAWS, MAX_FLOW_IP_VARIABLES, generate_two_radius_instance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-radius ORL experiment: Stage 1 greedy cover + Stage 2 rooted "
            "set-pair residual LP with iterative rounding vs. compact "
            "node-capacitated FLOW-IP."
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
            "Stage 2 set-pair separator: 'scipy' (A1+A2+A3 scipy C "
            "maximum_flow, default) or 'python' (A1+A3 flat-array Dinic)."))
    parser.add_argument(
        "--cut-mode", choices=["single", "double", "nested"], default="double",
        help=(
            "Stage 2 set pairs per max-flow: 'single' (smallest minimum cut "
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
    parser.add_argument(
        "--max-stalled-iterations", type=int,
        default=DEFAULT_MAX_STALLED_ITERATIONS,
        help=(
            "Stop the Stage 2 row generation once the residual master "
            "objective has not improved for this many consecutive iterations "
            "(0 = no guard). The guard only fires on flat separation tails and "
            "leaves instances whose bound keeps improving untouched."))
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
        "--lp-fill-max-stations", type=int, default=-1,
        help=(
            "Only solve the standalone LP relaxation for tiers with at most "
            "this many stations. The default -1 never solves it, so the "
            "comparison always uses the FLOW-IP optimum or its final best "
            "bound; pass a non-negative tier size to re-enable the backfill, "
            "where 0 means no limit. Above the threshold the LP is skipped and "
            "the reason is recorded, which keeps the backfill from re-running "
            "an LP that cannot finish on this machine."
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
        or args.max_stalled_iterations < 0
        or args.depot_cost[0] < 0
        or args.depot_cost[0] > args.depot_cost[1]
        or args.station_cost[0] < 0
        or args.station_cost[0] > args.station_cost[1]
        or args.lp_fill_max_stations < -1
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
                "rooted set-pair residual LP (formulation="
                + FORMULATION
                + ", version="
                + FORMULATION_VERSION
                + ", residual_mode="
                + RESIDUAL_MODE
                + ") solved by one split-graph threshold min-cut per terminal "
                "over the currently unfixed edges (cut_mode="
                + args.cut_mode
                + ", nested_levels="
                + str(args.nested_levels)
                + ", cut_cap_per_terminal="
                + str(args.cut_cap_per_terminal)
                + ", cut_cap_per_round="
                + str(args.cut_cap_per_round)
                + ", max_stalled_iterations="
                + str(args.max_stalled_iterations)
                + "); violated set pairs compiled into CPLEX rows; dual simplex "
                "to an optimal basic point; batch x>=1/2 rounding; biconnectivity "
                "Test(J) and reverse-delete; separator backend = "
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
                    max_stalled_iterations=args.max_stalled_iterations,
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
