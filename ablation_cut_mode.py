"""Cut-strength ablation for Stage 2: double vs nested with several levels.

The script re-runs Stage 1 + Stage 2 only (no FLOW-IP, which is far more
expensive than the algorithm) on the exact same pre-registered instances as
``output/orl_setpair_full.jsonl``.  The instance fingerprint recorded by the
full run is asserted before every case, so a configuration can never be compared
against a different geometry.

The IP optimum / final best bound already stored in the full run is reused, so
the exact and certified gaps of every configuration are computed without
solving any IP again.

Usage
-----
    python ablation_cut_mode.py                       # all four configurations
    python ablation_cut_mode.py --configs double nested:8 --scales 20,200,100
    python ablation_cut_mode.py --resume              # continue an interruption
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from algorithm import iterative_round_stage_two, stage_one_cover
from instance import generate_two_radius_instance

# Generator parameters of the full pre-registered run, copied verbatim from
# output/orl_setpair_full_settings.json so the instances are bit-identical.
GENERATOR = dict(
    side=100.0,
    eta=0.10,
    fixed_radius=20.0,
    max_facility_draws=500,
    service_ratio=0.50,
    depot_cost=(100, 100),
    station_cost=(10, 30),
)
REQUIREMENT_MODE = "all2"
EPSILON = 0.20
SCALES = [(5, 50, 25), (10, 100, 50), (20, 200, 100), (30, 300, 150), (50, 500, 250)]
VARIANTS = ["selectable", "preinstalled"]
SEEDS = [101, 102, 103, 104, 105]

# (cut_mode, nested_levels) of every configuration of the ablation.
CONFIGURATIONS = [
    ("double", 4),
    ("nested", 4),
    ("nested", 8),
    ("nested", 16),
]


def config_label(cut_mode: str, nested_levels: int, backend: str,
                 sep_cap: int = 0, stall: int = 0) -> str:
    base = cut_mode if cut_mode == "double" else f"{cut_mode}:{nested_levels}"
    if sep_cap:
        base = f"{base}+cap{sep_cap}"
    if stall:
        base = f"{base}+stall{stall}"
    return base if backend == "scipy" else f"{base}@{backend}"


def parse_config(text: str, default_backend: str) -> tuple[str, int, str, int, int]:
    backend = default_backend
    sep_cap = 0
    stall = 0
    if "@" in text:
        text, backend = text.split("@", 1)
    if "+stall" in text:
        text, stall_text = text.split("+stall", 1)
        stall = int(stall_text)
    if "+cap" in text:
        text, cap_text = text.split("+cap", 1)
        sep_cap = int(cap_text)
    if ":" in text:
        mode, levels = text.split(":", 1)
        return mode, int(levels), backend, sep_cap, stall
    return text, 4, backend, sep_cap, stall


def load_baseline(path: Path) -> dict[tuple, dict]:
    """Index the full run by case and keep its fingerprint and IP bound."""
    baseline: dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            instance = record.get("instance")
            if instance is None:
                continue
            key = (
                len(instance["depots"]),
                len(instance["stations"]),
                len(instance["demand_positions"]),
                instance["variant"],
                instance["seed"],
            )
            ip = record.get("ip", {})
            baseline[key] = {
                "fingerprint": record.get("generator", {}).get(
                    "paired_instance_sha256"),
                "ip_objective": ip.get("objective"),
                "ip_proven_optimal": ip.get("proven_optimal") is True,
                "ip_best_bound": ip.get("best_bound"),
                "ip_status": ip.get("status"),
                "double_cost": record.get("algorithm", {}).get("cost"),
                "double_iterations": record.get("algorithm", {}).get(
                    "master_lp_solves"),
                "double_seconds": record.get("algorithm", {}).get(
                    "seconds_total"),
            }
    return baseline


def load_completed(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            case = record.get("case")
            if case is None:
                continue
            done.add((
                case["depots"], case["stations"], case["demands"],
                case["variant"], case["seed"], case["config"],
            ))
    return done


def run_algorithm(instance, time_limit: float, cut_mode: str,
                  nested_levels: int, backend: str, sep_cap: int = 0,
                  stall: int = 0) -> dict:
    start = time.perf_counter()
    deadline = start + time_limit
    try:
        cover = stage_one_cover(instance, EPSILON, deadline, "greedy")
        result = iterative_round_stage_two(
            instance, cover, deadline, backend,
            cut_mode=cut_mode, nested_levels=nested_levels,
            max_separation_iterations=sep_cap,
            max_stalled_iterations=stall,
        )
    except Exception as error:
        return {
            "status": "time_limit" if isinstance(error, TimeoutError) else "failed",
            "error": f"{type(error).__name__}: {error}",
            "seconds_total": time.perf_counter() - start,
        }
    result["seconds_total"] = time.perf_counter() - start
    return result


def exact_or_certified_gap(cost: float, baseline: dict) -> tuple:
    """Return (gap_percent, basis) against the stored IP information."""
    if cost is None:
        return None, None
    if baseline.get("ip_proven_optimal") and baseline.get("ip_objective"):
        optimum = baseline["ip_objective"]
        return 100.0 * (cost - optimum) / optimum, "exact_ip_optimum"
    bound = baseline.get("ip_best_bound")
    if bound:
        return 100.0 * (cost - bound) / bound, "final_ip_lower_bound"
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path,
                        default=Path("output/orl_setpair_full.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/ablation_cut_mode.jsonl"))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of configurations, e.g. double nested:8")
    parser.add_argument("--separator-backend", choices=["python", "scipy"],
                        default="scipy",
                        help="Default backend for configurations without a "
                             "'@backend' suffix.")
    parser.add_argument("--scales", nargs="+", default=None,
                        help="Subset of scales as d,f,p triples")
    parser.add_argument("--sep-caps", nargs="+", type=int, default=None,
                        help="Sweep the separation-iteration cap; each value is "
                             "crossed with every configuration (0 = unlimited).")
    parser.add_argument("--stalls", nargs="+", type=int, default=None,
                        help="Sweep the stalled-iteration cap (stop after this "
                             "many iterations without a bound improvement).")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    base_configurations = (
        [parse_config(text, args.separator_backend) for text in args.configs]
        if args.configs else [
            (mode, levels, args.separator_backend, 0, 0)
            for mode, levels in CONFIGURATIONS
        ]
    )
    caps = args.sep_caps if args.sep_caps else sorted(
        {cap for *_, cap, _ in base_configurations})
    stalls = args.stalls if args.stalls else sorted(
        {stall for *_, stall in base_configurations})
    configurations = [
        (mode, levels, backend, cap, stall)
        for mode, levels, backend, _, _ in base_configurations
        for cap in caps
        for stall in stalls
    ]
    scales = (
        [tuple(int(value) for value in text.split(",")) for text in args.scales]
        if args.scales else SCALES
    )
    baseline = load_baseline(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output) if args.resume else set()
    if not args.resume and args.output.exists():
        args.output.unlink()

    for cut_mode, nested_levels, backend, sep_cap, stall in configurations:
        label = config_label(cut_mode, nested_levels, backend, sep_cap, stall)
        for num_depots, num_stations, num_demands in scales:
            for variant in VARIANTS:
                for seed in SEEDS:
                    key = (num_depots, num_stations, num_demands, variant, seed)
                    case_key = key + (label,)
                    if case_key in completed:
                        continue
                    instance, generator = generate_two_radius_instance(
                        num_depots, num_stations, num_demands, variant,
                        REQUIREMENT_MODE, seed, **GENERATOR,
                    )
                    expected = baseline.get(key, {}).get("fingerprint")
                    if expected is not None and (
                        generator["paired_instance_sha256"] != expected
                    ):
                        raise RuntimeError(
                            "Instance fingerprint mismatch for "
                            f"{key}: the ablation would compare different "
                            "geometry against the full run"
                        )
                    result = run_algorithm(
                        instance, args.time_limit, cut_mode, nested_levels,
                        backend, sep_cap, stall)
                    gap, basis = exact_or_certified_gap(
                        result.get("cost"), baseline.get(key, {}))
                    record = {
                        "case": {
                            "depots": num_depots,
                            "stations": num_stations,
                            "demands": num_demands,
                            "variant": variant,
                            "seed": seed,
                            "config": label,
                            "cut_mode": cut_mode,
                            "nested_levels": nested_levels,
                            "separator_backend": backend,
                            "max_separation_iterations": sep_cap,
                        },
                        "fingerprint": generator["paired_instance_sha256"],
                        "algorithm": result,
                        "gap_percent": gap,
                        "gap_basis": basis,
                        "reference": baseline.get(key, {}),
                    }
                    with args.output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                    print(json.dumps({
                        "config": label,
                        "case": list(key),
                        "status": result.get("status"),
                        "cost": result.get("cost"),
                        "iter": result.get("master_lp_solves"),
                        "mf": result.get("max_flow_calls"),
                        "cut": result.get("cuts_retained"),
                        "seconds": round(result.get("seconds_total", -1), 2),
                        "gap": None if gap is None else round(gap, 3),
                    }, ensure_ascii=True), flush=True)

    write_csv(args.output.with_suffix(".csv"))
    print(json.dumps({"jsonl": str(args.output),
                      "csv": str(args.output.with_suffix(".csv"))},
                     ensure_ascii=True), flush=True)


CSV_COLUMNS = [
    "config", "depots", "stations", "demands", "variant", "seed",
    "status", "cost", "iterations", "max_flow_calls", "cuts_retained",
    "terminals", "rounding_rounds", "separation_truncated",
    "lp_solver_seconds",
    "lp_separation_seconds", "seconds_total", "gap_percent", "gap_basis",
    "double_cost", "double_iterations", "double_seconds",
    "cost_delta_percent", "iteration_ratio", "seconds_ratio",
]


def write_csv(path: Path) -> None:
    configs: list[str] = []
    rows: list[dict] = []
    with path.with_suffix(".jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            case = record["case"]
            algorithm = record["algorithm"]
            reference = record.get("reference", {})
            label = case["config"]
            if label not in configs:
                configs.append(label)
            double_cost = reference.get("double_cost")
            cost = algorithm.get("cost")
            delta = (
                100.0 * (cost - double_cost) / double_cost
                if cost is not None and double_cost else None
            )
            double_iterations = reference.get("double_iterations")
            double_seconds = reference.get("double_seconds")
            rows.append({
                "config": label,
                "depots": case["depots"],
                "stations": case["stations"],
                "demands": case["demands"],
                "variant": case["variant"],
                "seed": case["seed"],
                "status": algorithm.get("status"),
                "cost": cost,
                "iterations": algorithm.get("master_lp_solves"),
                "max_flow_calls": algorithm.get("max_flow_calls"),
                "cuts_retained": algorithm.get("cuts_retained"),
                "terminals": algorithm.get("terminals_total"),
                "rounding_rounds": algorithm.get("num_rounds"),
                "separation_truncated": algorithm.get("separation_truncated"),
                "lp_solver_seconds": algorithm.get("lp_solver_seconds"),
                "lp_separation_seconds": algorithm.get("lp_separation_seconds"),
                "seconds_total": algorithm.get("seconds_total"),
                "gap_percent": record.get("gap_percent"),
                "gap_basis": record.get("gap_basis"),
                "double_cost": double_cost,
                "double_iterations": double_iterations,
                "double_seconds": double_seconds,
                "cost_delta_percent": delta,
                "iteration_ratio": (
                    algorithm["master_lp_solves"] / double_iterations
                    if algorithm.get("master_lp_solves") is not None
                    and double_iterations else None
                ),
                "seconds_ratio": (
                    algorithm["seconds_total"] / double_seconds
                    if algorithm.get("seconds_total") is not None
                    and double_seconds else None
                ),
            })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"csv_rows": len(rows)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
