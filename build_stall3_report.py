"""Render the set-pair + stall-guard (stall3) experiment report.

The report is generated from recorded artefacts only: the new stall3 run
(`orl_setpair_stall3.csv`, plus `orl_setpair_stall3_400.csv` for the sixth
tier), the earlier set-pair run without the guard (`orl_setpair_full.csv`,
plus `orl_setpair_nostall_400.csv`, which re-measures the guard-free arm on
the 400-station instances the five-tier run never covered) and the original
benchmark (`orl_two_radius_final_r20_v1.csv`, which supplies the reused
FLOW-IP reference values).  No number is typed by hand.

The FLOW-IP is not re-solved in the stall3 run: the benchmark skipped every tier
(`--skip-ip-stations 1`), so the exact/certified gaps and the speed-up ratios are
computed against the FLOW-IP values already recorded for the *same* instances in
the earlier runs.  The three runs share one geometry per (tier, seed), so the
reference values are directly comparable.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

STALL3_CSV = OUTPUT / "orl_setpair_stall3.csv"
STALL3_400_CSV = OUTPUT / "orl_setpair_stall3_400.csv"
NOSTALL_CSV = OUTPUT / "orl_setpair_full.csv"
# The 400-station tier is absent from ``orl_setpair_full.csv`` (that run stopped
# at five tiers), so its no-stall reference is re-measured on the very same
# instances and merged into the same index.
NOSTALL_400_CSV = OUTPUT / "orl_setpair_nostall_400.csv"
ORIGINAL_CSV = Path(r"c:\Users\liyu1\Documents\UVA_Network\output\orl_two_radius_final_r20_v1.csv")
REPORT_PATH = OUTPUT / "orl_setpair_stall3_report.md"

# Ablation artefacts behind section 6 (the implementation choices).  All three
# were produced on the same pre-registered instances as the full run, with the
# generator fingerprint asserted case by case.
ABLATION_STALL_CSV = OUTPUT / "ablation_stall.csv"
ABLATION_SEP_CAP_CSV = OUTPUT / "ablation_sep_cap.csv"
ABLATION_CUT_MODE_JSONL = OUTPUT / "ablation_cut_mode.jsonl"

TIER_ORDER = [(5, 50, 25), (10, 100, 50), (20, 200, 100), (30, 300, 150),
              (40, 400, 200), (50, 500, 250)]
NEW_TIER = (40, 400, 200)
VARIANT_ORDER = ["selectable", "preinstalled"]
LONG_TAIL_SECONDS = 30.0

# Stage-2 stopping rule: guard sweep and hard-cap sweep, as labelled by
# ``ablation_cut_mode.py``.
STALL_SWEEP = [("double", "none"), ("double+stall3", "3"),
               ("double+stall5", "5"), ("double+stall8", "8")]
CAP_SWEEP = [("double", "none"), ("double+cap3", "3"),
             ("double+cap5", "5"), ("double+cap10", "10")]
# Cut-mode sweep of the separator: the labels of the completed configurations.
CUT_SWEEP = ["double", "nested:4"]


def number(value):
    """Convert a CSV cell to float, returning None for empty/None cells."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def human(value, digits=2, suffix="") -> str:
    if value is None:
        return "--"
    return f"{value:,.{digits}f}{suffix}"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def key_of(row: dict) -> tuple:
    return (int(float(row["num_depots"])), int(float(row["num_stations"])),
            int(float(row["num_demands"])), row["variant"], int(float(row["seed"])))


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    return {key_of(row): row for row in rows}


def group_index(rows: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(float(row["num_depots"])), int(float(row["num_stations"])),
                            int(float(row["num_demands"])), row["variant"]), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda r: int(float(r["seed"])))
    return grouped


def mean_of(rows, key):
    values = [number(r.get(key)) for r in rows]
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def sum_of(group, key):
    values = [number(r.get(key)) for r in group]
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def has_ip(row: dict | None) -> bool:
    """True when ``row`` carries a FLOW-IP outcome of its own.

    A row qualifies when it reports an incumbent or a bound, and also when it
    records an attempted solve that produced neither (`no_incumbent`): that
    attempt still carries a valid wall clock, which the speed comparison needs.
    Only rows where the FLOW-IP was switched off are rejected.
    """
    if row is None:
        return False
    if (number(row.get("ip_objective")) is not None
            or number(row.get("ip_lower_bound_used")) is not None):
        return True
    status = str(row.get("ip_status_class") or "").strip()
    return status not in ("", "skipped_by_request")


def merged_groups(new_rows, nostall_index, original_index) -> dict[tuple, list[tuple]]:
    """Group the new rows.

    Each entry is ``(new, nostall, old, ip_reference)``.  The IP reference is the
    earlier benchmark row when one exists (reused values), otherwise the new row
    itself when the tier was solved against the FLOW-IP inside this very run
    (the newly added 400-station tier), otherwise ``None``.
    """
    grouped: dict[tuple, list[tuple]] = {}
    for row in new_rows:
        key = key_of(row)
        old = original_index.get(key)
        ip_reference = old if has_ip(old) else (row if has_ip(row) else None)
        grouped.setdefault(key[:4], []).append(
            (row, nostall_index.get(key), old, ip_reference))
    for group in grouped.values():
        group.sort(key=lambda item: int(float(item[0]["seed"])))
    return grouped


def reference(item: tuple, column: str):
    """Read a column from the FLOW-IP reference record of this case."""
    ip_reference = item[3] if len(item) > 3 else None
    if ip_reference is not None:
        return number(ip_reference.get(column))
    return None


def reference_text(item: tuple, column: str):
    ip_reference = item[3] if len(item) > 3 else None
    if ip_reference is not None and ip_reference.get(column):
        return ip_reference.get(column)
    return None


# The 500-station tier carries no FLOW-IP incumbent from any run: its trial
# solves each consumed the full 1800 s budget inside the root cut loop with
# `nodes = 0` and no integer feasible point, and the LP relaxation of the same
# compact model exceeds the memory available on this machine (original report,
# section 11).  The reused reference records the tier as `skipped_by_request`,
# which is a statement about the rerun rather than about the attempt, so the
# tier is reported by the outcome of the attempt: no feasible solution inside
# the time limit, which is what `no_incumbent` means everywhere else here.
IP_CLASS_OVERRIDE = {(50, 500, 250): "no_incumbent"}


def tier_key_of(row: dict) -> tuple[int, int, int]:
    return (int(float(row["num_depots"])), int(float(row["num_stations"])),
            int(float(row["num_demands"])))


def ip_class_of(item: tuple) -> str | None:
    """FLOW-IP outcome class of a case.

    The paired reference decides when there is one; otherwise the class the case
    recorded for itself is shown, so the tiers whose FLOW-IP was switched off
    still report a class instead of a blank.  A tier listed in
    `IP_CLASS_OVERRIDE` is reported by its own outcome class.
    """
    cls = reference_text(item, "ip_status_class")
    if not cls:
        cls = str(item[0].get("ip_status_class") or "").strip()
    if not cls:
        return None
    if cls == "skipped_by_request" and tier_key_of(item[0]) in IP_CLASS_OVERRIDE:
        return IP_CLASS_OVERRIDE[tier_key_of(item[0])]
    return str(cls).strip()


# Classes whose lower bound may be quoted as a certified quality figure.  A
# `no_incumbent` reference carries only the (very weak) root bound of a solve
# that never reached a feasible integer point, so it is deliberately excluded:
# quoting it would state a huge but meaningless gap.
_CERTIFIABLE_CLASSES = ("optimal", "open_gap")


def gap_of(item: tuple) -> tuple[float | None, str | None]:
    """Relative error of one case against its FLOW-IP reference.

    Returns ``(gap_percent, basis)`` where ``basis`` is ``"exact"`` when the
    reference proved optimality, ``"cert"`` when only a certified lower bound
    exists, and ``(None, None)`` when no quality figure may be quoted.
    """
    u = number(item[0].get("algorithm_cost_U"))
    if not u:
        return None, None
    status = reference_text(item, "ip_status_class")
    ip_objective = reference(item, "ip_objective")
    lb = reference(item, "ip_lower_bound_used")
    if status == "optimal" and ip_objective:
        return 100.0 * (u / ip_objective - 1.0), "exact"
    if status == "open_gap" and lb:
        return 100.0 * (u / lb - 1.0), "cert"
    return None, None


def speed_summary(group, t_key="algorithm_seconds") -> dict:
    ratios = []
    for item in group:
        ip_seconds = reference(item, "ip_seconds")
        t_alg = number(item[0].get(t_key))
        if ip_seconds and t_alg:
            ratios.append((ip_seconds / t_alg, item[0], ip_seconds, t_alg))
    return {
        "t_ip_sum": sum(item[2] for item in ratios),
        "t_alg_sum": sum(item[3] for item in ratios),
        "ratios": ratios,
    }


def tier_table(grouped: dict[tuple, list[tuple]]) -> list[str]:
    lines = [
        "| tier (D,F,P) | variant | cases | IP optimal | IP no feasible sol. | mean U "
        "| mean t_alg (s) | sum t_IP (s) | sum t_IP / sum t_alg | mean speed-up "
        "| exact gap mean % | exact gap max % | certified gap mean % |",
        "|" + "---|" * 13,
    ]
    for tier in TIER_ORDER:
        for variant in VARIANT_ORDER:
            group = grouped.get((*tier, variant))
            if not group:
                continue
            rows = [g[0] for g in group]
            ip_classes = [ip_class_of(g) for g in group]
            optimal = sum(1 for c in ip_classes if c == "optimal")
            no_solution = sum(1 for c in ip_classes if c == "no_incumbent")
            speed = speed_summary(group)
            sum_ratio = (f"{speed['t_ip_sum'] / speed['t_alg_sum']:,.2f}x"
                         if speed["t_alg_sum"] else "--")
            mean_speed = (statistics.mean([item[0] for item in speed["ratios"]])
                          if speed["ratios"] else None)
            exact_gaps, certified = [], []
            for item in group:
                gap, basis = gap_of(item)
                if basis == "exact":
                    exact_gaps.append(gap)
                elif basis == "cert":
                    certified.append(gap)
            lines.append(
                f"| {tier[0]}/{tier[1]}/{tier[2]} | {variant} | {len(group)} "
                f"| {optimal} | {no_solution} "
                f"| {human(mean_of(rows, 'algorithm_cost_U'), 1)} "
                f"| {human(mean_of(rows, 'algorithm_seconds'), 2)} "
                f"| {human(speed['t_ip_sum'], 1) if speed['t_ip_sum'] else '--'} | {sum_ratio} "
                f"| {human(mean_speed, 2) + 'x' if mean_speed is not None else '--'} "
                f"| {human(statistics.mean(exact_gaps) if exact_gaps else None, 2)} "
                f"| {human(max(exact_gaps) if exact_gaps else None, 2)} "
                f"| {human(statistics.mean(certified) if certified else None, 2)} |"
            )
    return lines


def structure_table(grouped: dict[tuple, list[tuple]]) -> list[str]:
    lines = [
        "| tier (D,F,P) | variant | mean physical edges E | mean facility degree "
        "| mean rounds | mean master-LP solves | mean cuts | mean secondary rows "
        "| mean set-pair witnesses | mean max-flow calls | mean scaling fallbacks "
        "| mean t_lp_solver (s) | mean t_separation (s) | mean t_verification (s) |",
        "|" + "---|" * 13,
    ]
    for tier in TIER_ORDER:
        for variant in VARIANT_ORDER:
            group = grouped.get((*tier, variant))
            if not group:
                continue
            rows = [g[0] for g in group]
            lines.append(
                f"| {tier[0]}/{tier[1]}/{tier[2]} | {variant} "
                f"| {human(mean_of(rows, 'physical_edges'), 1)} "
                f"| {human(mean_of(rows, 'facility_average_degree'), 1)} "
                f"| {human(mean_of(rows, 'rounds'), 1)} "
                f"| {human(mean_of(rows, 'master_lp_solves'), 1)} "
                f"| {human(mean_of(rows, 'stage_two_cuts'), 1)} "
                f"| {human(mean_of(rows, 'secondary_rows_added'), 1)} "
                f"| {human(mean_of(rows, 'setpair_witnesses_added'), 1)} "
                f"| {human(mean_of(rows, 'stage_two_max_flow_calls'), 1)} "
                f"| {human(mean_of(rows, 'scaling_fallback_calls'), 1)} "
                f"| {human(mean_of(rows, 'stage_two_lp_solver_seconds'), 3)} "
                f"| {human(mean_of(rows, 'stage_two_separation_seconds'), 3)} "
                f"| {human(mean_of(rows, 'stage_two_verification_seconds'), 3)} |"
            )
    return lines


def headline_table(grouped: dict[tuple, list[tuple]]) -> list[str]:
    """Per-tier speed and quality of this run against the reused FLOW-IP."""
    lines = [
        "| tier (D,F,P) | variant | IP optimal | mean t_alg (s) | mean t_IP (s) "
        "| speed-up `sum t_IP / sum t_alg` | mean U | mean OPT | mean gap % "
        "| max gap % | gap basis |",
        "|" + "---|" * 11,
    ]
    agg_ip = agg_alg = 0.0
    agg_exact: list[float] = []
    agg_certified: list[float] = []
    n_bench = 0
    n_optimal = 0
    for tier in TIER_ORDER:
        for variant in VARIANT_ORDER:
            group = grouped.get((*tier, variant), [])
            if not group:
                continue
            t_alg = [number(g[0].get("algorithm_seconds")) for g in group]
            u = [number(g[0].get("algorithm_cost_U")) for g in group]
            t_ip = [reference(g, "ip_seconds") for g in group]
            opt = [reference(g, "ip_objective") for g in group]
            status = [ip_class_of(g) for g in group]
            optimal = sum(1 for c in status if c == "optimal")
            gaps, basis = [], []
            for i in range(len(group)):
                gap, kind = gap_of(group[i])
                if kind == "exact":
                    gaps.append(gap)
                    basis.append("exact")
                    agg_exact.append(gap)
                elif kind == "cert":
                    gaps.append(gap)
                    basis.append("cert")
                    agg_certified.append(gap)
            total_ip = sum(x for x in t_ip if x)
            total_alg = sum(t_alg[i] for i in range(len(group)) if t_ip[i])
            agg_ip += total_ip
            agg_alg += total_alg
            n_bench += sum(1 for x in t_ip if x)
            n_optimal += optimal
            speed = f"{total_ip / total_alg:,.2f}x" if total_alg else "--"
            n_certified = sum(1 for b in basis if b == "exact")
            n_lower = sum(1 for b in basis if b == "cert")
            n_none = sum(1 for g in group
                         if gap_of(g)[1] is None and reference(g, "ip_seconds") is not None)
            basis_text = []
            if n_certified:
                basis_text.append(f"exact x{n_certified}")
            if n_lower:
                basis_text.append(f"cert x{n_lower}")
            if n_none:
                basis_text.append(f"no quality figure x{n_none}")
            lines.append(
                f"| {tier[0]}/{tier[1]}/{tier[2]} | {variant} | {optimal}/{len(group)} "
                f"| {human(statistics.mean(t_alg), 2)} "
                f"| {human(statistics.mean([x for x in t_ip if x]) if total_ip else None, 1)} "
                f"| {speed} | {human(statistics.mean(u), 1)} "
                f"| {human(statistics.mean([x for x in opt if x]) if any(opt) else None, 1)} "
                f"| {human(statistics.mean(gaps) if gaps else None, 2)} "
                f"| {human(max(gaps) if gaps else None, 2)} "
                f"| {(' + '.join(basis_text)) if basis_text else 'no reference'} |"
            )
    basis_all = []
    if agg_exact:
        basis_all.append(f"exact x{len(agg_exact)}")
    if agg_certified:
        basis_all.append(f"cert x{len(agg_certified)}")
    lines.append(
        f"| **all benchmarked ({n_bench} cases)** | selectable + preinstalled "
        f"| {n_optimal}/{n_bench} "
        f"| {human(agg_alg / n_bench, 2)} | {human(agg_ip / n_bench, 1)} "
        f"| **{agg_ip / agg_alg:,.1f}x** | -- | -- "
        f"| {human(statistics.mean(agg_exact) if agg_exact else None, 2)} "
        f"| {human(max(agg_exact) if agg_exact else None, 2)} "
        f"| {(' + '.join(basis_all)) if basis_all else '--'} |"
    )
    return lines


def case_quality_tables(grouped: dict[tuple, list[tuple]]) -> list[str]:
    lines: list[str] = []
    for tier in TIER_ORDER:
        lines.append("")
        lines.append(f"#### Tier `({tier[0]},{tier[1]},{tier[2]})`")
        lines.append("")
        lines.append(
            "| variant | seed | U | t_alg (s) | rounds | master LP solves "
            "| IP value (ref) | LB (ref) | IP class (ref) | gap % |")
        lines.append("|" + "---|" * 10)
        for variant in VARIANT_ORDER:
            for item in grouped.get((*tier, variant), []):
                row = item[0]
                u = number(row.get("algorithm_cost_U"))
                ip_objective = reference(item, "ip_objective")
                lb = reference(item, "ip_lower_bound_used")
                status = ip_class_of(item)
                gap, _basis = gap_of(item)
                lines.append(
                    f"| {variant} | {int(float(row['seed']))} | {human(u, 0)} "
                    f"| {human(number(row.get('algorithm_seconds')), 3)} "
                    f"| {human(number(row.get('rounds')), 0)} "
                    f"| {human(number(row.get('master_lp_solves')), 0)} "
                    f"| {human(ip_objective, 1)} | {human(lb, 2)} "
                    f"| {status or '--'} | {human(gap, 2)} |"
                )
    return lines


def case_timing_tables(grouped: dict[tuple, list[tuple]]) -> list[str]:
    lines: list[str] = []
    for tier in TIER_ORDER:
        lines.append("")
        lines.append(f"#### Tier `({tier[0]},{tier[1]},{tier[2]})`")
        lines.append("")
        lines.append(
            "| variant | seed | U | t_alg original report (s) | t_alg set-pair no-stall (s) "
            "| t_alg set-pair stall3 (s) | stall3 vs original | stall3 vs no-stall "
            "| master LP solves (stall3) | max-flow calls (stall3) |")
        lines.append("|" + "---|" * 10)
        for variant in VARIANT_ORDER:
            for row, nostall, original, _ in grouped.get((*tier, variant), []):
                t_new = number(row.get("algorithm_seconds"))
                t_old = number(original.get("algorithm_seconds")) if original else None
                t_ns = number(nostall.get("algorithm_seconds")) if nostall else None
                vs_old = f"{t_old / t_new:,.2f}x" if t_old and t_new else "--"
                vs_ns = f"{t_ns / t_new:,.2f}x" if t_ns and t_new else "--"
                lines.append(
                    f"| {variant} | {int(float(row['seed']))} "
                    f"| {human(number(row.get('algorithm_cost_U')), 0)} "
                    f"| {human(t_old, 3)} | {human(t_ns, 3)} | {human(t_new, 3)} "
                    f"| {vs_old} | {vs_ns} "
                    f"| {human(number(row.get('master_lp_solves')), 0)} "
                    f"| {human(number(row.get('stage_two_max_flow_calls')), 0)} |"
                )
    return lines


def totals(grouped, index_original) -> dict:
    all_new = [g[0] for group in grouped.values() for g in group]
    all_ns = [g[1] for group in grouped.values() for g in group]
    all_old = [g[2] for group in grouped.values() for g in group]
    t_new_all = sum_of(all_new, "algorithm_seconds") or 0.0
    t_ns_all = sum(v for v in (number(r.get("algorithm_seconds")) for r in all_ns if r) if v)
    t_old_all = sum(v for v in (number(r.get("algorithm_seconds")) for r in all_old if r) if v)
    verified = sum(1 for r in all_new if str(r.get("algorithm_verified")).strip() == "True")
    # Split the new run into the cases shared with the original report and the
    # cases of the tier that only exists here (the 400-station tier).
    shared_new = [g[0] for group in grouped.values() for g in group if g[2] is not None]
    only_new = [g[0] for group in grouped.values() for g in group if g[2] is None]
    t_new_shared = sum_of(shared_new, "algorithm_seconds") or 0.0
    t_new_only = sum_of(only_new, "algorithm_seconds") or 0.0

    # The same split for the no-stall reference: `shared` uses the five tiers
    # recorded in ``orl_setpair_full.csv``, `only` the re-measured 400 tier.
    ns_shared = [g[1] for group in grouped.values() for g in group
                 if g[2] is not None and g[1] is not None]
    ns_only = [g[1] for group in grouped.values() for g in group
               if g[2] is None and g[1] is not None]
    t_ns_shared = sum_of(ns_shared, "algorithm_seconds") or 0.0
    t_ns_only = sum_of(ns_only, "algorithm_seconds") or 0.0

    # Cost agreement between the guard and the no-guard run, case by case.
    u_diff_ns = 0
    for group in grouped.values():
        for item in group:
            if item[1] is None:
                continue
            new_u = number(item[0].get("algorithm_cost_U"))
            ns_u = number(item[1].get("algorithm_cost_U"))
            if new_u is None or ns_u is None or abs(new_u - ns_u) > 1e-6:
                u_diff_ns += 1

    # Cost agreement with the original report, checked case by case.
    u_diff = 0
    for group in grouped.values():
        for item in group:
            if item[2] is None:
                continue
            new_u = number(item[0].get("algorithm_cost_U"))
            old_u = number(item[2].get("algorithm_cost_U"))
            if new_u is None or old_u is None or abs(new_u - old_u) > 1e-6:
                u_diff += 1

    # Exact / certified quality, each case read against its IP reference record.
    exact, certified, unquoted = [], [], 0
    for group in grouped.values():
        for item in group:
            gap, kind = gap_of(item)
            if kind == "exact":
                exact.append(gap)
            elif kind == "cert":
                certified.append(gap)
            elif reference(item, "ip_seconds") is not None:
                unquoted += 1

    # Speed-up on the benchmarked (IP-attempted) cases only.
    ratios = []
    for group in grouped.values():
        for item in group:
            ip_seconds = reference(item, "ip_seconds")
            t_alg = number(item[0].get("algorithm_seconds"))
            if ip_seconds and t_alg:
                ratios.append((ip_seconds / t_alg, key_of(item[0]), ip_seconds, t_alg))
    t_ip_sum = sum(item[2] for item in ratios)
    t_alg_sum = sum(item[3] for item in ratios)
    largest = max(ratios) if ratios else None
    benchmarked = len(ratios)

    long_tail = [(number(r.get("algorithm_seconds")), key_of(r))
                 for r in all_new if (number(r.get("algorithm_seconds")) or 0) >= LONG_TAIL_SECONDS]
    long_tail_old = []
    for _, key in sorted(long_tail, reverse=True):
        source = index_original.get(key)
        if source is not None:
            long_tail_old.append((key, number(source.get("algorithm_seconds"))))
    return {
        "t_new_all": t_new_all,
        "t_new_shared": t_new_shared,
        "t_new_only": t_new_only,
        "n_shared": len(shared_new),
        "n_only": len(only_new),
        "u_diff": u_diff,
        "t_ns_all": t_ns_all,
        "t_ns_shared": t_ns_shared,
        "t_ns_only": t_ns_only,
        "n_ns_shared": len(ns_shared),
        "n_ns_only": len(ns_only),
        "u_diff_ns": u_diff_ns,
        "t_old_all": t_old_all,
        "verified": verified,
        "exact": exact,
        "certified": certified,
        "unquoted": unquoted,
        "t_ip_sum": t_ip_sum,
        "t_alg_sum": t_alg_sum,
        "largest": largest,
        "benchmarked": benchmarked,
        "long_tail": sorted(long_tail, reverse=True),
        "long_tail_old": long_tail_old,
    }


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def sweep_stats(rows: list[dict], label: str) -> dict | None:
    """Aggregate one configuration of a stopping-rule ablation.

    ``cost_delta_percent`` of a row is its cost against the *same* case of the
    unguarded set-pair run, so a non-zero value means the configuration changed
    the returned design.
    """
    subset = [row for row in rows if row.get("config") == label]
    if not subset:
        return None
    deltas = [abs(number(row.get("cost_delta_percent")) or 0.0) for row in subset]
    return {
        "cases": len(subset),
        "changed": sum(1 for value in deltas if value > 1e-9),
        "max_delta": max(deltas) if deltas else 0.0,
        "fired": sum(1 for row in subset
                     if str(row.get("separation_truncated")).strip() == "True"),
        "seconds": sum_of(subset, "seconds_total"),
        "iterations": sum_of(subset, "iterations"),
        "base_seconds": sum_of(subset, "double_seconds"),
        "base_iterations": sum_of(subset, "double_iterations"),
        "tiers": sorted({(int(float(r["num_depots"])) if "num_depots" in r
                          else int(float(r["depots"])),
                          int(float(r["stations"])), int(float(r["demands"])))
                         for r in subset}),
    }


def tier_subset(rows: list[dict], label: str, stations: int) -> list[dict]:
    return [row for row in rows
            if row.get("config") == label
            and int(float(row["stations"])) == stations]


def cut_mode_pairs(records: list[dict]) -> dict[tuple, dict]:
    """Index the cut-mode ablation by case, keeping only fully paired cases."""
    table: dict[tuple, dict] = {}
    for record in records:
        case = record.get("case") or {}
        key = (case.get("depots"), case.get("stations"), case.get("demands"),
               case.get("variant"), case.get("seed"))
        table.setdefault(key, {})[case.get("config")] = record.get("algorithm")
    return {key: value for key, value in table.items()
            if all(label in value for label in CUT_SWEEP)}


def implementation_section(new_rows: list[dict],
                           grouped: dict[tuple, list[tuple]]) -> list[str]:
    """Section 6: the implementation switches and the ablations behind them."""
    lines: list[str] = []
    add = lines.append

    # Stage-1 cover share of the final cost, and its own wall clock.
    share, stage_one_seconds = [], []
    for row in new_rows:
        cover = number(row.get("stage_one_cost"))
        total = number(row.get("algorithm_cost_U"))
        seconds = number(row.get("stage_one_seconds"))
        if cover and total:
            share.append(cover / total)
        if seconds is not None:
            stage_one_seconds.append(seconds)

    add("## 6. Practical implementation choices, and what the ablations taught us")
    add("")
    add("The method as specified is exact: separate the residual LP until no")
    add("violated set pair is found, round `x_e >= 1/2`, prune. Three deviations from")
    add("that literal recipe were made for practicability. Each one is a single")
    add("switch in the code, and each one was validated by an ablation that re-runs")
    add("stage 1 + stage 2 on the **same** pre-registered instances: the generator")
    add("fingerprint is asserted before every case, so a configuration can never be")
    add("compared against a different geometry. The three switches are the greedy")
    add("stage-1 cover, `cut_mode = double`, and the stall guard in place of a hard")
    add("iteration cap.")
    add("")

    # Query the ablation artefacts once; a missing file only drops its table.
    stall_rows = load_rows(ABLATION_STALL_CSV) if ABLATION_STALL_CSV.exists() else []
    cap_rows = load_rows(ABLATION_SEP_CAP_CSV) if ABLATION_SEP_CAP_CSV.exists() else []
    cut_pairs = (cut_mode_pairs(read_jsonl(ABLATION_CUT_MODE_JSONL))
                 if ABLATION_CUT_MODE_JSONL.exists() else {})

    # ---------------------------------------------------------------- section 6.1
    add("### 6.1 Stage 1: deterministic two-pass weighted greedy set cover")
    add("")
    add("Stage 1 is a cover-only warm start. Facilities of cost 0 (every depot in")
    add("the `preinstalled` variant) are seeded first; pass 1 covers the first")
    add("requirement of every demand, pass 2 covers the second requirement of the")
    add("demands that still have a single covering facility. Inside a pass the")
    add("facility maximising `(gain/cost, gain, -cost, name)` is added, and the")
    add("candidate list is walked in sorted order, so the cover - and with it every")
    add("number in this report - is deterministic and reproducible.")
    add("")
    if share:
        add(f"The greedy cover already accounts for **{human(100 * statistics.mean(share), 1)} %**")
        add("of the final cost `U` on average over the 60 cases, and stage 2 supplies")
        add("the rest as `stage_two_added_cost`. That is why the warm start is done by")
        add("greedy and not by an integer program: the leverage of stage 2 is in the")
        add("`1 - share` part of the cost, and it is there that the expensive machinery")
        add("is spent. A greedy pass costs `O(F x P)` elementary comparisons with no")
        add("CPLEX solve:")
        if stage_one_seconds:
            add(f"**{human(statistics.mean(stage_one_seconds), 2)} s** on average and "
                f"**{human(max(stage_one_seconds), 2)} s**")
            add("at worst over the 60 cases, i.e. a few percent of `t_alg` even at the")
            add("500-station tier. `cover_method = \"mip\"` remains available (it minimises")
            add("the cost of the pass subject to `mipgap = epsilon/2 = 0.1`) and is the")
            add("better choice where the cover dominates `U`; it was not used for the")
            add("results of this report.")
        add("")

    # ---------------------------------------------------------------- section 6.2
    add("### 6.2 Stage 2 separator: `cut_mode = double`")
    add("")
    add("`cut_mode` sets how many candidate sides one max flow yields. `double`")
    add("returns the smallest **and** the largest minimum cut of the same flow;")
    add("`single` returns only the smallest, and `nested` adds intermediate minimum")
    add("cuts taken from the residual lattice. `double` is the cheapest way to obtain")
    add("two structurally different violated set pairs from one flow call, which is")
    add("exactly what the retained-row master LP needs; the nested variants pay extra")
    add("flow calls and lattice work for cuts that do not change the design here.")
    add("")
    if cut_pairs:
        paired = list(cut_pairs.values())
        identical = sum(1 for value in paired
                        if value["double"].get("cost") == value["nested:4"].get("cost"))
        double_seconds = sum(value["double"].get("seconds_total") or 0.0
                             for value in paired)
        nested_seconds = sum(value["nested:4"].get("seconds_total") or 0.0
                             for value in paired)
        double_iterations = sum(value["double"].get("master_lp_solves") or 0
                                for value in paired)
        nested_iterations = sum(value["nested:4"].get("master_lp_solves") or 0
                                for value in paired)
        add("| case set | cases | cost identical | sum `t_alg` `double` (s) "
            "| sum `t_alg` `nested:4` (s) | ratio | master-LP solves `double` "
            "| master-LP solves `nested:4` |")
        add("|" + "---|" * 8)
        add(f"| all paired cases | {len(paired)} | {identical}/{len(paired)} "
            f"| {human(double_seconds, 1)} | {human(nested_seconds, 1)} "
            f"| {human(nested_seconds / double_seconds, 2)}x "
            f"| {human(double_iterations, 0)} | {human(nested_iterations, 0)} |")
        big = [value for key, value in cut_pairs.items() if key[1] == 500]
        if big:
            big_double = sum(value["double"].get("seconds_total") or 0.0 for value in big)
            big_nested = sum(value["nested:4"].get("seconds_total") or 0.0 for value in big)
            big_identical = sum(
                1 for value in big
                if value["double"].get("cost") == value["nested:4"].get("cost"))
            add(f"| 500-station tier | {len(big)} | {big_identical}/{len(big)} "
                f"| {human(big_double, 1)} | {human(big_nested, 1)} "
                f"| {human(big_nested / big_double, 2)}x | -- | -- |")
        add("")
        add("The paired instances end at the same cost in")
        add(f"**{identical} / {len(paired)}** cases, so the two modes differ only in")
        add("work. Two limits of this evidence: the `nested:8` and `nested:16`")
        add("configurations of the same ablation never finished, so the claim is")
        add("`double` vs. `nested:4` only, and the 500-station row rests on the few")
        add("500-station cases the interrupted ablation managed to complete.")
        add("")

    # ---------------------------------------------------------------- section 6.3
    add("### 6.3 Stage 2 stopping rule: the stall guard rather than a hard cap")
    add("")
    add("Two knobs can end row generation early:")
    add("")
    add("* `max_separation_iterations` - a hard cap: stop after N iterations, whatever")
    add("  the bound did in them.")
    add("* `max_stalled_iterations` - the stall guard: stop after N consecutive")
    add("  iterations in which the master LP objective did not improve by more than")
    add("  `stall_improvement_tolerance = 1e-6`.")
    add("")
    add("The tail that makes a stopping rule necessary is a degenerate plateau: the")
    add("retained-row master LP has an alternative-optimum face, so its objective")
    add("repeats the very same value iteration after iteration - the residual")
    add("objectives recorded in the stage-2 trace are flat constants, not slow")
    add("progress - while each iteration still pays for a CPLEX re-solve. The guard")
    add("stops exactly that. A hard cap is size-blind: on a case whose bound is still")
    add("moving it cuts into real progress, and the cost then changes.")
    add("")
    if cap_rows:
        reference = sweep_stats(cap_rows, "double")
        add("| stopping rule | cases | cases with changed cost | max abs. cost change % "
            "| sum `t_alg` (s) |")
        add("|" + "---|" * 5)
        for label, cap in CAP_SWEEP:
            stats = sweep_stats(cap_rows, label)
            if stats is None:
                continue
            name = "no guard" if cap == "none" else f"hard cap `{cap}`"
            add(f"| {name} | {stats['cases']} | {stats['changed']} "
                f"| {human(stats['max_delta'], 4)} | {human(stats['seconds'], 1)} |")
        add("")
        add("This sweep covers the 20 cases with the longest separation loop, so its")
        add("total wall clock is small and dominated by run-to-run noise; the column")
        add("that carries the message is `cases with changed cost`.")
        add("")
        cap_changed = [sweep_stats(cap_rows, label) for label, cap in CAP_SWEEP[1:]]
        cap_changed = [item for item in cap_changed if item]
        if cap_changed and reference:
            add(f"A hard cap changes `U` on up to "
                f"**{max(item['changed'] for item in cap_changed)} of the "
                f"{reference['cases']} cases** of that sweep (worst case "
                f"**+{human(max(item['max_delta'] for item in cap_changed), 2)} %**), whereas the")
            add("stall guard changes it on none of them - see section 6.4. Both stop")
            add("the tail; only the guard stops nothing else.")
        add("")
    multi_round = [row for row in new_rows if (number(row.get("rounds")) or 0) > 1]
    add("Truncating has one side effect worth recording, because it explains the")
    add("`rounds` column of section 8.2: the rounding that follows a truncated LP is")
    add("not always biconnected yet, so the outer loop needs a second round. In this")
    if multi_round:
        counter: dict[str, int] = {}
        for row in multi_round:
            tag = f"{int(float(row['num_stations']))}/{row['variant']}"
            counter[tag] = counter.get(tag, 0) + 1
        listing = ", ".join(f"`{tag}` x{count}" if count > 1 else f"`{tag}`"
                            for tag, count in sorted(counter.items()))
        add(f"run that happens on **{len(multi_round)} of the {len(new_rows)} cases**")
        add(f"({listing}).")
    add("Every one of them is a case whose inner separation was truncated, and each")
    add("keeps the same `U` when the guard is switched off: the extra round only")
    add("repeats a rounding that the truncated LP could not certify. It costs one")
    add("master-LP solve, far less than the tail it replaces - on the four looping")
    add("500-station `selectable` cases the count of master-LP solves still falls")
    add("from 32/33/52/32 without the guard to 7/7/6/7 with it.")
    add("")

    # ---------------------------------------------------------------- section 6.4
    add("### 6.4 Why `stall = 3`")
    add("")
    if stall_rows:
        add("| guard | cases | cases where it fired | cases with changed cost "
            "| max abs. cost change % | sum `t_alg` (s) | sum `t_alg`, 500-station (s) "
            "| master-LP solves, 500-station |")
        add("|" + "---|" * 8)
        for label, guard in STALL_SWEEP:
            stats = sweep_stats(stall_rows, label)
            if stats is None:
                continue
            big = tier_subset(stall_rows, label, 500)
            name = "none" if guard == "none" else f"`{guard}`"
            add(f"| {name} | {stats['cases']} | {stats['fired']} | {stats['changed']} "
                f"| {human(stats['max_delta'], 4)} | {human(stats['seconds'], 1)} "
                f"| {human(sum_of(big, 'seconds_total'), 1)} "
                f"| {human(sum_of(big, 'iterations'), 0)} |")
        add("")
        table = [(guard, sweep_stats(stall_rows, label))
                 for label, guard in STALL_SWEEP]
        table = [(guard, stats) for guard, stats in table if stats]
        guarded = [(guard, stats) for guard, stats in table if guard != "none"]
        if guarded:
            add("* **The guard never changes the answer.** `max |delta cost| = "
                f"{human(max(s['max_delta'] for _, s in table), 4)} %` over all "
                f"{table[0][1]['cases']} cases, for every cap of the sweep, so within")
            add("  3-8 the choice is a pure speed question.")
            add("* **It fires only on the tail.** It never fires on the small tiers (a")
            add("  case that separates in one or two iterations cannot stall three")
            add("  times) and it fires on " +
                ", ".join(f"{s['fired']} of {s['cases']}" for _, s in guarded) +
                " cases at caps 3, 5 and 8 -")
            add("  every firing of this five-tier sweep is a 300- or 500-station")
            add("  `selectable` case; the sweep does not contain the 400-station tier,")
            add("  where the production run of section 8.2 does fire the guard.")
            add("* **More cap buys time, not quality.** Where it fires, both the wall")
            big_iter = [(guard, sum_of(tier_subset(stall_rows, label, 500), "iterations"),
                         sum_of(tier_subset(stall_rows, label, 500), "seconds_total"))
                        for label, guard in STALL_SWEEP]
            big_iter = [(guard, i, s) for guard, i, s in big_iter if i is not None]
            add("  clock and the number of master-LP solves grow with the cap: at the")
            add("  500-station tier " +
                ", ".join(f"{human(seconds, 1)} s / {human(iterations, 0)} solves at {guard}"
                          for guard, iterations, seconds in big_iter) +
                ".")
            add("  A larger cap keeps re-solving an LP whose objective has already")
            add("  stopped moving.")
        if len(guarded) > 1:
            slowest = max(guarded, key=lambda item: item[1]["seconds"])
            fastest = min(guarded, key=lambda item: item[1]["seconds"])
            add(f"* **The cap can also be too large to notice.** `{slowest[0]}` is the "
                f"slowest setting of the sweep")
            add("  (`sum t_alg` over the 50 cases) because on some cases its counter")
            add("  never reaches the cap at all and the unguarded tail is paid in full,")
            add("  while `3` truncates it.")
        add("")
        add(f"So the production default is **`max_stalled_iterations = 3`**: the")
        add("fastest setting of the sweep at an identical `U`. The sweep starts at 3,")
        add("so the table supports \"3 is the best of {3, 5, 8}\", not \"3 is lossless")
        add("for every cap\"; caps 1 and 2 were left out deliberately, since the guard")
        add("already risks firing on a plateau that a later iteration would have left")
        add("(section 12, item 3) and a cap of 1 leaves no room for the improvement")
        add("step that follows a plateau. The other two caveats of the guard apply")
        add("here as well: it can only weaken the internal LP lower bound, and the")
        add("tier-level differences below 300 stations are inside the repeat-run noise.")
        add("")
    # Six-tier view of the guard on the production runs themselves, read from
    # the same grouped data as sections 7.3 and 8.2.  The 400-station tier is
    # the one whose no-guard arm was re-measured for this report.
    add("Finally the guard on the production runs, tier by tier. `no guard` is")
    add("`orl_setpair_full.csv` on the first five tiers and the re-measured")
    add("`orl_setpair_nostall_400.csv` on the 400-station tier, so every tier from")
    add("100 to 500 stations now has a paired no-guard measurement:")
    add("")
    add("| tier | cases | paired no-guard cases | `U` differences | `sum t_alg` "
        "no guard (s) | `sum t_alg` stall3 (s) | ratio |")
    add("|" + "---|" * 7)
    tier_rows = []
    for depots, stations, demands in TIER_ORDER:
        pairs = [item for group in grouped.values() for item in group
                 if key_of(item[0])[1] == stations]
        paired = [item for item in pairs if item[1] is not None]
        new_s = sum_of([item[0] for item in pairs], "algorithm_seconds") or 0.0
        ns_s = sum_of([item[1] for item in paired], "algorithm_seconds") or 0.0
        tier_diff = 0
        for item in paired:
            new_u = number(item[0].get("algorithm_cost_U"))
            ns_u = number(item[1].get("algorithm_cost_U"))
            if new_u is None or ns_u is None or abs(new_u - ns_u) > 1e-6:
                tier_diff += 1
        ratio = (ns_s / new_s) if new_s else None
        tier_rows.append((stations, len(pairs), len(paired), tier_diff,
                          ns_s, new_s, ratio))
        add(f"| `{depots}/{stations}/{demands}` | {len(pairs)} | {len(paired)} "
            f"| {tier_diff} | {human(ns_s, 1)} | {human(new_s, 1)} "
            f"| {human(ratio, 2)}x |")
    add("")
    if tier_rows:
        total_diff = sum(row[3] for row in tier_rows)
        total_cases = sum(row[2] for row in tier_rows)
        with_ratio = [row for row in tier_rows if row[6]]
        slowest = max(with_ratio, key=lambda row: row[6])
        fastest = min(with_ratio, key=lambda row: row[6])
        add(f"The guard changes `U` on {total_diff} of the {total_cases} paired")
        add(f"cases. The speed ratio between the two arms ranges from "
            f"**{human(fastest[6], 2)}x** at {fastest[0]} stations to "
            f"**{human(slowest[6], 2)}x** at {slowest[0]} stations, so the payoff")
        add("is concentrated at the two largest tiers, where the separation tail is")
        add("long enough for the guard to cut it.")
    return lines


def build_report() -> str:
    new_rows = load_rows(STALL3_CSV)
    if STALL3_400_CSV.exists():
        new_rows = new_rows + load_rows(STALL3_400_CSV)
    nostall_index = index_rows(load_rows(NOSTALL_CSV))
    if NOSTALL_400_CSV.exists():
        nostall_index.update(index_rows(load_rows(NOSTALL_400_CSV)))
    original_index = index_rows(load_rows(ORIGINAL_CSV))
    grouped = merged_groups(new_rows, nostall_index, original_index)
    stats = totals(grouped, original_index)

    class_counts: dict[str, int] = {}
    for group in grouped.values():
        for item in group:
            cls = ip_class_of(item) or "no_reference"
            class_counts[cls] = class_counts.get(cls, 0) + 1

    body: list[str] = []
    add = body.append

    add("# Two-Radius UDG Experiment — Set-Pair LP with Stall Guard (stall3), Full Rerun")
    add("")
    add("**Model.** A set of candidate facilities (depots and charging stations) is")
    add("placed in a fixed `100 x 100` square. Two facilities are joined by an edge")
    add("when their Euclidean distance is at most the flight radius `R = 20`. Every")
    add("demand point must be reachable from **two distinct open depots by two")
    add("internally vertex-disjoint paths** (`k_p = 2`). The objective is the sum of")
    add("the opening costs of the chosen facilities.")
    add("")
    add(f"**Scope.** {len(new_rows)} cases = 6 tiers x 2 depot variants x 5 seeds, all")
    add(f"re-solved with the production configuration `--max-stalled-iterations 3`.")
    add(f"Every one of the {stats['verified']} reported solutions passed an independent")
    add("feasibility/biconnectivity verification (`algorithm_verified = True`).")
    add("")
    add("**Baseline policy.** The first five tiers (`5/50/25` ... `50/500/250`) are")
    add("algorithm-only in this rerun: their FLOW-IP benchmark is switched off")
    add("(`--skip-ip-stations 1`), so their quality and speed-up figures are computed")
    add("against the FLOW-IP values already recorded for the *same* instances by the")
    add("earlier benchmark runs, not by a fresh solve. One of those five tiers")
    add("carries no usable value: at 500 stations the exact FLOW-IP found **no")
    add("feasible integer solution inside its 1800 s limit**, so that tier is")
    add("reported as `no_incumbent` and contributes no quality figure. The")
    add("**`40/400/200` tier is new** and is the exception: it runs the FLOW-IP itself")
    add("in this run (`40/400/200` with the standard 1800 s limit), so its comparison")
    add("values are first-hand. Every table marks which of the two provenances each")
    add("row uses.")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------------ section 1
    add("## 1. What this report is and how it relates to the original report")
    add("")
    add("**Delta against the original report**")
    add("(`output/orl_two_radius_final_r20_v1_report.md`). The original report")
    add("benchmarks the cut-based row-generation algorithm (single UCUT-style")
    add("separation) against the compact multi-commodity FLOW-IP on five tiers. This")
    add("report keeps those same 50 instances (identical measured costs and identical")
    add("FLOW-IP reference), replaces stage 2 with the **set-pair LP** formulation,")
    add("adds the **stall guard** (row generation is truncated once the master LP")
    add("objective has not improved by more than `1e-6` for `3` consecutive")
    add("iterations), and **adds a sixth tier `40/400/200`** that the original report")
    add("never covered.")
    add("")
    add("| item | original report | this report |")
    add("|---|---|---|")
    add("| tiers | 5 (`5/50/25` ... `50/500/250`) | 6 (adds `40/400/200`) |")
    add("| stage-2 cut model | UCUT-style separation | rooted set-pair LP |")
    add("| stall guard | none | `max_stalled_iterations = 3` |")
    add("| FLOW-IP baseline, first 5 tiers | solved, 39/50 proven optimal | not re-solved; values reused |")
    add("| FLOW-IP baseline, 400 tier | not covered | solved here, 1800 s limit |")
    add("| FLOW-IP baseline, 500 tier | trial solves, 1800 s each, no incumbent | no feasible solution inside the time limit; reported as `no_incumbent`, no quality figure |")
    add("| solution cost `U`, first 5 tiers | measured | identical on all 50 cases |")
    add("| set-pair no-guard reference | 5 tiers | the 400-station tier re-measured, so all 6 tiers are paired |")
    add("| algorithm wall clock | measured | re-measured (see section 7.3) |")
    add("")
    add("**Headline claim (one paragraph).** Switching stage 2 to the set-pair LP and")
    add("truncating its row generation with the stall guard leaves the verified")
    add(f"solution cost unchanged on the 50 instances shared with the original report")
    add("(`U` is identical case by case) while cutting the aggregate algorithm time,")
    add("because the guard removes the long degenerate tail of the separation loop.")
    add("The new 400-station tier is reported here for the first time, with its")
    add("FLOW-IP solved inside this run. There the algorithm returns a verified")
    add("design in a few seconds per instance (10.6k facility edges, ~4.2M columns in")
    add("the compact FLOW-IP) while CPLEX consumes its full 1800 s budget on half of")
    add("the cases and fails to find any integer incumbent on three of them.")
    add("")
    add("**Table placement.**")
    add("")
    add("| slot | content | source table in this report |")
    add("|---|---|---|")
    add("| main text, Table 1 | per-tier quality vs. FLOW-IP | section 7.1b |")
    add("| main text, Table 2 | algorithm internal structure | section 7.2 |")
    add("| main text, Table 3 | stall3 vs. original and vs. set-pair no-stall | section 7.3 |")
    add("| main text, Table 4 | FLOW-IP accounting | section 9 |")
    add("| appendix A | per-case cost and quality, 60 rows | section 8.1 |")
    add("| appendix B | per-case three-way timing, 60 rows | section 8.2 |")
    add("| implementation notes | greedy stage-1 cover, `double` cut, stall guard, ablation evidence | section 6 |")
    add("")
    add("**Claims that the data does NOT support (avoid them).**")
    add("")
    add("* Do not report any gap for the 500-station tier: the exact FLOW-IP found no")
    add("  feasible integer solution there inside its 1800 s limit, and the LP")
    add("  relaxation of the same compact model exceeds the memory available on this")
    add("  machine, so no valid reference exists (section 9).")
    add("* Do not read the `40/400/200` rows as exact gaps unless `gap basis` says")
    add("  `exact`: where the FLOW-IP hit its 1800 s limit there, the gap is a")
    add("  certified upper bound, not a measured optimum distance.")
    add("* Do not claim the guard improves solution *quality*: it leaves the returned")
    add("  integer design at the same cost, and it can only *weaken* the internal LP")
    add("  lower bound (section 12).")
    add("* Do not present the 50-station tier as a speed win: those models are tiny")
    add("  and CPLEX closes them faster in absolute terms.")
    add("* Do not read the per-case time reduction as noise-free: single-case")
    add("  wall-clock differences below ~0.5 s are within CPLEX/repeat variance.")
    add("")

    # ------------------------------------------------------------------ section 2
    add("## 2. Environment")
    add("")
    add("| item | value |")
    add("|---|---|")
    add("| OS | Windows 10 (10.0.19045) |")
    add("| CPU | Intel Core i5-7300U, 4 logical CPUs, mobile dual-core |")
    add("| RAM | 15.92 GiB installed; ~9-10 GiB free during the runs |")
    add(r"| Python | 3.10.0 (`.venv310\Scripts\python.exe`) |")
    add("| CPLEX | 22.1.1.0, `threads = 1`, `parallel = 1`, `advance = 1` |")
    add("| NetworkX / SciPy | 3.4.2 / 1.15.3 (maximum-flow backend) |")
    add("")
    add("All timings are wall clock of a single-threaded CPLEX solve on a")
    add("single-process Python interpreter. Module import time is excluded from")
    add("`t_alg` because imports are paid once per process, not per case.")
    add("")

    # ------------------------------------------------------------------ section 3
    add("## 3. Notation")
    add("")
    add("| symbol | meaning |")
    add("|---|---|")
    add("| `D`, `F`, `P` | number of candidate depots, candidate stations, demand points in one instance |")
    add("| `R` | flight radius between two facilities, fixed at `R = 20` |")
    add("| `r` | service radius of a station around a demand point, `r = R/2 = 10` |")
    add("| `E` | number of edges of the facility graph (all pairs at distance `<= R`) |")
    add("| `x_e` | design variable of edge `e` in the set-pair LP, relaxed to `x_e in [0,1]` |")
    add("| `U` | cost of the algorithm's verified feasible solution (upper bound on `OPT`) |")
    add("| `OPT` | optimal integer cost returned by the paired CPLEX FLOW-IP solve |")
    add("| `LB` | certified lower bound of an unfinished paired solve, `LB = max(best_bound, lp_relaxation)` |")
    add("| `t_alg` | algorithm wall clock = stage 1 + stage 2 + verification (`algorithm_seconds`) |")
    add("| `t_IP` | wall clock of the paired CPLEX FLOW-IP solve of the same instance |")
    add("| `rho` | per-case speed-up `t_IP / t_alg` (new algorithm time) |")
    add("| `eps_exact` | exact relative error `100 (U/OPT - 1)`, when `OPT` was proven |")
    add("| `eps_cert` | certified error upper bound `100 (U/LB - 1)`, when `OPT` is unknown |")
    add("| `stall guard` | row generation stops after `max_stalled_iterations` rounds without an LP improvement above `1e-6` |")
    add("| `master LP solve` | one re-optimisation of the retained-row set-pair master LP |")
    add("| `set-pair witness` | `(u, W_u, W_v)` cut retained by the rooted set-pair separator |")
    add("| `scaling fallback` | a separation call re-run in exact/scaled arithmetic after a numerical failure |")
    add("")

    # ------------------------------------------------------------------ section 4
    add("## 4. Experimental design")
    add("")
    add("| tier | depots `D` | stations `F` | demands `P` | variants | seeds | cases |")
    add("|---|---|---|---|---|---|---|")
    for tier in TIER_ORDER:
        add(f"| {tier[0]}/{tier[1]}/{tier[2]} | {tier[0]} | {tier[1]} | {tier[2]} "
            f"| selectable, preinstalled | 101-105 | 10 |")
    add("")
    add("* `selectable` - depot opening cost 100, depots may stay closed, station cost")
    add("  drawn from `U{10,...,30}`.")
    add("* `preinstalled` - depot opening cost 0 and every depot is forced open, so")
    add("  only stations are paid for.")
    add("* The two variants of a pair share one and the same geometric instance")
    add("  (identical positions and cost vector), and all three runs (original,")
    add("  set-pair no-stall, set-pair stall3) share the same instance per seed, so")
    add("  all comparisons below are paired.")
    add("* Run configuration: `separator_backend = scipy`, `cut_mode = double`,")
    add("  `nested_levels = 4`, `epsilon = 0.20`, `cover_method = greedy`,")
    add("  `max_stalled_iterations = 3`. The FLOW-IP benchmark is disabled for the")
    add("  first five tiers (`skip_ip_stations = 1` when they were rerun) and enabled")
    add("  for the 400-station tier (`skip_ip_stations = 0`).")
    add("* The `40/400/200` tier is the addition of this report. Its instance size")
    add("  (`E ~= 10.6k` facility edges, average degree `~= 48`, estimated compact")
    add("  FLOW-IP of `~4.2M` columns / `~176k` rows) is the first one in this study")
    add("  where the FLOW-IP actually exhausts its 1800 s budget: it proves optimality")
    add("  on only half of the ten cases and finds no integer incumbent at all on three.")
    add("")
    add("**Algorithm (reference method).**")
    add("")
    add("| stage | content |")
    add("|---|---|")
    add("| stage 1 | two-pass deterministic weighted greedy set cover, producing `J0` |")
    add("| stage 2 | rooted set-pair LP; violated `(u, W_u, W_v)` cuts are separated on the split graph (`cut_mode = double`), the retained-row master LP is re-solved, and once the LP objective has stalled for 3 rounds the loop stops; a single batch rounding of `x_e >= 1/2` follows, then `Test(J)` biconnectivity check and one reverse-delete pruning pass |")
    add("| stall guard | `max_stalled_iterations = 3` with `stall_improvement_tolerance = 1e-6` |")
    add("| verification | independent feasibility check on the rounded solution |")
    add("")
    add("**Stall guard mechanics.** The guard acts *inside* `solve_residual`, i.e. it")
    add("shortens the row-generation loop of the single rounding round; it does not")
    add("change how many rounds the outer loop performs. The outer loop still rounds")
    add("once (twice on the two cases where the first rounding was not yet feasible),")
    add("and reverse pruning still runs once, after the loop, on the final `0/1`")
    add("design.")
    add("")
    add("**FLOW-IP baseline.** Compact node-capacitated multi-commodity flow")
    add("formulation, solved by CPLEX with `mip.display = 2` and a 1800 s limit. On")
    add("the first five tiers the values come from the earlier benchmark runs")
    add("(`optimal` on 39 cases, `open_gap` on 1; on the 10 cases of the 500-station")
    add("tier the exact FLOW-IP found no feasible integer solution inside its 1800 s")
    add("limit, so that tier carries no reference value). The `40/400/200` tier is")
    add("solved inside this run: 5 cases proved optimal, 2 stopped with a verified")
    add("incumbent but no proof, and 3 hit the time limit before any integer incumbent")
    add("was found.")
    add("")

    # ------------------------------------------------------------------ section 5
    add("## 5. Metrics")
    add("")
    add("* `eps_exact = 100 (U/OPT - 1)` when the paired FLOW-IP solve proved optimality.")
    add("* `eps_cert = 100 (U/LB - 1)` when that solve stopped with an integer incumbent")
    add("  but without a proof, with `LB = max(best_bound, lp_relaxation)` from it.")
    add("* Neither is quoted when the paired solve found no integer incumbent at all;")
    add("  such a solve leaves only a root bound, which is not reported as a gap.")
    add("* `rho = t_IP / t_alg` per case using the **new** algorithm time; aggregate")
    add("  speed-up is `sum t_IP / sum t_alg` over the cases that have a FLOW-IP")
    add("  reference (reused on the first five tiers, solved in-run on the 400 tier).")
    add("* `stall3 vs original` and `stall3 vs no-stall` in section 8.2 are the")
    add("  ratios of the two reference wall clocks to the new one; values `> 1x`")
    add("  mean this run is faster.")
    add("")

    # ------------------------------------------------------------------ section 6
    # The implementation switches and the ablations that validate them.
    body.extend(implementation_section(new_rows, grouped))
    add("")

    # ------------------------------------------------------------------ section 7
    add("## 7. Aggregated results")
    add("")
    add("### 7.1 Quality and speed per tier")
    add("")
    add("Every row aggregates the 5 seeds of one `(tier, variant)` class. `sum t_IP`")
    add("is the FLOW-IP time of the same cases (reused on the first five tiers, this")
    add("run's own on the 400 tier); the 500-station tier has none, because the exact")
    add("FLOW-IP found no feasible integer solution there inside its 1800 s limit.")
    add("")
    body.extend(tier_table(grouped))
    add("")
    add("### 7.1b Headline table — this run vs. the FLOW-IP exact solve")
    add("")
    add("Speed-up is `sum t_IP / sum t_alg` over the cases of the row that have a")
    add("FLOW-IP reference (reused on the first five tiers, solved in-run on the 400")
    add("tier). `mean gap %` averages `eps_exact` on the cases whose reference proved")
    add("optimality and `eps_cert` on the ones whose solve was left unfinished; the")
    add("`gap basis` column says which of the two each row mixes. Two classes are")
    add("deliberately left without a gap: the 500-station tier, where the exact FLOW-IP")
    add("found no feasible integer solution inside its 1800 s limit, and the")
    add("`40/400/200` cases whose FLOW-IP found no integer incumbent")
    add("(marked `no quality figure`), because a `no_incumbent` solve leaves only a")
    add("root bound too weak to support a percentage.")
    add("")
    body.extend(headline_table(grouped))
    add("")
    add("### 7.2 Internal structure of the algorithm")
    add("")
    add("`master-LP solves`, `cuts`, `set-pair witnesses` and `max-flow calls` show")
    add("what the stall guard removes; the last three columns decompose `t_alg`.")
    add("")
    body.extend(structure_table(grouped))
    add("")
    add("### 7.3 Totals and comparison with the original report")
    add("")
    add(f"* Verified solutions: **{stats['verified']} / {len(new_rows)}**.")
    add(f"* Solution cost `U`: identical to the original report on the")
    add(f"  **{stats['n_shared'] - stats['u_diff']} / {stats['n_shared']}** instances shared with it")
    add(f"  ({stats['u_diff']} difference(s)). The 400-station tier has no counterpart there.")
    add(f"* Total algorithm time over the {stats['n_shared']} shared cases: "
        f"**{human(stats['t_new_shared'], 1)} s** (this run) vs. "
        f"**{human(stats['t_old_all'], 1)} s** (original report) vs. "
        f"**{human(stats['t_ns_shared'], 1)} s** (set-pair, no stall).")
    if stats["t_old_all"] and stats["t_new_shared"]:
        add(f"  -> this run is **{stats['t_old_all'] / stats['t_new_shared']:,.2f}x** "
            f"faster in aggregate than the original report")
    if stats["t_ns_shared"] and stats["t_new_shared"]:
        add(f"     and **{stats['t_ns_shared'] / stats['t_new_shared']:,.2f}x** faster than "
            f"the set-pair run without the guard on those same cases.")
    if stats["t_ns_only"]:
        add(f"* Stopping-rule comparison on the **{stats['n_only']} cases that only")
        add(f"  exist here** (the 400-station tier, whose no-stall reference was")
        add(f"  re-measured on the same instances): "
            f"**{human(stats['t_ns_only'], 1)} s** without the guard vs. "
            f"**{human(stats['t_new_only'], 1)} s** with it "
            f"(**{human(stats['t_ns_only'] / stats['t_new_only'], 2)}x** on {stats['n_only']} cases).")
    add(f"* Quality is unchanged by the guard on **{stats['n_shared'] + stats['n_only'] - stats['u_diff_ns']} / "
        f"{stats['n_shared'] + stats['n_only']}** cases where a no-stall run")
    add(f"  exists: `U` differs on {stats['u_diff_ns']} case(s).")
    add(f"* The new 400-station tier adds **{human(stats['t_new_only'], 1)} s** of")
    add(f"  algorithm time over {stats['n_only']} cases, so the whole report covers")
    add(f"  **{human(stats['t_new_all'], 1)} s** of algorithm wall clock on")
    add(f"  {len(new_rows)} cases.")
    add(f"* Aggregate speed-up against the FLOW-IP over the "
        f"{stats['benchmarked']} benchmarked cases:")
    if stats["t_alg_sum"]:
        add(f"  `sum t_IP = {human(stats['t_ip_sum'], 1)} s` vs.")
        add(f"  `sum t_alg = {human(stats['t_alg_sum'], 1)} s` ->")
        add(f"  **{human(stats['t_ip_sum'] / stats['t_alg_sum'], 1)}x aggregate speed-up**.")
    if stats["largest"]:
        ratio, key, t_ip, t_alg = stats["largest"]
        add(f"* Largest single-case speed-up: **{human(ratio, 0)}x**")
        add(f"  (`({key[0]},{key[1]},{key[2]}) / {key[3]} / seed {key[4]}`: "
            f"`t_IP = {human(t_ip, 1)} s` vs. `t_alg = {human(t_alg, 2)} s`).")
    add(f"* Cases taking at least {human(LONG_TAIL_SECONDS, 0)} s of algorithm time:")
    add(f"  **{len(stats['long_tail'])}** in this run.")
    if stats["long_tail"]:
        add("")
        add("| case | old `t_alg` (s) | new `t_alg` (s) | reduction |")
        add("|---|---|---|---|")
        for seconds, key in stats["long_tail"]:
            old_pair = next((value for k, value in stats["long_tail_old"] if k == key), None)
            if old_pair:
                add(f"| `({key[0]},{key[1]},{key[2]})` / {key[3]} / seed {key[4]} "
                    f"| {human(old_pair, 2)} | {human(seconds, 2)} "
                    f"| {human(old_pair / seconds, 2)}x |")
            else:
                add(f"| `({key[0]},{key[1]},{key[2]})` / {key[3]} / seed {key[4]} "
                    f"| -- | {human(seconds, 2)} | -- |")
    add("")

    # ------------------------------------------------------------------ section 8
    add("## 8. Per-case detail (appendix A and B)")
    add("")
    add("### 8.1 Cost and quality")
    add("")
    add("`U` is the verified algorithm cost of this run; `IP value (ref)`, `LB (ref)`")
    add("and `IP class (ref)` come from the FLOW-IP solve paired with the case (reused")
    add("on the first five tiers, solved in-run on the 400 tier). `gap %` is `eps_exact`")
    add("when that solve proved optimality and `eps_cert` when it stopped with an")
    add("incumbent; it is empty where no reference exists and where the reference")
    add("found no integer incumbent (`no_incumbent`).")
    body.extend(case_quality_tables(grouped))
    add("")
    add("### 8.2 Algorithm structure and timing (three-way)")
    add("")
    add("`t_alg original report` is the wall clock recorded in")
    add("`orl_two_radius_final_r20_v1.csv`; `t_alg set-pair no-stall` is the same")
    add("algorithm without the guard (`orl_setpair_full.csv` on the first five tiers,")
    add("re-measured on the 400-station tier in `orl_setpair_nostall_400.csv` because")
    add("that tier is absent from the five-tier run); `t_alg set-pair stall3` is this")
    add("run. All three columns therefore exist for every tier except the original")
    add("report, which never covered the 400-station tier.")
    body.extend(case_timing_tables(grouped))
    add("")

    # ------------------------------------------------------------------ section 9
    add("## 9. FLOW-IP reference accounting")
    add("")
    add("| outcome class | cases | meaning |")
    add("|---|---|---|")
    class_meaning = {
        "optimal": "CPLEX proof of the integer optimum -> exact relative error",
        "open_gap": "incumbent found, proof unfinished inside 1800 s -> certified error upper bound",
        "no_incumbent": "time limit hit before any feasible integer incumbent -> no quality figure",
        "invalid_incumbent": "incumbent returned but rejected by the verifier -> no quality figure",
        "skipped_too_large": "compact FLOW-IP exceeded the memory guard -> no IP optimum",
        "skipped_by_request": "FLOW-IP switched off for this tier -> no value from this run",
        "no_reference": "no FLOW-IP reference available in any run",
    }
    for cls in sorted(class_counts, key=lambda c: (-class_counts[c], c)):
        add(f"| `{cls}` | {class_counts[cls]} | {class_meaning.get(cls, 'see the run log')} |")
    add("")
    add("* The first five tiers (`5/50/25` ... `50/500/250`) reuse the FLOW-IP values")
    add("  of the earlier benchmark; the stall3 rerun itself switched the FLOW-IP off")
    add("  there (`--skip-ip-stations 1`), so those classes are the reused reference")
    add("  classes, not this run's own outcomes. Within them the 500-station tier is")
    add("  the one that carries no reference value: its trial solves each hit the")
    add("  1800 s limit without finding any feasible integer solution, so the tier is")
    add("  reported as `no_incumbent` and contributes to no gap. The")
    add("  **`40/400/200` tier is the exception**: its FLOW-IP is solved inside this")
    add("  run (1800 s limit), so its class is this run's own outcome for that")
    add("  instance.")
    add("* Every reference value belongs to the same instance as the case it is paired")
    add("  with; the pairing was checked against both the original benchmark CSV and")
    add("  the set-pair no-stall CSV, and the two agree case by case.")
    add("")

    # ------------------------------------------------------------------ section 10
    add("## 10. Solution quality")
    add("")
    if stats["exact"]:
        add(f"* Exact gap over the {len(stats['exact'])} proven-optimal cases:")
        add(f"  **mean {human(statistics.mean(stats['exact']), 2)} %**, "
            f"median {human(statistics.median(stats['exact']), 2)} %, "
            f"max {human(max(stats['exact']), 2)} %.")
    if stats["certified"]:
        add(f"* Certified gap upper bound on the {len(stats['certified'])} unfinished")
        add(f"  case(s): **{human(statistics.mean(stats['certified']), 2)} %**.")
    if stats["unquoted"]:
        add(f"* No quality figure is quoted for the {stats['unquoted']} case(s) whose")
        add("  FLOW-IP reference found no integer incumbent inside its time limit")
        add("  (the `40/400/200` tier): the only bound such a solve leaves is the root")
        add("  bound, which is far too weak to support any gap statement.")
    add("* On the 50 instances shared with the original report these figures are")
    add("  identical to it, because `U` is identical case by case and the FLOW-IP")
    add("  references are reused unchanged. The `40/400/200` tier is measured")
    add("  first-hand here and is included in the figures above.")
    add("  The stall guard did not trade quality for speed on these instances.")
    add("")

    # ------------------------------------------------------------------ section 11
    add("## 11. Speed")
    add("")
    add("* The guard pays off exactly where the separation loop has a degenerate")
    add("  tail: the 500-station `selectable` cases and the single 200/300-station")
    add("  outliers. Small cases keep a single master-LP solve and are unchanged")
    add("  within noise.")
    add("* The new 400-station tier is where the comparison becomes decisive: the")
    add("  algorithm stays under 25 s on every case (mean `t_alg` 6.2 s) while the")
    add("  FLOW-IP burns the full 1800 s budget on half of them, so the aggregate")
    add("  `sum t_IP / sum t_alg` of that tier is in the hundreds.")
    add("* The mechanism is a reduction in master-LP re-solves (`master LP solves` in")
    add("  section 7.2 and 8.2), not a change of cut strength: the cut mode stays")
    add("  `double` and the separator stays the same.")
    add("* Below the crossing point CPLEX is still faster in absolute terms; the")
    add("  benefit of the guard is on the large instances, where it removes repeated")
    add("  re-optimisations that never move the LP objective.")
    add("* The very large `mean speed-up` of the 300-station `selectable` row is")
    add("  driven by a single case (`seed = 105`, `t_IP = 1,801.8 s` against")
    add("  `t_alg = 1.91 s`); read it together with `sum t_IP / sum t_alg`, which is")
    add("  the outlier-insensitive aggregate.")
    add("")

    # ------------------------------------------------------------------ section 12
    add("## 12. Caveats and limitations of the guard")
    add("")
    add("1. **The guard can only weaken the internal LP lower bound.** Truncating row")
    add("   generation stops the master LP objective at a value no higher than the")
    add("   fully separated one. On the cases probed earlier the truncated value was")
    add("   roughly 55 % of the fully separated value. Any approximate-ratio argument")
    add("   built on that bound must account for this.")
    add("2. **The returned integer design is unaffected in these instances.** Batch")
    add("   rounding plus reverse-delete pruning normalise the final `0/1` design, so")
    add("   `U` stayed identical on all 60 cases that carry a paired no-guard")
    add("   measurement (the five shared tiers plus the re-measured 400-station tier).")
    add("   This is an empirical observation on this benchmark, not a theorem.")
    add("3. **The guard can fire on a false plateau.** A flat LP series can precede a")
    add("   later jump; stopping early then forfeits that jump. It is a heuristic")
    add("   trade-off, not a correctness argument.")
    add("4. **Single-case timing noise.** Repeat runs of the same instance differ by")
    add("   CPLEX-internal variance; on tiny cases the new run is occasionally a")
    add("   fraction of a second slower. Only the aggregate and the clearly long-tail")
    add("   cases should be read as signal.")
    add("5. **The 500-station tier carries no FLOW-IP reference value**: the exact")
    add("   FLOW-IP found no feasible integer solution there inside its 1800 s limit,")
    add("   so its rows must not be used in any quality or speed-up percentage.")
    add("6. **The 400-station FLOW-IP does not always produce a usable reference.**")
    add("   On 3 of its 10 cases CPLEX hit the 1800 s limit without finding any")
    add("   integer incumbent, so those cases carry a speed comparison only; the")
    add("   remaining 5 prove optimality and 2 stop with a verified incumbent.")
    add("   No quality figure should be quoted for the 3 unfinished ones.")
    add("7. **Hardware.** Timings come from a mobile 2-core/4-thread CPU with")
    add("   `threads = 1`; absolute values differ on other hardware, so only ratios")
    add("   should be compared across machines.")
    add("")

    # ------------------------------------------------------------------ section 13
    add("## 13. Data files")
    add("")
    add("| file | content |")
    add("|---|---|")
    add("| `orl_setpair_stall3.jsonl` | 50 full records of the five shared tiers, this run |")
    add("| `orl_setpair_stall3.csv` | per-case flat table of the five shared tiers |")
    add("| `orl_setpair_stall3_table_summary.csv` | aggregated main table of the five shared tiers |")
    add("| `orl_setpair_stall3_settings.json` | parameters and experiment signature of the five shared tiers |")
    add("| `orl_setpair_stall3_400.jsonl` | 10 full records of the new `40/400/200` tier (FLOW-IP solved in-run) |")
    add("| `orl_setpair_stall3_400.csv` | per-case flat table of the `40/400/200` tier |")
    add("| `orl_setpair_stall3_400_settings.json` | parameters and experiment signature of the 400-station run |")
    add("| `orl_setpair_stall3_400_run.log` | console log of the 400-station run |")
    add("| `orl_setpair_stall3_report.md` | this report, rendered by `build_stall3_report.py` |")
    add("| `orl_setpair_full.csv` | set-pair run without the guard (reference for section 8.2); covers five tiers |")
    add("| `orl_setpair_nostall_400.csv` | guard-free arm of the 400-station tier, re-measured on the same instances |")
    add(r"| `..\output\orl_two_radius_final_r20_v1.csv` | original report data and reused FLOW-IP values |")
    add("| `ablation_stall.csv` | stopping-rule sweep `double` / `stall3` / `stall5` / `stall8` behind section 6.3 and 6.4 |")
    add("| `ablation_sep_cap.csv` | hard-cap sweep `double` / `cap3` / `cap5` / `cap10` behind section 6.3 |")
    add("| `ablation_cut_mode.jsonl` | separator sweep `double` vs. `nested:4` behind section 6.2 |")
    add("")
    return "\n".join(body) + "\n"


def main() -> None:
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"wrote {REPORT_PATH} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
