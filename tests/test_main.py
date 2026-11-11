"""End-to-end tests of the command-line entry point.

A single small pre-registered tier is solved so the whole pipeline (instance
generation, both stages, the FLOW-IP baseline and every writer) is exercised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The smallest pre-registered tier: fast enough for a test, large enough for
# the generator to find a two-root-connected layout on the first tries.
PILOT_SCALE = "5,50,25"


def run_main(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )


def test_pilot_run_writes_every_artifact(tmp_path: Path) -> None:
    output = tmp_path / "pilot.jsonl"
    result = run_main(
        "--scales", PILOT_SCALE,
        "--seeds", "101",
        "--variants", "selectable",
        "--time-limit", "300",
        "--output", str(output),
    )
    assert result.returncode == 0, result.stderr

    jsonl_path = output
    csv_path = output.with_suffix(".csv")
    table_csv_path = output.with_name(output.stem + "_table_summary.csv")
    settings_path = output.with_name(output.stem + "_settings.json")
    for artifact in (jsonl_path, csv_path, table_csv_path, settings_path):
        assert artifact.exists(), f"missing artifact {artifact.name}"

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["instance"]["variant"] == "selectable"
    assert record["algorithm"]["verified"] is True
    assert record["algorithm"]["status"] == "feasible"
    assert record["ip"]["status_class"] in {
        "optimal", "open_gap", "no_incumbent"
    }
    # The paired fingerprint is recorded so two variants can be compared later.
    assert record["generator"]["paired_instance_sha256"]

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["expected_cases"] == 1
    assert settings["scales_depots_stations_demands"] == [[5, 50, 25]]

    table_rows = [
        line for line in table_csv_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(table_rows) == 2  # header plus the single aggregated group

    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["cases"] == 1

    # A second identical run refuses to overwrite the existing experiment, and
    # --resume recognises the already completed case instead of solving it again.
    refused = run_main("--scales", PILOT_SCALE, "--seeds", "101",
                       "--variants", "selectable", "--output", str(output))
    assert refused.returncode != 0
    assert "Output exists" in refused.stderr

    resumed = run_main("--scales", PILOT_SCALE, "--seeds", "101",
                       "--variants", "selectable", "--resume",
                       "--time-limit", "300", "--output", str(output))
    assert resumed.returncode == 0, resumed.stderr
    assert "skipped_completed_case" in resumed.stdout
