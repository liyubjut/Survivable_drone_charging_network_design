"""Tests for the instance model and the deterministic instance generator."""

from __future__ import annotations

import pytest

from instance import (
    DEFAULT_FLIGHT_RADIUS,
    TwoRadiusInstance,
    generate_two_radius_instance,
)


def tiny_instance(**overrides) -> TwoRadiusInstance:
    """Two depots and one station on a line; p0 has two covering facilities.

    The depot-depot edge exists at ``R = 2`` (distance 2.0), so the cheapest
    feasible design opens only the two depots for a total cost of 7.
    """
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


def test_demands_are_coverage_points_not_backbone_vertices() -> None:
    instance = tiny_instance()
    assert instance.coverage["p0"] == ("d0", "d1", "f0")
    assert all("p0" not in edge for edge in instance.physical_edges)


def test_physical_edges_follow_the_flight_radius() -> None:
    instance = tiny_instance()
    assert set(instance.physical_edges) == {("d0", "d1"), ("d0", "f0"), ("d1", "f0")}
    assert set(instance.backbone_edges) == {
        ("d0", "d1"), ("d0", "f0"), ("d1", "f0"),
        ("rho", "d0"), ("rho", "d1"),
    }


def test_rejects_service_radius_above_half_flight_radius() -> None:
    with pytest.raises(ValueError, match="2r <= R"):
        tiny_instance(service_radius=1.5)


def test_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="selectable or preinstalled"):
        tiny_instance(variant="whatever")


def test_requires_at_least_two_depots() -> None:
    with pytest.raises(ValueError, match="two depots"):
        TwoRadiusInstance(
            variant="selectable",
            facility_positions={"d0": (0.0, 0.0), "f0": (1.0, 0.0)},
            demand_positions={"p0": (1.0, 0.0)},
            depots=("d0",),
            stations=("f0",),
            requirements={"p0": 1},
            costs={"d0": 3, "f0": 2},
            flight_radius=2.0,
            service_radius=1.0,
            seed=1,
        )


def test_preinstalled_depots_must_be_free() -> None:
    with pytest.raises(ValueError, match="zero opening cost"):
        tiny_instance(variant="preinstalled")


def test_generator_pairing_and_radii() -> None:
    common = dict(
        num_depots=5, num_stations=50, num_demands=25,
        requirement_mode="all2", seed=101, fixed_radius=DEFAULT_FLIGHT_RADIUS,
    )
    selectable, meta_selectable = generate_two_radius_instance(
        variant="selectable", **common)
    preinstalled, meta_preinstalled = generate_two_radius_instance(
        variant="preinstalled", **common)

    # The two variants of a pair must describe exactly the same problem.
    assert (
        meta_selectable["paired_instance_sha256"]
        == meta_preinstalled["paired_instance_sha256"]
    )
    assert selectable.flight_radius == pytest.approx(DEFAULT_FLIGHT_RADIUS)
    assert selectable.service_radius == pytest.approx(DEFAULT_FLIGHT_RADIUS / 2)
    assert 2 * selectable.service_radius <= selectable.flight_radius + 1e-10
    assert all(
        cost == 0 for depot, cost in preinstalled.costs.items()
        if depot in set(preinstalled.depots)
    )
    assert set(selectable.facilities) == set(selectable.facility_positions)
    for demand, requirement in selectable.requirements.items():
        assert len(selectable.coverage[demand]) >= requirement


def test_generator_is_deterministic_for_a_fixed_seed() -> None:
    common = dict(
        num_depots=5, num_stations=50, num_demands=25,
        variant="selectable", requirement_mode="all2", seed=102,
    )
    first, meta_first = generate_two_radius_instance(**common)
    second, meta_second = generate_two_radius_instance(**common)
    assert first == second
    assert meta_first["paired_instance_sha256"] == meta_second["paired_instance_sha256"]


def test_generator_raises_when_no_valid_layout_is_found() -> None:
    with pytest.raises(ValueError, match="two-root-connected"):
        generate_two_radius_instance(
            num_depots=2,
            num_stations=1,
            num_demands=1,
            variant="selectable",
            requirement_mode="all1",
            seed=7,
            fixed_radius=20.0,
            max_facility_draws=3,
        )
