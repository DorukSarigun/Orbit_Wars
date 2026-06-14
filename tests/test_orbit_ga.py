"""Tests for the GA tuner (operators + genome plumbing, no slow matches)."""

from __future__ import annotations

import random

from orbit_ga import (
    GENE_SPEC,
    config_from_genome,
    crossover,
    default_genome,
    mutate,
    random_genome,
    _coerce,
)
from orbit_scripted_agent import OrbitScriptedConfig
from orbit_lookahead_agent import LookaheadConfig


def test_default_genome_round_trips_to_defaults():
    cfg, look = config_from_genome(default_genome())
    base = OrbitScriptedConfig()
    assert cfg.influence_weight == base.influence_weight
    assert cfg.eta_horizon == base.eta_horizon
    assert cfg.max_wave_launches == base.max_wave_launches
    assert cfg.prioritize_enemy_weakspots == base.prioritize_enemy_weakspots
    assert isinstance(look, LookaheadConfig)


def test_config_from_genome_coerces_types():
    cfg, look = config_from_genome(random_genome(random.Random(0)))
    assert isinstance(cfg.max_wave_launches, int)
    assert isinstance(cfg.support_reserve_slots, int)
    assert isinstance(cfg.prioritize_enemy_weakspots, bool)
    assert isinstance(cfg.allow_partial_enemy_waves, bool)
    assert isinstance(look.max_depth, int)
    assert isinstance(look.lookahead_weight, float)


def test_coerce_clamps_into_range():
    for name, g in GENE_SPEC.items():
        lo = _coerce(name, g.low - 100.0)
        hi = _coerce(name, g.high + 100.0)
        assert g.low - 1 <= float(lo) <= g.high + 1
        assert g.low - 1 <= float(hi) <= g.high + 1


def test_crossover_and_mutation_stay_in_bounds():
    rng = random.Random(1)
    a = random_genome(rng)
    b = random_genome(rng)
    for _ in range(50):
        child = mutate(crossover(a, b, rng), rng, rate=0.5, scale=0.3)
        for name, g in GENE_SPEC.items():
            assert g.low <= child[name] <= g.high, (name, child[name])


def test_lookahead_eta_tied_to_attack_horizon():
    genome = default_genome()
    genome["eta_horizon"] = 47.0
    cfg, look = config_from_genome(genome)
    assert look.eta_horizon == cfg.eta_horizon == 47.0
