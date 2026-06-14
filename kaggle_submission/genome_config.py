"""Genome → OrbitScriptedConfig (submission slice of orbit_ga, no GA loop)."""

from __future__ import annotations

from typing import Any

from orbit_scripted_agent import OrbitScriptedConfig

GENE_SPEC: dict[str, tuple[float, float, str]] = {
    "horizon_cap": (10.0, 40.0, "f"),
    "potential_weight": (0.0, 1.0, "f"),
    "critical_threshold": (-25.0, -2.0, "f"),
    "comet_evac_turns": (0.0, 5.0, "i"),
    "max_rescue_launches": (1.0, 5.0, "i"),
    "eta_horizon": (25.0, 90.0, "f"),
    "denial_weight": (0.0, 3.0, "f"),
    "influence_weight": (0.0, 3.0, "f"),
    "production_pressure_weight": (0.0, 3.0, "f"),
    "enemy_weakness_weight": (0.0, 3.0, "f"),
    "enemy_empire_support_weight": (0.0, 3.0, "f"),
    "prioritize_enemy_weakspots": (0.0, 1.0, "b"),
    "capture_margin": (0.0, 0.5, "f"),
    "max_wave_launches": (1.0, 6.0, "i"),
    "allow_partial_enemy_waves": (0.0, 1.0, "b"),
    "support_reserve_slots": (0.0, 4.0, "i"),
    "support_threat_threshold": (2.0, 20.0, "f"),
    "support_min_source_surplus": (1.0, 15.0, "f"),
    "support_max_source_threat": (0.0, 12.0, "f"),
    "support_cushion": (0.0, 10.0, "f"),
    "max_support_launches_per_target": (1.0, 4.0, "i"),
    "consolidation_min_surplus": (3.0, 25.0, "f"),
    "consolidation_fraction": (0.1, 0.9, "f"),
    "consolidation_min_send": (1.0, 15.0, "i"),
}


def _coerce(name: str, raw: float) -> Any:
    low, high, kind = GENE_SPEC[name]
    v = min(high, max(low, float(raw)))
    if kind == "i":
        return int(round(v))
    if kind == "b":
        return bool(v >= 0.5)
    return float(v)


# Genes forced to a fixed safe value regardless of the genome. The partial
# enemy-wave branch is the only structural code path the GA genome enables that
# the (Kaggle-validated COMPLETE) default config never runs; pin it off so the
# GA variant stays within the proven-safe code surface.
_FORCED: dict[str, Any] = {"allow_partial_enemy_waves": False}


def config_from_genome(genome: dict[str, float]) -> OrbitScriptedConfig:
    cfg_kwargs: dict[str, Any] = {}
    for name in GENE_SPEC:
        val = _coerce(name, genome.get(name, GENE_SPEC[name][0]))
        cfg_kwargs[name] = val
    cfg_kwargs.update(_FORCED)
    return OrbitScriptedConfig(**cfg_kwargs)
