"""Pure genome operations + phenotype resolution. No sim state, no I/O (Pitfall-12 note:
callers observe the GENE, never the derived stat -- see resolve_phenotype in Task 3)."""
import numpy as np
from collections import namedtuple

GENE_COUNT = 9
SIZE, METABOLISM, SPEED, STAMINA, SENSES, LIFESPAN, AGGRESSION, KIN_CARE, BOLDNESS = range(GENE_COUNT)

_NORM = float(np.sqrt(GENE_COUNT))   # ‖ones(9)‖, the max genome L2 distance

Phenotype = namedtuple("Phenotype", [
    "max_length", "turn_deg", "v_snake", "v_dash", "s_max", "stamina_regen",
    "ray_range", "smell_reach", "energy_decay", "max_lifespan_base",
])


def sample_genome(rng):
    return rng.random(GENE_COUNT).astype(np.float32)


def crossover(a, b, rng):
    pick = rng.random(GENE_COUNT) < 0.5
    return np.where(pick, a, b).astype(np.float32)


def mutate(g, rng, sigma):
    return np.clip(g + rng.normal(0.0, sigma, GENE_COUNT), 0.0, 1.0).astype(np.float32)


def relatedness(a, b):
    d = float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum()))
    return float(np.clip(1.0 - d / _NORM, 0.0, 1.0))


def _lerp(lo, hi, t):
    return lo + (hi - lo) * float(t)


def resolve_phenotype(genome, cfg):
    g = genome
    size_hunger = 1.0 + _lerp(0.0, cfg.gene_size_hunger_hi, g[SIZE])
    metab = _lerp(cfg.gene_metab_lo, cfg.gene_metab_hi, g[METABOLISM])
    return Phenotype(
        max_length=cfg.length_cap * _lerp(cfg.gene_size_len_lo, cfg.gene_size_len_hi, g[SIZE]),
        turn_deg=cfg.turn_deg * _lerp(cfg.gene_size_turn_lo, cfg.gene_size_turn_hi, g[SIZE]),
        v_snake=cfg.v_snake * _lerp(cfg.gene_speed_lo, cfg.gene_speed_hi, g[SPEED]),
        v_dash=cfg.v_dash * _lerp(cfg.gene_speed_lo, cfg.gene_speed_hi, g[SPEED]),
        s_max=cfg.s_max * _lerp(cfg.gene_stamina_lo, cfg.gene_stamina_hi, g[STAMINA]),
        stamina_regen=cfg.stamina_regen * _lerp(cfg.gene_stamina_regen_lo, cfg.gene_stamina_regen_hi, g[STAMINA]),
        ray_range=_lerp(cfg.gene_rayrange_lo, cfg.gene_rayrange_hi, g[SENSES]),
        smell_reach=_lerp(cfg.gene_smell_lo, cfg.gene_smell_hi, g[SENSES]),
        energy_decay=cfg.energy_decay * metab * size_hunger,
        max_lifespan_base=_lerp(cfg.gene_lifespan_lo, cfg.gene_lifespan_hi, g[LIFESPAN]),
    )
