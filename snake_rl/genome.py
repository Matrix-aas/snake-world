"""Pure genome operations + phenotype resolution. No sim state, no I/O (Pitfall-12 note:
callers observe the GENE, never the derived stat -- see resolve_phenotype in Task 3)."""
import numpy as np

GENE_COUNT = 9
SIZE, METABOLISM, SPEED, STAMINA, SENSES, LIFESPAN, AGGRESSION, KIN_CARE, BOLDNESS = range(GENE_COUNT)

_NORM = float(np.sqrt(GENE_COUNT))   # ‖ones(9)‖, the max genome L2 distance


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
