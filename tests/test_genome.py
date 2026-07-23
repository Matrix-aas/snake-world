import numpy as np
from snake_rl import genome as gm


def test_sample_shape_and_range():
    rng = np.random.default_rng(0)
    g = gm.sample_genome(rng)
    assert g.shape == (gm.GENE_COUNT,) and g.dtype == np.float32
    assert (g >= 0).all() and (g <= 1).all()


def test_crossover_is_per_gene_from_a_parent():
    rng = np.random.default_rng(1)
    a = np.zeros(gm.GENE_COUNT, np.float32)
    b = np.ones(gm.GENE_COUNT, np.float32)
    c = gm.crossover(a, b, rng)
    # every gene came from exactly one parent (0 or 1), and both parents contribute over many draws
    assert set(np.unique(c)).issubset({0.0, 1.0})
    seen = set()
    for _ in range(50):
        seen.update(np.unique(gm.crossover(a, b, np.random.default_rng(_))))
    assert seen == {0.0, 1.0}


def test_mutate_stays_in_unit_box():
    rng = np.random.default_rng(2)
    g = np.array([0.0, 1.0] + [0.5] * (gm.GENE_COUNT - 2), np.float32)
    for _ in range(200):
        m = gm.mutate(g, rng, sigma=0.5)
        assert (m >= 0).all() and (m <= 1).all()


def test_relatedness_bounds():
    ones = np.ones(gm.GENE_COUNT, np.float32)
    zeros = np.zeros(gm.GENE_COUNT, np.float32)
    assert gm.relatedness(ones, ones) == 1.0
    assert gm.relatedness(ones, zeros) == 0.0
    assert 0.0 <= gm.relatedness(ones, ones * 0.5) <= 1.0
