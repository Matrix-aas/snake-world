import numpy as np
from snake_rl.worldgen import generate_world
from snake_rl.config import CFG
from snake_rl import genome as gm


def test_snake_has_genome_and_phenotype():
    w = generate_world(CFG, seed=3, n_snakes=2)   # live snakes (arrivals default False)
    s = w.snakes[0]
    assert s.genome.shape == (gm.GENE_COUNT,)
    assert s.sex in (0, 1)
    assert s.age == 0 and s.max_lifespan > 0
    ph = w._phenotype_of(s)
    assert ph.max_length > 0 and ph.v_dash > 0
