import math
import numpy as np
import config
from pid_system import LOWER_BOUNDS, UPPER_BOUNDS, clip_bounds, hitung_itae


class AntColonyOptimizationPID:
    def __init__(
        self,
        num_ants=config.POPULATION_SIZE,
        num_iterations=config.MAX_ITERATIONS,
        evaporation_rate=0.85,
        q=0.3,
    ):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.xi = evaporation_rate
        self.q = q

    def optimize(self, seed=None):
        rng = np.random.default_rng(seed)
        k = self.num_ants

        archive = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(k, 3))
        scores = np.array([hitung_itae(a[0], a[1], a[2]) for a in archive])
        order = np.argsort(scores)
        archive = archive[order]
        scores = scores[order]

        # Bobot feromon Gaussian berdasarkan peringkat solusi di arsip
        ranks = np.arange(1, k + 1, dtype=float)
        weights = np.exp(-((ranks - 1) ** 2) / (2 * (self.q * k) ** 2)) / (
            self.q * k * math.sqrt(2 * math.pi)
        )
        probs = weights / np.sum(weights)

        history = []

        for _ in range(self.num_iterations):
            new_ants = []
            new_scores = []

            for _ in range(self.num_ants):
                idx = int(rng.choice(k, p=probs))
                guide = archive[idx]

                sigma = self.xi * np.sum(np.abs(archive - guide), axis=0) / max(1, k - 1)
                sigma = np.maximum(sigma, 1e-6)

                cand = clip_bounds(rng.normal(guide, sigma))
                new_ants.append(cand)
                new_scores.append(hitung_itae(cand[0], cand[1], cand[2]))

            # Gabungkan lalu simpan
            archive = np.vstack([archive, np.array(new_ants)])
            scores = np.concatenate([scores, np.array(new_scores)])
            best_order = np.argsort(scores)[:k]
            archive = archive[best_order]
            scores = scores[best_order]

            history.append(float(scores[0]))

        return archive[0].copy(), float(scores[0]), np.array(history)
