import math
import numpy as np
import config
from pid_system import LOWER_BOUNDS, UPPER_BOUNDS, clip_bounds, hitung_itae


class FlowerPollinationAlgorithmPID:
    def __init__(
        self,
        num_flowers=config.POPULATION_SIZE,
        num_iterations=config.MAX_ITERATIONS,
        switch_prob=config.FPA_SWITCHING_PROB,
    ):
        self.num_flowers = num_flowers
        self.num_iterations = num_iterations
        self.switch_prob = switch_prob

    def _levy_step(self, rng, beta=1.5):
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (num / den) ** (1 / beta)
        u = rng.normal(0.0, sigma, size=3)
        v = rng.normal(0.0, 1.0, size=3)
        return u / (np.abs(v) ** (1 / beta) + 1e-12)

    def optimize(self, seed=None):
        rng = np.random.default_rng(seed)

        flowers = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(self.num_flowers, 3))
        fitness = np.array([hitung_itae(f[0], f[1], f[2]) for f in flowers])

        best_idx = int(np.argmin(fitness))
        gbest_pos = flowers[best_idx].copy()
        gbest_val = float(fitness[best_idx])
        history = []

        for _ in range(self.num_iterations):
            for i in range(self.num_flowers):
                if rng.random() < self.switch_prob:
                    # Global pollination
                    step = 0.1 * self._levy_step(rng) * (gbest_pos - flowers[i])
                    candidate = flowers[i] + step
                else:
                    # Local pollination
                    j, k = rng.integers(0, self.num_flowers, size=2)
                    eps = rng.random()
                    candidate = flowers[i] + eps * (flowers[j] - flowers[k])

                candidate = clip_bounds(candidate)
                val = hitung_itae(candidate[0], candidate[1], candidate[2])

                if val < fitness[i]:
                    flowers[i] = candidate
                    fitness[i] = val
                    if val < gbest_val:
                        gbest_val = float(val)
                        gbest_pos = candidate.copy()

            history.append(gbest_val)

        return gbest_pos, gbest_val, np.array(history)
