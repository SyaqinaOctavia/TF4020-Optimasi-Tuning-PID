import math
import numpy as np
import config
from pid_system import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    clip_bounds,
    evaluate_population_itae,
)


class FlowerPollinationAlgorithmPID:
    """
    Flower Pollination Algorithm (FPA, Yang 2012) untuk optimasi parameter PID [Kp, Ki, Kd].
    - Switching probability p = 0.8 sesuai config.py.
    - Global pollination (r < p): Levy flight (algoritma Mantegna, beta=1.5) menuju gbest.
    - Local pollination (r >= p): langkah diferensial acak antara dua bunga sejawat.
    - Seleksi greedy: solusi diperbarui hanya jika ITAE membaik.
    """

    def __init__(
        self,
        num_flowers: int = config.POPULATION_SIZE,
        num_iterations: int = config.MAX_ITERATIONS,
        switch_prob: float = config.FPA_SWITCHING_PROB,
        gamma_scale: float = 0.1,
        levy_beta: float = 1.5,
    ):
        self.num_flowers = num_flowers
        self.num_iterations = num_iterations
        self.switch_prob = switch_prob
        self.gamma_scale = gamma_scale
        self.levy_beta = levy_beta

    def _levy_flight(self, rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
        beta = self.levy_beta
        num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
        sigma_u = (num / den) ** (1.0 / beta)
        u = rng.normal(0.0, sigma_u, size=size)
        v = rng.normal(0.0, 1.0, size=size)
        return u / (np.abs(v) ** (1.0 / beta) + 1e-12)

    def optimize(self, seed: int | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        rng = np.random.default_rng(seed)
        pop = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(self.num_flowers, config.NUM_PARAMETERS))
        scores = evaluate_population_itae(pop)

        best_idx = int(np.argmin(scores))
        gbest_pos = pop[best_idx].copy()
        gbest_val = float(scores[best_idx])
        history = np.empty(self.num_iterations, dtype=float)

        for it in range(self.num_iterations):
            candidates = np.empty_like(pop)
            is_global = rng.random(self.num_flowers) < self.switch_prob

            levy = self._levy_flight(rng, (self.num_flowers, config.NUM_PARAMETERS))
            global_steps = pop + self.gamma_scale * levy * (gbest_pos[None, :] - pop)

            j_idx = rng.integers(0, self.num_flowers, size=self.num_flowers)
            k_idx = rng.integers(0, self.num_flowers, size=self.num_flowers)
            eps = rng.random((self.num_flowers, 1))
            local_steps = pop + eps * (pop[j_idx] - pop[k_idx])

            candidates[is_global] = global_steps[is_global]
            candidates[~is_global] = local_steps[~is_global]
            candidates = clip_bounds(candidates)

            cand_scores = evaluate_population_itae(candidates)
            improved = cand_scores < scores
            pop[improved] = candidates[improved]
            scores[improved] = cand_scores[improved]

            cur_idx = int(np.argmin(scores))
            if scores[cur_idx] < gbest_val:
                gbest_val = float(scores[cur_idx])
                gbest_pos = pop[cur_idx].copy()
            history[it] = gbest_val

        return gbest_pos, gbest_val, history
