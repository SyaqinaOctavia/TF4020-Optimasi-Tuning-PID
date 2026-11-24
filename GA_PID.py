import numpy as np
import config
from pid_system import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    clip_bounds,
    evaluate_population_itae,
)


class GeneticAlgorithmPID:
    """
    Real-Coded Genetic Algorithm (GA) untuk optimasi parameter PID [Kp, Ki, Kd].
    - Tournament selection (k=3): tahan terhadap skala ITAE lebar dan nilai penalti
      (mengatasi error probabilitas negatif pada roulette-wheel naive).
    - BLX-alpha crossover (alpha=0.3) untuk eksplorasi kontinu di sekitar induk.
    - Gaussian mutation dengan sigma meluruh linear (15% -> 2% rentang variabel).
    - Elitism mempertahankan individu terbaik tiap generasi.
    """

    def __init__(
        self,
        population_size: int = config.POPULATION_SIZE,
        generations: int = config.MAX_ITERATIONS,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        elitism: int = 2,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism

    def optimize(self, seed: int | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        rng = np.random.default_rng(seed)
        pop = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(self.population_size, config.NUM_PARAMETERS))
        fitness = evaluate_population_itae(pop)

        best_idx = int(np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_val = float(fitness[best_idx])
        history = np.empty(self.generations, dtype=float)
        span = UPPER_BOUNDS - LOWER_BOUNDS

        for it in range(self.generations):
            order = np.argsort(fitness)
            next_pop = [pop[i].copy() for i in order[: self.elitism]]
            sigma_scale = 0.15 - (0.15 - 0.02) * (it / max(1, self.generations - 1))

            while len(next_pop) < self.population_size:
                parents = []
                for _ in range(2):
                    cand = rng.choice(self.population_size, size=3, replace=False)
                    winner = cand[int(np.argmin(fitness[cand]))]
                    parents.append(pop[winner])
                p1, p2 = parents[0], parents[1]

                if rng.random() < self.crossover_rate:
                    alpha = 0.3
                    d = np.abs(p1 - p2)
                    low = np.minimum(p1, p2) - alpha * d
                    high = np.maximum(p1, p2) + alpha * d
                    c1 = rng.uniform(low, high)
                    c2 = rng.uniform(low, high)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                for child in (c1, c2):
                    if rng.random() < self.mutation_rate:
                        mask = rng.random(config.NUM_PARAMETERS) < 0.5
                        if not np.any(mask):
                            mask[rng.integers(0, config.NUM_PARAMETERS)] = True
                        child[mask] += rng.normal(0.0, sigma_scale * span[mask])
                    next_pop.append(clip_bounds(child))
                    if len(next_pop) >= self.population_size:
                        break

            pop = np.array(next_pop[: self.population_size], dtype=float)
            fitness = evaluate_population_itae(pop)

            cur_idx = int(np.argmin(fitness))
            if fitness[cur_idx] < best_val:
                best_val = float(fitness[cur_idx])
                best_pos = pop[cur_idx].copy()
            history[it] = best_val

        return best_pos, best_val, history
