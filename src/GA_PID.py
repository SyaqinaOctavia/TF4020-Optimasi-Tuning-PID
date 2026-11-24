import numpy as np
import config
from pid_system import LOWER_BOUNDS, UPPER_BOUNDS, clip_bounds, hitung_itae


class GeneticAlgorithmPID:
    def __init__(
        self,
        population_size=config.POPULATION_SIZE,
        generations=config.MAX_ITERATIONS,
        crossover_rate=0.8,
        mutation_rate=0.2,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _select_parent(self, rng, pop, fitness):
        # pilih 2 acak, ambil ITAE terkecil
        i, j = rng.integers(0, len(pop), size=2)
        return pop[i] if fitness[i] < fitness[j] else pop[j]

    def optimize(self, seed=None):
        rng = np.random.default_rng(seed)
        pop = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(self.population_size, 3))
        fitness = np.array([hitung_itae(p[0], p[1], p[2]) for p in pop])

        best_idx = int(np.argmin(fitness))
        best_pos = pop[best_idx].copy()
        best_val = float(fitness[best_idx])
        history = []

        span = UPPER_BOUNDS - LOWER_BOUNDS

        for _ in range(self.generations):
            # Elitism: simpan 1 individu terbaik ke generasi berikutnya
            new_pop = [best_pos.copy()]

            while len(new_pop) < self.population_size:
                p1 = self._select_parent(rng, pop, fitness)
                p2 = self._select_parent(rng, pop, fitness)

                # Crossover
                if rng.random() < self.crossover_rate:
                    alpha = rng.random()
                    c1 = alpha * p1 + (1 - alpha) * p2
                    c2 = (1 - alpha) * p1 + alpha * p2
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutasi acak
                for child in (c1, c2):
                    if rng.random() < self.mutation_rate:
                        child = child + rng.normal(0.0, 0.05 * span)
                    new_pop.append(clip_bounds(child))
                    if len(new_pop) >= self.population_size:
                        break

            pop = np.array(new_pop)
            fitness = np.array([hitung_itae(p[0], p[1], p[2]) for p in pop])

            cur_idx = int(np.argmin(fitness))
            if fitness[cur_idx] < best_val:
                best_val = float(fitness[cur_idx])
                best_pos = pop[cur_idx].copy()
            history.append(best_val)

        return best_pos, best_val, np.array(history)
