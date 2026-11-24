import numpy as np
import config
from pid_system import LOWER_BOUNDS, UPPER_BOUNDS, clip_bounds, hitung_itae


class ParticleSwarmOptimizationPID:
    def __init__(
        self,
        num_particles=config.POPULATION_SIZE,
        num_iterations=config.MAX_ITERATIONS,
        w_max=config.PSO_W_MAX,
        w_min=config.PSO_W_MIN,
        c1=2.0,
        c2=2.0,
    ):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self, seed=None):
        rng = np.random.default_rng(seed)

        pos = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(self.num_particles, 3))
        vel = np.zeros((self.num_particles, 3))

        pbest_pos = pos.copy()
        pbest_val = np.array([hitung_itae(p[0], p[1], p[2]) for p in pos])

        gbest_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = float(pbest_val[gbest_idx])
        history = []

        for it in range(self.num_iterations):
            # Inertia weight menurun linear dari 0.9 -> 0.4
            w = self.w_max - (self.w_max - self.w_min) * (it / max(1, self.num_iterations - 1))

            for i in range(self.num_particles):
                r1 = rng.random(3)
                r2 = rng.random(3)

                vel[i] = (
                    w * vel[i]
                    + self.c1 * r1 * (pbest_pos[i] - pos[i])
                    + self.c2 * r2 * (gbest_pos - pos[i])
                )
                pos[i] = clip_bounds(pos[i] + vel[i])

                val = hitung_itae(pos[i, 0], pos[i, 1], pos[i, 2])
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()
                    if val < gbest_val:
                        gbest_val = float(val)
                        gbest_pos = pos[i].copy()

            history.append(gbest_val)

        return gbest_pos, gbest_val, np.array(history)
