import numpy as np
import config
from pid_system import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    clip_bounds,
    evaluate_population_itae,
)


class ParticleSwarmOptimizationPID:
    """
    Particle Swarm Optimization (PSO) untuk optimasi parameter PID [Kp, Ki, Kd].
    - Inertia weight meluruh linear dari w_max = 0.9 ke w_min = 0.4 sesuai config.py.
    - Koefisien kognitif & sosial standar c1 = c2 = 2.0.
    - Velocity clamping 20% rentang variabel agar partikel tidak melompat ke batas tak stabil.
    """

    def __init__(
        self,
        num_particles: int = config.POPULATION_SIZE,
        num_iterations: int = config.MAX_ITERATIONS,
        w_max: float = config.PSO_W_MAX,
        w_min: float = config.PSO_W_MIN,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self, seed: int | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        rng = np.random.default_rng(seed)
        span = UPPER_BOUNDS - LOWER_BOUNDS
        v_max = 0.2 * span

        pos = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(self.num_particles, config.NUM_PARAMETERS))
        vel = rng.uniform(-0.1 * span, 0.1 * span, size=(self.num_particles, config.NUM_PARAMETERS))
        scores = evaluate_population_itae(pos)

        pbest_pos = pos.copy()
        pbest_val = scores.copy()
        gbest_idx = int(np.argmin(pbest_val))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = float(pbest_val[gbest_idx])

        history = np.empty(self.num_iterations, dtype=float)

        for it in range(self.num_iterations):
            w = self.w_max - (self.w_max - self.w_min) * (it / max(1, self.num_iterations - 1))
            r1 = rng.random((self.num_particles, config.NUM_PARAMETERS))
            r2 = rng.random((self.num_particles, config.NUM_PARAMETERS))

            vel = (
                w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos[None, :] - pos)
            )
            vel = np.clip(vel, -v_max, v_max)
            pos = clip_bounds(pos + vel)

            scores = evaluate_population_itae(pos)
            improved = scores < pbest_val
            pbest_pos[improved] = pos[improved]
            pbest_val[improved] = scores[improved]

            cur_idx = int(np.argmin(pbest_val))
            if pbest_val[cur_idx] < gbest_val:
                gbest_val = float(pbest_val[cur_idx])
                gbest_pos = pbest_pos[cur_idx].copy()
            history[it] = gbest_val

        return gbest_pos, gbest_val, history
