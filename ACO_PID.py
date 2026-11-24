import math
import numpy as np
import config
from pid_system import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    clip_bounds,
    evaluate_population_itae,
)


class AntColonyOptimizationPID:
    """
    Varian ACO yang digunakan: ACO_R (Ant Colony Optimization for Continuous Domains,
    Socha & Dorigo 2008).

    Alasan pemilihan & cara kerja pada optimasi parameter PID kontinu [Kp, Ki, Kd]:
    - ACO klasik hanya bekerja pada graf diskrit (matriks feromon antar node).
      Untuk ruang pencarian kontinu R^3, ACO_R mengganti tabel feromon diskrit
      dengan Solution Archive berukuran k yang menyimpan solusi terbaik terurut.
    - Bobot feromon tiap peringkat l mengikuti kernel Gaussian (parameter q),
      sehingga solusi dengan ITAE lebih kecil memiliki probabilitas terpilih lebih besar.
    - Setiap semut memilih solusi pemandu l lalu mensampel tiap parameter dari
      distribusi normal N(mu_l^d, (sigma_l^d)^2), dengan sigma_l^d proporsional
      terhadap jarak rata-rata ke solusi arsip lain dikalikan parameter xi
      (berfungsi analog dengan laju evaporasi feromon).
    - Arsip lama dan semut baru digabung lalu diambil k terbaik per iterasi.
    """

    def __init__(
        self,
        num_ants: int = config.POPULATION_SIZE,
        num_iterations: int = config.MAX_ITERATIONS,
        evaporation_rate: float = 0.85,
        q: float = 0.3,
    ):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.xi = evaporation_rate
        self.q = q

    def optimize(self, seed: int | None = None) -> tuple[np.ndarray, float, np.ndarray]:
        rng = np.random.default_rng(seed)
        k_archive = self.num_ants
        archive = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS, size=(k_archive, config.NUM_PARAMETERS))
        archive_scores = evaluate_population_itae(archive)

        order = np.argsort(archive_scores)
        archive = archive[order]
        archive_scores = archive_scores[order]

        ranks = np.arange(1, k_archive + 1, dtype=float)
        weights = np.exp(-((ranks - 1.0) ** 2) / (2.0 * (self.q * k_archive) ** 2)) / (
            self.q * k_archive * math.sqrt(2.0 * math.pi)
        )
        probs = weights / np.sum(weights)

        history = np.empty(self.num_iterations, dtype=float)

        for it in range(self.num_iterations):
            diff = np.abs(archive[:, None, :] - archive[None, :, :])
            sigma = self.xi * np.sum(diff, axis=1) / max(1, k_archive - 1)
            sigma = np.maximum(sigma, 1e-6)

            chosen = rng.choice(k_archive, size=self.num_ants, p=probs)
            new_ants = rng.normal(loc=archive[chosen], scale=sigma[chosen])
            new_ants = clip_bounds(new_ants)
            new_scores = evaluate_population_itae(new_ants)

            merged = np.vstack([archive, new_ants])
            merged_scores = np.concatenate([archive_scores, new_scores])

            best_order = np.argsort(merged_scores)[:k_archive]
            archive = merged[best_order]
            archive_scores = merged_scores[best_order]

            history[it] = float(archive_scores[0])

        return archive[0].copy(), float(archive_scores[0]), history
