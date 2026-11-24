"""
Model sistem closed-loop PID, simulasi numerik RK4, evaluasi ITAE, dan metrik transien.
Plant: G(s) = 10 / (s * (s + 2) * (s + 5)) = 10 / (s^3 + 7s^2 + 10s).
"""

import numpy as np
import config


LOWER_BOUNDS = np.array([config.KP_MIN, config.KI_MIN, config.KD_MIN], dtype=float)
UPPER_BOUNDS = np.array([config.KP_MAX, config.KI_MAX, config.KD_MAX], dtype=float)


def clip_bounds(x: np.ndarray) -> np.ndarray:
    return np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)


def routh_hurwitz_violation(params: np.ndarray) -> np.ndarray:
    """
    Cek syarat perlu & cukup kestabilan Routh-Hurwitz untuk persamaan karakteristik:
      s^4 + 7s^3 + 10(1 + Kd)s^2 + 10*Kp*s + 10*Ki = 0
    Mengembalikan 0.0 jika stabil, atau nilai pelanggaran > 0 jika tidak stabil.
    """
    kp = params[:, 0]
    ki = params[:, 1]
    kd = params[:, 2]

    margin_1 = 7.0 * (1.0 + kd) - kp  # harus > 0
    margin_2 = kp * margin_1 - 4.9 * ki  # harus > 0

    v1 = (
        np.maximum(0.0, 1e-6 - kp)
        + np.maximum(0.0, 1e-6 - ki)
        + np.maximum(0.0, -1.0 + 1e-6 - kd)
    )
    v2 = np.maximum(0.0, 1e-6 - margin_1)
    v3 = np.maximum(0.0, 1e-6 - margin_2)
    return v1 + v2 + v3


def rk4_closed_loop_step(
    params: np.ndarray,
    dt: float = config.DT,
    t_sim: float = config.T_SIM,
    r_val: float = config.REF_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrasi RK4 orde-4 pada bentuk kanonik terkontrol dari T(s):
      z' = A*z + b_u*r(t),  y(t) = c^T * z
    Dieksekusi dengan operator transisi afin eksak RK4 per blok B=50 langkah.
    """
    n_candidates = params.shape[0]
    n_steps = int(round(t_sim / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    kp = params[:, 0]
    ki = params[:, 1]
    kd = params[:, 2]

    A = np.zeros((n_candidates, 4, 4), dtype=float)
    A[:, 0, 1] = 1.0
    A[:, 1, 2] = 1.0
    A[:, 2, 3] = 1.0
    A[:, 3, 0] = -10.0 * ki
    A[:, 3, 1] = -10.0 * kp
    A[:, 3, 2] = -10.0 * (1.0 + kd)
    A[:, 3, 3] = -7.0

    b_u = np.zeros((n_candidates, 4, 1), dtype=float)
    b_u[:, 3, 0] = r_val

    C = np.zeros((n_candidates, 4), dtype=float)
    C[:, 0] = 10.0 * ki
    C[:, 1] = 10.0 * kp
    C[:, 2] = 10.0 * kd

    I4 = np.broadcast_to(np.eye(4, dtype=float), (n_candidates, 4, 4))
    dtA = dt * A
    dtA2 = dtA @ dtA
    dtA3 = dtA2 @ dtA
    dtA4 = dtA3 @ dtA

    M = I4 + dtA + 0.5 * dtA2 + (1.0 / 6.0) * dtA3 + (1.0 / 24.0) * dtA4
    g = dt * ((I4 + 0.5 * dtA + (1.0 / 6.0) * dtA2 + (1.0 / 24.0) * dtA3) @ b_u)[..., 0]

    block_size = min(50, n_steps)
    M_blk = np.empty((n_candidates, block_size, 4, 4), dtype=float)
    g_blk = np.empty((n_candidates, block_size, 4), dtype=float)
    cur_M = M.copy()
    cur_g = g.copy()
    for j in range(block_size):
        M_blk[:, j] = cur_M
        g_blk[:, j] = cur_g
        if j + 1 < block_size:
            cur_g = (M @ cur_g[..., None])[..., 0] + g
            cur_M = M @ cur_M

    y = np.zeros((n_candidates, n_steps + 1), dtype=float)
    z = np.zeros((n_candidates, 4), dtype=float)

    pos = 0
    while pos < n_steps:
        take = min(block_size, n_steps - pos)
        Z_blk = np.einsum("nbij,nj->nbi", M_blk[:, :take], z) + g_blk[:, :take]
        y[:, pos + 1 : pos + 1 + take] = np.einsum("ni,nbi->nb", C, Z_blk)
        z = Z_blk[:, -1]
        pos += take

    return t, y


def evaluate_population_itae(candidates: np.ndarray) -> np.ndarray:
    """
    Evaluasi objektif numerik J_ITAE = sum(t_k * |r - y(t_k)| * dt) untuk batch (N, 3).
    Kandidat tak stabil menurut Routh-Hurwitz diberi penalti kontinu tanpa simulasi.
    """
    candidates = np.atleast_2d(candidates)
    n = candidates.shape[0]
    scores = np.empty(n, dtype=float)

    violation = routh_hurwitz_violation(candidates)
    unstable_mask = violation > 0.0
    stable_mask = ~unstable_mask

    if np.any(unstable_mask):
        scores[unstable_mask] = 1e6 + 1e3 * violation[unstable_mask]

    if np.any(stable_mask):
        t, y = rk4_closed_loop_step(candidates[stable_mask])
        err = np.abs(config.REF_VALUE - y)
        itae = np.sum(err * t[None, :], axis=1) * config.DT
        invalid = ~np.isfinite(itae)
        itae[invalid] = 1e6
        scores[stable_mask] = itae

    return scores


def simulate_single_step(kp: float, ki: float, kd: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulasikan satu set parameter PID dan kembalikan (t, y, itae)."""
    params = np.array([[kp, ki, kd]], dtype=float)
    t, y_batch = rk4_closed_loop_step(params)
    y = y_batch[0]
    itae = float(np.sum(np.abs(config.REF_VALUE - y) * t) * config.DT)
    return t, y, itae


def compute_transient_metrics(
    t: np.ndarray, y: np.ndarray, r_val: float = config.REF_VALUE
) -> dict[str, float]:
    """Hitung Mp (%), Ts (toleransi 2%), ess, dan J_ITAE dari respons step."""
    y_max = float(np.max(y))
    mp_percent = max(0.0, (y_max - r_val) / r_val * 100.0)

    outside_band = np.where(np.abs(y - r_val) > config.SETTLING_TOL * r_val)[0]
    if len(outside_band) == 0:
        ts = 0.0
    elif outside_band[-1] >= len(t) - 1:
        ts = float("inf")
    else:
        ts = float(t[outside_band[-1] + 1])

    ess = float(abs(r_val - y[-1]))
    itae = float(np.sum(np.abs(r_val - y) * t) * (t[1] - t[0]))
    return {"Mp": mp_percent, "Ts": ts, "ess": ess, "ITAE": itae}
