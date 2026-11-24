import numpy as np
from control.matlab import tf, feedback, poles, step
import config

G_PLANT = tf([10], [1, 7, 10, 0])

LOWER_BOUNDS = np.array([config.KP_MIN, config.KI_MIN, config.KD_MIN], dtype=float)
UPPER_BOUNDS = np.array([config.KP_MAX, config.KI_MAX, config.KD_MAX], dtype=float)

T_OPT = np.arange(0.0, config.T_SIM + 0.05, 0.05)
T_SIM = np.arange(0.0, config.T_SIM + config.DT, config.DT)


def clip_bounds(x):
    return np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS)


def closed_loop(kp, ki, kd):
    C = tf([kd, kp, ki], [1, 0])
    return feedback(C * G_PLANT, 1)


def hitung_itae(kp, ki, kd):
    # Persamaan karakteristik: s^4 + 7s^3 + (10 + 10*Kd)s^2 + 10*Kp*s + 10*Ki = 0
    den = [1.0, 7.0, 10.0 + 10.0 * kd, 10.0 * kp, 10.0 * ki]
    if np.any(np.real(np.roots(den)) >= 0):
        return 1e6

    num = [10.0 * kd, 10.0 * kp, 10.0 * ki]
    T = tf(num, den)
    y, t = step(T, T_OPT)

    if not np.all(np.isfinite(y)):
        return 1e6

    dt = t[1] - t[0]
    error = np.abs(config.REF_VALUE - y)
    return float(np.sum(t * error) * dt)


def simulate_single_step(kp, ki, kd):
    T = closed_loop(kp, ki, kd)
    y, t = step(T, T_SIM)
    dt = t[1] - t[0]
    itae = float(np.sum(t * np.abs(config.REF_VALUE - y)) * dt)
    return t, y, itae


def compute_transient_metrics(t, y, r_val=config.REF_VALUE):
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
    dt = t[1] - t[0]
    itae = float(np.sum(t * np.abs(r_val - y)) * dt)
    return {"Mp": mp_percent, "Ts": ts, "ess": ess, "ITAE": itae}
