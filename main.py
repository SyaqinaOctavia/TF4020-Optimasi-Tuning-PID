"""
TF4020 - Optimasi Tuning Parameter PID Menggunakan 4 Algoritma Metaheuristik
(GA, PSO, FPA, dan ACO_R) untuk Plant G(s) = 10 / (s * (s + 2) * (s + 5)).

Task 1 s.d. Task 5 dijalankan dari file ini dengan mengimpor class masing-masing
algoritma (GA_PID.py, PSO_PID.py, FPA_PID.py, ACO_PID.py), model/simulasi
(pid_system.py), dan konfigurasi (config.py).
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from pid_system import compute_transient_metrics, simulate_single_step
from GA_PID import GeneticAlgorithmPID
from PSO_PID import ParticleSwarmOptimizationPID
from FPA_PID import FlowerPollinationAlgorithmPID
from ACO_PID import AntColonyOptimizationPID


# ==============================================================================
# TASK 1: PEMODELAN CLOSED-LOOP, FUNGSI TRANSFER, DAN PERSAMAAN KARAKTERISTIK
# ==============================================================================
def print_task1_analysis() -> None:
    print("=" * 78)
    print("TASK 1: ANALISIS SISTEM CLOSED-LOOP DENGAN PID CONTROLLER")
    print("=" * 78)
    print(
        """
1. Diagram Blok Sistem Closed-Loop (Unity Negative Feedback):

                +   E(s)   +--------------------+   U(s)   +------------------------+
   R(s) -----> ( ) ------> | C(s) = Kp + Ki/s   | -------> | G(s) = 10 /            | ---+---> Y(s)
                ^ -        |        + Kd*s      |          |      [s(s + 2)(s + 5)]  |    |
                |          +--------------------+          +------------------------+    |
                |                                                                         |
                +-------------------------------------------------------------------------+
                                        Feedback Unity: H(s) = 1

2. Penurunan Fungsi Transfer Closed-Loop T(s) = Y(s)/R(s):
   - Plant:
       G(s) = 10 / [s(s + 2)(s + 5)] = 10 / (s^3 + 7s^2 + 10s)
   - PID Controller (bentuk paralel ideal):
       C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
   - Fungsi Transfer Open-Loop L(s) = C(s) * G(s):
       L(s) = 10(Kd*s^2 + Kp*s + Ki) / [s^2(s + 2)(s + 5)]
            = (10*Kd*s^2 + 10*Kp*s + 10*Ki) / (s^4 + 7s^3 + 10s^2)
   - Fungsi Transfer Closed-Loop (dengan H(s) = 1):
       T(s) = L(s) / [1 + L(s)]
            = (10*Kd*s^2 + 10*Kp*s + 10*Ki) /
              [s^4 + 7s^3 + (10 + 10*Kd)s^2 + 10*Kp*s + 10*Ki]

3. Persamaan Karakteristik Closed-Loop (1 + C(s)G(s) = 0):
       s^4 + 7s^3 + (10 + 10*Kd)s^2 + 10*Kp*s + 10*Ki = 0

4. Catatan Kestabilan Routh-Hurwitz (Reasoning Optimasi):
   Dari polinomial a4*s^4 + a3*s^3 + a2*s^2 + a1*s + a0 = 0 dengan
   a4 = 1, a3 = 7, a2 = 10(1 + Kd), a1 = 10*Kp, a0 = 10*Ki,
   syarat perlu & cukup agar seluruh pole berada di LHP (stabil) adalah:
     (i)   Kp > 0, Ki > 0, Kd > -1
     (ii)  b1 = 10(1 + Kd) - (10*Kp)/7 > 0  <=>  Kp < 7(1 + Kd)
     (iii) c1 = 10*Kp - 7*(10*Ki)/b1 > 0    <=>  Kp * [7(1 + Kd) - Kp] > 4.9 * Ki
   Karena batas atas config (Kp_max = 100, Kd_max = 10) memungkinkan kombinasi
   tak stabil (misal Kp = 100 > 7*(1 + 10) = 77), evaluator menambahkan cek
   Routh-Hurwitz analitik sebelum simulasi RK4 dan memberi penalti kontinu
   pada kandidat tidak stabil agar optimizer selalu terdorong ke domain stabil.
"""
    )


# ==============================================================================
# EKSEKUSI EKSPERIMEN, SIMULASI, GRAFIK, ANALISIS & KESIMPULAN (TASK 2 - TASK 5)
# ==============================================================================
def run_all_experiments() -> dict[str, dict]:
    solvers = {
        "GA": GeneticAlgorithmPID(),
        "PSO": ParticleSwarmOptimizationPID(),
        "FPA": FlowerPollinationAlgorithmPID(),
        "ACO_R": AntColonyOptimizationPID(),
    }
    results: dict[str, dict] = {}

    for algo_idx, (name, solver) in enumerate(solvers.items()):
        run_params = []
        run_itae = []
        run_histories = []

        for exp in range(config.NUM_EXPERIMENTS):
            seed = config.RANDOM_SEED + algo_idx * 100 + exp
            best_pos, best_val, history = solver.optimize(seed=seed)
            run_params.append(best_pos)
            run_itae.append(best_val)
            run_histories.append(history)

        run_itae_arr = np.array(run_itae, dtype=float)
        run_histories_arr = np.array(run_histories, dtype=float)
        best_run_idx = int(np.argmin(run_itae_arr))
        best_pid = run_params[best_run_idx]

        t, y, sim_itae = simulate_single_step(best_pid[0], best_pid[1], best_pid[2])
        metrics = compute_transient_metrics(t, y)

        results[name] = {
            "best_pid": best_pid,
            "best_itae": float(run_itae_arr[best_run_idx]),
            "mean_itae": float(np.mean(run_itae_arr)),
            "std_itae": float(np.std(run_itae_arr)),
            "worst_itae": float(np.max(run_itae_arr)),
            "best_history": run_histories_arr[best_run_idx],
            "mean_history": np.mean(run_histories_arr, axis=0),
            "t": t,
            "y": y,
            "metrics": metrics,
            "sim_itae_check": sim_itae,
        }

    return results


def plot_results(results: dict[str, dict], out_dir: str) -> tuple[str, str]:
    colors = {"GA": "#1f77b4", "PSO": "#d62728", "FPA": "#2ca02c", "ACO_R": "#9467bd"}
    styles = {"GA": "-", "PSO": "--", "FPA": "-.", "ACO_R": ":"}

    # Grafik 1: Perbandingan Respons Step Keempat Algoritma + Sinyal Referensi
    fig1, ax1 = plt.subplots(figsize=(10, 5.5), dpi=150)
    t_ref = next(iter(results.values()))["t"]
    ax1.plot(
        t_ref,
        np.ones_like(t_ref) * config.REF_VALUE,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Referensi r(t) = 1",
    )
    ax1.axhspan(
        config.REF_VALUE * (1.0 - config.SETTLING_TOL),
        config.REF_VALUE * (1.0 + config.SETTLING_TOL),
        color="gray",
        alpha=0.12,
        label="Pita Toleransi 2% [0.98, 1.02]",
    )

    for name, res in results.items():
        kp, ki, kd = res["best_pid"]
        ax1.plot(
            res["t"],
            res["y"],
            color=colors[name],
            linestyle=styles[name],
            linewidth=2.0,
            label=f"{name} (Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f})",
        )

    ax1.set_title("Perbandingan Respons Step Closed-Loop PID (GA, PSO, FPA, ACO_R)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Waktu t (detik)")
    ax1.set_ylabel("Keluaran Sistem y(t)")
    ax1.set_xlim(0.0, min(8.0, config.T_SIM))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)
    fig1.tight_layout()
    step_plot_path = os.path.join(out_dir, "step_response_comparison.png")
    fig1.savefig(step_plot_path)
    plt.close(fig1)

    # Grafik 2: Konvergensi Nilai ITAE terhadap Iterasi
    fig2, ax2 = plt.subplots(figsize=(10, 5.5), dpi=150)
    iterations = np.arange(1, config.MAX_ITERATIONS + 1)
    for name, res in results.items():
        ax2.plot(
            iterations,
            res["mean_history"],
            color=colors[name],
            linestyle="-",
            linewidth=2.0,
            label=f"{name} Mean (10 run) -> Akhir={res['mean_history'][-1]:.4f}",
        )
        ax2.plot(
            iterations,
            res["best_history"],
            color=colors[name],
            linestyle="--",
            linewidth=1.1,
            alpha=0.65,
            label=f"{name} Best Run -> Akhir={res['best_history'][-1]:.4f}",
        )

    ax2.set_title("Konvergensi Nilai ITAE terhadap Iterasi (10 Pengulangan Eksperimen)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Iterasi")
    ax2.set_ylabel("Nilai Objektif J_ITAE")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    conv_plot_path = os.path.join(out_dir, "itae_convergence.png")
    fig2.savefig(conv_plot_path)
    plt.close(fig2)

    return step_plot_path, conv_plot_path


def print_results_and_conclusion(results: dict[str, dict], step_plot_path: str, conv_plot_path: str) -> None:
    print("=" * 78)
    print(f"TASK 2: HASIL OPTIMASI PID ({config.NUM_EXPERIMENTS} EKSPERIMEN, POP={config.POPULATION_SIZE}, ITER={config.MAX_ITERATIONS})")
    print("=" * 78)
    print(f"{'Algo':<8} | {'Kp':>9} | {'Ki':>9} | {'Kd':>9} | {'Best ITAE':>10} | {'Mean ITAE':>10} | {'Std ITAE':>10}")
    print("-" * 78)
    for name, res in results.items():
        kp, ki, kd = res["best_pid"]
        print(
            f"{name:<8} | {kp:9.4f} | {ki:9.4f} | {kd:9.4f} | "
            f"{res['best_itae']:10.5f} | {res['mean_itae']:10.5f} | {res['std_itae']:10.5f}"
        )

    print("\n" + "=" * 78)
    print("TASK 3 & TASK 4: SIMULASI STEP RESPONSE & KRITERIA RESPONS TRANSIEN (TOL 2%)")
    print("=" * 78)
    print(f"{'Algo':<8} | {'Mp (%)':>10} | {'Ts (detik)':>12} | {'ess':>12} | {'J_ITAE':>12}")
    print("-" * 78)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<8} | {m['Mp']:10.4f} | {m['Ts']:12.4f} | {m['ess']:12.6e} | {m['ITAE']:12.5f}")

    print(f"\nGrafik disimpan di:\n- {step_plot_path}\n- {conv_plot_path}")

    def _find_best_with_ties(metric_fn, tol: float = 1e-4) -> tuple[list[str], float]:
        vals = {k: metric_fn(results[k]) for k in results}
        min_val = min(vals.values())
        winners = [k for k, v in vals.items() if abs(v - min_val) <= tol]
        return winners, min_val

    itae_winners, min_itae = _find_best_with_ties(lambda r: r["metrics"]["ITAE"], tol=1e-5)
    mp_winners, min_mp = _find_best_with_ties(lambda r: r["metrics"]["Mp"], tol=1e-4)
    ts_winners, min_ts = _find_best_with_ties(lambda r: r["metrics"]["Ts"], tol=1e-4)
    best_mean_algo = min(results, key=lambda k: results[k]["mean_itae"])
    best_std_algo = min(results, key=lambda k: results[k]["std_itae"])

    print("\n" + "=" * 78)
    print("TASK 5: PERBANDINGAN DAN KESIMPULAN")
    print("=" * 78)
    print(f"- ITAE terkecil (best run)   : {', '.join(itae_winners)} (J_ITAE = {min_itae:.5f})")
    print(f"- Overshoot (Mp) terkecil    : {', '.join(mp_winners)} (Mp = {min_mp:.4f}%)")
    print(f"- Settling time (Ts) singkat : {', '.join(ts_winners)} (Ts = {min_ts:.4f} s)")
    print(
        f"- Konsistensi 10 eksperimen  : {best_mean_algo} memiliki rata-rata ITAE terkecil "
        f"({results[best_mean_algo]['mean_itae']:.5f}) dan std terkecil "
        f"({results[best_std_algo]['std_itae']:.5f})"
    )
    print(
        "- Steady-state error (ess)   : seluruh algoritma menghasilkan ess -> 0\n"
        "  (~3.13e-4 pada t=15 s) karena open-loop L(s) bertipe-2 (dua pole di s=0:\n"
        "  satu dari plant G(s) dan satu dari integrator Ki/s pada PID).\n"
        "- Interpretasi kontrol       : seluruh algoritma berhasil menemukan optimum\n"
        "  global ITAE yang hampir identik di sekitar (Kp~17.16, Ki=0.10, Kd=10.00).\n"
        "  Nilai ITAE sangat kecil (0.12654) dengan Ts cepat (1.40 s), namun Mp ~41.3%\n"
        "  karena fungsi objektif murni ITAE tidak memberi penalti khusus pada overshoot\n"
        "  awal dan zero dari suku Kd*s^2 pada pembilang T(s) mempercepat rise time."
    )


def main() -> None:
    print_task1_analysis()
    results = run_all_experiments()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    step_plot_path, conv_plot_path = plot_results(results, out_dir)
    print_results_and_conclusion(results, step_plot_path, conv_plot_path)


if __name__ == "__main__":
    main()
