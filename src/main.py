import csv
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from pid_system import compute_transient_metrics, simulate_single_step
from GA_PID import GeneticAlgorithmPID
from ACO_PID import AntColonyOptimizationPID
from PSO_PID import ParticleSwarmOptimizationPID
from FPA_PID import FlowerPollinationAlgorithmPID


def iterasi_konvergen(history, tol=0.01):
    idx = np.where(history <= history[-1] * (1 + tol))[0]
    return int(idx[0] + 1) if len(idx) > 0 else len(history)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)

    solvers = {
        "GA_PID": GeneticAlgorithmPID(),
        "ACO_PID": AntColonyOptimizationPID(),
        "PSO_PID": ParticleSwarmOptimizationPID(),
        "FPA_PID": FlowerPollinationAlgorithmPID(),
    }

    results = {}
    for algo_idx, (name, solver) in enumerate(solvers.items()):
        print(f"Menjalankan {name} ({config.NUM_EXPERIMENTS} run)...", flush=True)
        params_list, itae_list, hist_list, time_list = [], [], [], []

        for exp in range(config.NUM_EXPERIMENTS):
            seed = config.RANDOM_SEED + algo_idx * 100 + exp
            t0 = time.perf_counter()
            best_pos, best_val, hist = solver.optimize(seed=seed)
            time_list.append(time.perf_counter() - t0)
            params_list.append(best_pos)
            itae_list.append(best_val)
            hist_list.append(hist)

        itae_arr = np.array(itae_list)
        hist_arr = np.array(hist_list)
        best_idx = int(np.argmin(itae_arr))
        best_pid = params_list[best_idx]
        mean_hist = np.mean(hist_arr, axis=0)

        t, y, _ = simulate_single_step(best_pid[0], best_pid[1], best_pid[2])
        results[name] = {
            "pid": best_pid,
            "best_itae": float(itae_arr[best_idx]),
            "mean_itae": float(np.mean(itae_arr)),
            "std_itae": float(np.std(itae_arr)),
            "time_mean": float(np.mean(time_list)),
            "best_hist": hist_arr[best_idx],
            "mean_hist": mean_hist,
            "conv_iter": iterasi_konvergen(mean_hist),
            "t": t,
            "y": y,
            "metrics": compute_transient_metrics(t, y),
        }

    # --- Tabel A, B, C ---
    print("\n=== TABEL A: PARAMETER PID OPTIMAL ===")
    print(f"{'Algoritma':<10} | {'Kp':>9} | {'Ki':>9} | {'Kd':>9}")
    for name, r in results.items():
        kp, ki, kd = r["pid"]
        print(f"{name:<10} | {kp:9.4f} | {ki:9.4f} | {kd:9.4f}")

    print("\n=== TABEL B: RESPONS TRANSIEN (TOLERANSI 2%) ===")
    print(f"{'Algoritma':<10} | {'Mp (%)':>9} | {'Ts (s)':>9} | {'ess':>11} | {'J_ITAE':>9}")
    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<10} | {m['Mp']:9.4f} | {m['Ts']:9.4f} | {m['ess']:11.4e} | {m['ITAE']:9.5f}")

    print("\n=== TABEL C: KONSISTENSI OPTIMASI (10 RUN) ===")
    print(f"{'Algoritma':<10} | {'Best':>9} | {'Mean':>9} | {'Std':>9} | {'Waktu (s)':>9} | {'Konvergen':>9}")
    for name, r in results.items():
        print(
            f"{name:<10} | {r['best_itae']:9.5f} | {r['mean_itae']:9.5f} | "
            f"{r['std_itae']:9.5f} | {r['time_mean']:9.4f} | {r['conv_iter']:>9d}"
        )

    # --- Simpan Tabel CSV ---
    csv_path = os.path.join(out_dir, "tabel_hasil_optimasi.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Algoritma", "Kp", "Ki", "Kd", "Mp (%)", "Ts (detik)", "ess", "J_ITAE", "Best ITAE", "Mean ITAE", "Std ITAE", "Waktu/Run (s)", "Konvergen (it)"])
        for name, r in results.items():
            kp, ki, kd = r["pid"]
            m = r["metrics"]
            w.writerow([name, f"{kp:.4f}", f"{ki:.4f}", f"{kd:.4f}", f"{m['Mp']:.4f}", f"{m['Ts']:.4f}", f"{m['ess']:.6e}", f"{m['ITAE']:.5f}", f"{r['best_itae']:.5f}", f"{r['mean_itae']:.5f}", f"{r['std_itae']:.5f}", f"{r['time_mean']:.4f}", r["conv_iter"]])

    # --- Grafik 1: Respons Step ---
    colors = {"GA_PID": "#1f77b4", "ACO_PID": "#9467bd", "PSO_PID": "#d62728", "FPA_PID": "#2ca02c"}
    styles = {"GA_PID": "-", "ACO_PID": ":", "PSO_PID": "--", "FPA_PID": "-."}
    zorders = {"GA_PID": 3, "FPA_PID": 4, "PSO_PID": 5, "ACO_PID": 6}
    widths = {"GA_PID": 2.0, "FPA_PID": 2.0, "PSO_PID": 2.0, "ACO_PID": 2.6}

    t_ref = next(iter(results.values()))["t"]
    plt.figure(figsize=(10, 5.5), dpi=180)
    plt.plot(t_ref, np.ones_like(t_ref), "k--", lw=1.5, label="Referensi r(t) = 1")
    plt.axhspan(0.98, 1.02, color="gray", alpha=0.14, label="Pita Toleransi 2% [0.98, 1.02]")
    for name, r in results.items():
        kp, ki, kd = r["pid"]
        plt.plot(r["t"], r["y"], color=colors[name], ls=styles[name], lw=widths[name], zorder=zorders[name], label=f"{name} (Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f})")
    plt.title("Perbandingan Respons Step Closed-Loop PID (GA, ACO, PSO, FPA)")
    plt.xlabel("Waktu t (detik)")
    plt.ylabel("Keluaran Sistem y(t)")
    plt.xlim(0, 6)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_response_comparison.png"))
    plt.close()

    # --- Grafik 2: Konvergensi ITAE ---
    iters = np.arange(1, config.MAX_ITERATIONS + 1)
    plt.figure(figsize=(10, 5.5), dpi=180)
    for name, r in results.items():
        plt.plot(iters, r["mean_hist"], color=colors[name], lw=2.0, label=f"{name} Mean -> {r['mean_hist'][-1]:.5f} (it-{r['conv_iter']})")
        plt.plot(iters, r["best_hist"], color=colors[name], ls="--", lw=1.1, alpha=0.55, label=f"{name} Best -> {r['best_hist'][-1]:.5f}")
    plt.title("Konvergensi Nilai ITAE terhadap Iterasi (10 Pengulangan Eksperimen)")
    plt.xlabel("Iterasi")
    plt.ylabel("Nilai Objektif J_ITAE")
    plt.yscale("log")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "itae_convergence.png"))
    plt.close()

    print(f"\nSelesai. Grafik dan CSV tersimpan di folder: {out_dir}")


if __name__ == "__main__":
    main()
