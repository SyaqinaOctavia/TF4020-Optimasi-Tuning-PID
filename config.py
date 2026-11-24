KP_MAX = 100
KP_MIN = 0.1
KI_MAX = 100
KI_MIN = 0.1
KD_MAX = 10
KD_MIN = 0.001

# Parameter yang dicari X = [Kp, Ki, Kd]

# Plant: G(s) = 10 / (s * (s + 2) * (s + 5)) = 10 / (s^3 + 7s^2 + 10s)
TRANSFER_FUNC = (lambda s: 10 / (s * (s + 2) * (s + 5)))
# et = rt - yt -> rt = sinyal referensi = 1, yt = respons keluaran sistem
# fungsi objektif optimasi: Integral of Time-weighted Absolute Error (ITAE)
# OBJECTIVE_FUNCTION = lambda t: integral(t, lambda tau: abs(error(tau)) * tau)  # Integral of Time-weighted Absolute Error (ITAE)
# untuk simulasi numerik pakai sigma/sum jangan integral, karena integral tidak bisa dihitung secara numerik

"""
Parameter Awal dan nilainya:
Jumlah populasi/agen = 30
Maksimum iterasi = 100
Jumlah pengulangan eksperimen = 10
Jumlah parameter PID: 3
Swtiching probability FPA = 0.8
Inertia weight PSO = 0.9 -> 0,4
"""

POPULATION_SIZE = 30
MAX_ITERATIONS = 100
NUM_EXPERIMENTS = 10
NUM_PARAMETERS = 3
FPA_SWITCHING_PROB = 0.8
PSO_W_MAX = 0.9
PSO_W_MIN = 0.4

# Parameter simulasi numerik respons step
DT = 0.01
T_SIM = 15.0
REF_VALUE = 1.0
SETTLING_TOL = 0.02
RANDOM_SEED = 42
