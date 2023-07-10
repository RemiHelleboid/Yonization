import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.matrixlib.defmatrix import matrix
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.style.use(["seaborn-paper"])


from impact_ionization import *

""" Computation of the McIntyre solution with a LETI Method

"""


def f1_log(x):
    return -np.log(1.0 - x)

def f(x_line: np.array, electric_field: np.array) -> np.array:
    # Find the index of x in x_line
    alpha_e =  np.array([impact_ionization_rate_electron_van_overstraten_silicon(f) for f in electric_field])
    alpha_h =  np.array([impact_ionization_rate_hole_van_overstraten_silicon(f) for f in electric_field])
    integrand = alpha_e - alpha_h
    f = [np.trapz(integrand[:k], x_line[:k]) for k in range(len(integrand))]
    return np.exp(f)

def integral_Ph(x_line: np.array, electric_field: np.array, Ph_0):
    alpha_h = np.array([impact_ionization_rate_hole_van_overstraten_silicon(f) for f in electric_field])
    func = f(x_line, electric_field)
    numerator = Ph_0 * alpha_h * func
    denominator = Ph_0 * func + 1 - Ph_0
    integrand = numerator / denominator
    integral = np.trapz(integrand, x_line)
    return integral


def numerically_find_Ph_0(x_line: np.array, electric_field: np.array):
    eps = 1e-3
    Xgrid = np.linspace(0, 1.0-eps, 100)
    discrete_flog = f1_log(Xgrid)
    discrete_f_integral = np.array([integral_Ph(x_line, electric_field, x) for x in Xgrid])
    f_difference = interp1d(Xgrid, discrete_flog - discrete_f_integral)
    arg_zeros = fsolve(f_difference, x0=0.3)
    print(f"Zeros: {arg_zeros}")
    return arg_zeros[-1]


    
def read_sl_data(filename):
    print(f"Reading file: {filename}")
    x_line, field = np.loadtxt(filename, delimiter=',', unpack=True)
    # fig, axs = plt.subplots()
    # axs.plot(x_line, field, "-", label="Electric field")
    # axs.set_xlabel("X ($\mu$m)")
    # axs.set_ylabel("Electric field (V/cm)")
    # plt.show()
    return x_line, field

def plot_problem(x_line: np.array, electric_field: np.array):
    # Xgrid represent all the possible values for Ph(0) ! 
    eps = 1e-3
    Xgrid = np.linspace(-1.0, 1.0-eps, 1000)
    flog = f1_log(Xgrid)
    f_integral = np.array([integral_Ph(x_line, electric_field, x) for x in Xgrid])

    print(Xgrid.shape)
    print(flog.shape)
    print(f_integral.shape)
    fig, axs = plt.subplots()

    axs.plot(Xgrid, flog, label='$f_1$')
    axs.plot(Xgrid, f_integral, label='$f_2$')
    axs.legend()
    plt.show()


def extract_Pe_Ph_from_Ptotal(x_line: np.array, electric_field: np.array, Ph0, Ptotal):
    # Ph(x) = 1.0 - (1-Ph(0)) * exp(\int_0^x alpha_h(s)P_total(s) ds)
    alpha_h = np.array([impact_ionization_rate_hole_van_overstraten_silicon(f) for f in electric_field])
    P_tot_alpha_h = alpha_h * Ptotal
    Ph = np.zeros_like(x_line)
    Pe = np.zeros_like(x_line)
    for idx, _ in enumerate(x_line):
        ph = 1.0 - (1 - Ph0) * np.exp(np.trapz(P_tot_alpha_h[:idx], x_line[:idx]))
        Ph[idx] = ph
        Pe[idx] =  1 + (Ptotal[idx] - 1) / (1.0 - ph)
    return Pe, Ph


def Compute_Ptotal(x_line: np.array, electric_field: np.array):
    # plot_problem(x_line, electric_field)
    # Ph0 = 0.24188503345377904
    Ph0 = numerically_find_Ph_0(x_line, electric_field)
    f_func = f(x_line, electric_field)
    P_total = np.zeros_like(x_line)
    for idx, x in enumerate(x_line):
        P_total[idx] = (Ph0 * f_func[idx]) / ((Ph0 * f_func[idx]) + 1 - Ph0)

    Pe, Ph = extract_Pe_Ph_from_Ptotal(x_line, electric_field, Ph0, P_total)

    return Pe, Ph, P_total


# Pt = Pe + Ph - Pe*Ph
# Pt = 1 - (1-Pe) * (1-Ph)
# 1 - Pt = (1-Pe) * (1-Ph)
# (1-Pt) / (1-Ph) = (1-Pe)
# -Pe = (1-Pt) / (1-Ph) - 1
# Pe = 1 - (1-Pt) / (1-Ph)

if __name__ == "__main__":
    print("Weee")
    x_line, field = read_sl_data("SPAD_SL.csv")
    x_line *= 1e2
    # field = field[::-1]
    fig1, axs1 = plt.subplots()
    axs1.plot(x_line, field, label="Electric field")
    plt.show()
    plot_problem(x_line, field)
    numerically_find_Ph_0(x_line, field)
    Pe, Ph, P_total = Compute_Ptotal(x_line, field)
    Re_Ptotal = Pe + Ph - Pe*Ph
    fig, axs = plt.subplots()
    axs.plot(x_line, Pe, label=r"P$_{e}$")
    axs.plot(x_line, Ph, label=r"P$_{h}$")
    axs.plot(x_line, P_total, label=r"P$_{total}$")
    axs.plot(x_line, Re_Ptotal, ls="--", c='k', label=r"P$_{e}$+P$_{h}$-P$_{e}$P$_{h}$")
    axs.legend()
    axs.set_xlabel("X ($\mu$m)")
    axs.set_ylabel("Probability")
    # axs.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    plt.show()
