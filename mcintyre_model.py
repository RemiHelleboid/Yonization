import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import impact_ionization
import electric_field_profile


def dPe_dx(x, PePh, temperature=300):
    Pe = PePh[0]
    Ph = PePh[1]
    electric_field = [
        electric_field_profile.function_electric_field(x_pos) for x_pos in x]
    # print(x, electric_field)
    gamma_temperature = impact_ionization.compute_gamma_temperature_dependence(
        temperature)
    alpha = [impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        ef, gamma_temperature) for ef in electric_field]
    beta = [impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        ef, gamma_temperature) for ef in electric_field]
    dPe = (1-Pe) * alpha * (Pe + Ph - Pe*Ph)
    dPh = -(1-Ph) * beta * (Pe + Ph - Pe*Ph)
    # plt.plot(electric_field)
    # plt.plot(alpha)
    # plt.plot(beta)
    # plt.show()
    return(np.vstack((dPe, dPh)))


def BoundaryCondition(ya, yb):
    return(np.array([ya[0], yb[-1]]))

def function_initial_guess(x_line_efield, x_max_ef):
    initial_guess_e = np.zeros_like(x_line_efield)
    initial_guess_h = np.zeros_like(x_line_efield)
    for k in range(len(x_line_efield)):
        if x_line_efield[k] > x_max_ef:
            initial_guess_e[k] = 0.5
        else:
            initial_guess_h[k] = 0.2
    return np.vstack((initial_guess_e, initial_guess_h))


def solve_mcintyre(x_line_electric_field, y_line_electric_field):
    # initial_guess = np.zeros((2, x_line_electric_field.size)) + 1.1
    initial_guess = function_initial_guess(x_line_electric_field, 1.36e-4)
    # initial_guess_e = []
    # initial_guess_h = 0.15 * (x_line_electric_field[-1] - x_line_electric_field)
    # initial_guess = np.vstack((initial_guess_e, initial_guess_h))
    initial_guess[0][0] = 0
    initial_guess[1][-1] = 0
    plt.plot(initial_guess[0])
    plt.plot(initial_guess[1])
    plt.show()
    solution = solve_bvp(dPe_dx, BoundaryCondition,
                         x_line_electric_field, initial_guess, max_nodes=10000, tol=1e-6)
    return solution


if __name__ == "__main__":
    mesh_line = np.linspace(0.0, 1.8e-4, 2000)
    electric_field = [
        electric_field_profile.function_electric_field(x) for x in mesh_line]
    sol = solve_mcintyre(mesh_line, electric_field_profile.function_electric_field)
    x_plot = mesh_line
    y1_plot = sol.sol(x_plot)[0]
    y2_plot = sol.sol(x_plot)[1]
    plt.plot(x_plot, y1_plot, "-", c="b", label="Electron")
    plt.plot(x_plot, y2_plot, "--", c="r",  label="Holes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
