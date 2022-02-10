import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import impact_ionization
import electric_field_profile


boost_alpha = 1.0


def dPe_dx(x, PePh, temperature=300):
    Pe = PePh[0]
    Ph = PePh[1]
    electric_field = np.array([
        electric_field_profile.function_electric_field(x_pos) for x_pos in x])
    electric_field *= boost_alpha
    # print(x, electric_field)
    gamma_temperature = impact_ionization.compute_gamma_temperature_dependence(
        temperature)
    alpha = [impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        ef, gamma_temperature) for ef in electric_field]
    beta = [impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        ef, gamma_temperature) for ef in electric_field]
    dPe = (1-Pe) * alpha * (Pe + Ph - Pe*Ph)
    dPh = -(1-Ph) * beta * (Pe + Ph - Pe*Ph)
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


def solve_mcintyre(x_line_electric_field, y_line_electric_field, tolerance):
    # initial_guess = np.zeros((2, x_line_electric_field.size)) + 1.1
    initial_guess = function_initial_guess(x_line_electric_field, 1.36e-4)
    initial_guess[0][0] = 0
    initial_guess[1][-1] = 0
    solution = solve_bvp(dPe_dx, BoundaryCondition,
                         x_line_electric_field, initial_guess, max_nodes=len(x_line_electric_field), tol=tolerance, verbose=2)
    return solution


if __name__ == "__main__":
    mesh_line = np.linspace(0.0, 1.8e-4, 1000)
    electric_field = np.array(
        [electric_field_profile.function_electric_field(x) for x in mesh_line])
    boost_ef = 0.93229
    electric_field = boost_ef * electric_field
    tolerance = 1e-12
    sol = solve_mcintyre(mesh_line, electric_field, tolerance)
    x_plot = mesh_line
    y1_plot = sol.sol(x_plot)[0]
    y2_plot = sol.sol(x_plot)[1]

    fig, axs = plt.subplots(1, figsize=(8, 8))
    axs.plot(x_plot, y1_plot, "-", c="b", label="Electron")
    axs.plot(x_plot, y2_plot, "--", c="r",  label="Holes")
    axs.ticklabel_format(style="scientific", axis="x")
    axs.ticklabel_format(axis='x', style='sci', scilimits=(0, 2))
    axs.set_xlabel("X (a.u.)")
    axs.set_ylabel("Breakdown Probability")
    axs.legend()
    plt.show()
