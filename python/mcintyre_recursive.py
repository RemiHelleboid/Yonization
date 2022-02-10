import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.matrixlib.defmatrix import matrix
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer


import mcintyre_model
import mcintyre_newton
import impact_ionization
import electric_field_profile

""" Computation of the McIntyre solution with a recursive method.

"""


def compute_mcintyre_recursive_local(x_line, electric_field, tolerance, boost=1.0, plot=False):
    electric_field = boost * np.array(electric_field)
    alpha_line = np.array([impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        f) for f in electric_field])
    beta_line = np.array([impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        f) for f in electric_field])
    e_brp_line, h_brp_line = mcintyre_model.function_initial_guess(
        x_line, x_line[np.argmax(electric_field)])
    # e_brp_line = np.zeros_like(x_line) +0.5
    # h_brp_line = np.zeros_like(x_line) + 0.2
    e_brp_line_new = np.zeros_like(x_line)
    h_brp_line_new = np.zeros_like(x_line)
    list_epochs = []
    list_differences = []
    list_max_ebrp = []
    MaxEpoch = 500
    epoch = 0
    difference = 1e6
    if plot:
        fig, axs = plt.subplots(2, figsize=(10, 12))
        axs[0].plot(x_line, e_brp_line, label="Electron")
        axs[0].plot(x_line, h_brp_line, label="Hole")
        axsmax = axs[1].twinx()

    while difference >= tolerance and epoch <= MaxEpoch:
        total_brp = e_brp_line + h_brp_line - e_brp_line * h_brp_line
        for index in range(len(x_line)):
            new_e_brp = e_brp_line[0] + np.trapz(alpha_line[:index] * (
                1.0 - e_brp_line[:index]) * total_brp[:index], x_line[:index])
            new_h_brp = h_brp_line[-1] + np.trapz(beta_line[index:] * (
                1.0 - h_brp_line[index:]) * total_brp[index:], x_line[index:])
            e_brp_line_new[index] = new_e_brp
            h_brp_line_new[index] = new_h_brp
        difference = (1.0 / (np.max(e_brp_line) + np.max(h_brp_line))) * np.linalg.norm(
            e_brp_line - e_brp_line_new) + np.linalg.norm(h_brp_line - h_brp_line_new)
        e_brp_line = np.copy(e_brp_line_new)
        h_brp_line = np.copy(h_brp_line_new)
        list_max_ebrp.append(np.max(e_brp_line))
        epoch += 1
        list_epochs.append(epoch)
        list_differences.append(difference)
        # print(f"\rEpoch n° {epoch}  ---->   difference = {difference:2e}  with a maximum of {np.max(e_brp_line):2e} ", end="", flush=True)
        if plot:
            axs[0].plot(x_line, e_brp_line, label="Electron")
            axs[0].plot(x_line, h_brp_line, ls = "--", label="Hole")
            axs[0].set_title(
                f"Recursive soluions of the McIntyre problem after {epoch} epochs")
            axs[0].set_xlabel("X (a.u.)")
            axs[0].set_ylabel("Breakdown Probability")
            axs[0].set_ylim(-0.1, )
            axs[1].plot(list_epochs, list_differences, label = "Error between sucessive iterations")
            axs[1].set_title("Total difference between succesive itterations.")
            axs[1].set_yscale("log")
            axsmax.plot(list_epochs, list_max_ebrp, ls="--", label = "max electron breakdown probability")
            axs[1].legend()
            axsmax.set_ylim(0, 1.0)
            plt.pause(1.0e-3)
            axs[1].clear()
            axsmax.clear()
            # fig.clear()
            plt.show()
            fig.savefig("McIntyreRecursive.pdf")
    return


def fast_compute_mcintyre_recursive_local(x_line, electric_field, tolerance, boost=1.0, plot=False):
    electric_field = boost * np.array(electric_field)
    alpha_line = np.array([impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        f) for f in electric_field])
    beta_line = np.array([impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        f) for f in electric_field])
    e_brp_line, h_brp_line = mcintyre_model.function_initial_guess(
        x_line, x_line[np.argmax(electric_field)])
    # e_brp_line = np.zeros_like(x_line) +0.5
    # h_brp_line = np.zeros_like(x_line) + 0.2
    e_brp_line_new = np.zeros_like(x_line)
    h_brp_line_new = np.zeros_like(x_line)
    list_epochs = []
    list_differences = []
    list_max_ebrp = []
    MaxEpoch = 1000
    epoch = 0
    difference = 1e6
    size_line = len(x_line)
    if plot:
        fig, axs = plt.subplots(2, figsize=(10, 12))
        axs[0].plot(x_line, e_brp_line, label="Electron")
        axs[0].plot(x_line, h_brp_line, label="Hole")
        axsmax = axs[1].twinx()

    dx = x_line[1] - x_line[0]
    while difference >= tolerance and epoch <= MaxEpoch:
        total_brp = e_brp_line + h_brp_line - e_brp_line * h_brp_line
        sum_integral_electron = 0.0

        sum_integral_hole = 0.0
        line_integral_holes_2 = np.zeros(size_line)
        line_integral_holes = np.zeros(size_line)
        for index in range(1, size_line):
            sum_integral_hole += dx * (beta_line[index] * (1.0 - h_brp_line[index]) * total_brp[index])
            line_integral_holes[index] = sum_integral_hole
            # line_integral_holes[index] = np.trapz(beta_line[index:] * (1.0 - h_brp_line[index:]) * total_brp[index:], x_line[index:])
        line_integral_holes = line_integral_holes[-1] - line_integral_holes
        # plt.plot(line_integral_holes_2)
        # plt.plot(line_integral_holes, ls="--")
        # plt.show()

        for index in range(0, size_line):
            sum_integral_electron += dx * alpha_line[index] * (1.0 - e_brp_line[index]) * total_brp[index]
            new_e_brp = e_brp_line[0] + sum_integral_electron
            new_h_brp = h_brp_line[-1] +  line_integral_holes[index]
            e_brp_line_new[index] = new_e_brp
            h_brp_line_new[index] = new_h_brp

        difference = (1.0 / (np.max(e_brp_line) + np.max(h_brp_line))) * np.linalg.norm(
            e_brp_line - e_brp_line_new) + np.linalg.norm(h_brp_line - h_brp_line_new)
        e_brp_line = np.copy(e_brp_line_new)
        h_brp_line = np.copy(h_brp_line_new)
        list_max_ebrp.append(np.max(e_brp_line))
        epoch += 1
        list_epochs.append(epoch)
        list_differences.append(difference)
        # print(f"\rEpoch n° {epoch}  ---->   difference = {difference:2e}  with a maximum of {np.max(e_brp_line):2e} ", end="", flush=True)
        if plot:
            axs[0].plot(x_line, e_brp_line, label="Electron")
            axs[0].plot(x_line, h_brp_line, ls = "--", label="Hole")
            axs[0].set_title(
                f"Recursive soluions of the McIntyre problem after {epoch} epochs")
            axs[0].set_xlabel("X (a.u.)")
            axs[0].set_ylabel("Breakdown Probability")
            axs[0].set_ylim(-0.1, 1)
            axs[1].plot(list_epochs, list_differences, label = "Error between sucessive iterations")
            axs[1].set_title("Total difference between succesive itterations.")
            axs[1].set_yscale("log")
            axsmax.plot(list_epochs, list_max_ebrp, ls="--", label = "max electron breakdown probability")
            axs[1].legend()
            axsmax.set_ylim(0, 1.0)
            plt.pause(1.0e-3)
            axs[1].clear()
            axsmax.clear()
            # fig.clear()
            plt.show()
            fig.savefig("McIntyreRecursive.pdf")
    return e_brp_line, h_brp_line



if __name__ == "__main__":
    mesh_line = np.linspace(0.0, 3e-4, 5000)
    electric_field = np.array([
        electric_field_profile.function_electric_field(x) for x in mesh_line])
    electric_field *= 0.925
    start_slow_method = timer()
    # compute_mcintyre_recursive_local(
    #     mesh_line, electric_field, 1e-6, 1.0, False)
    end_slow_method = timer()
    time_compute_slow_method = end_slow_method - start_slow_method
    print(f"Time to compute with the slow method : {time_compute_slow_method}")

    start_fast_method = timer()
    fast_compute_mcintyre_recursive_local(
        mesh_line, electric_field, 1e-9, 1.0, False)
    end_fast_method = timer()
    time_compute_fast_method = end_fast_method - start_fast_method
    print(f"Time to compute with the fast method : {time_compute_fast_method}")
    print(f"Acceleration : {time_compute_slow_method/time_compute_fast_method}")

    x_max_ef = mesh_line[np.argmax(electric_field)]
    initial_guess = np.zeros(2 * len(mesh_line))
    init_initial_guess = mcintyre_model.function_initial_guess(mesh_line, x_max_ef)
    for k in range(len(init_initial_guess[0])):
        initial_guess[2 * k] = init_initial_guess[0][k]
        initial_guess[2 * k + 1] = init_initial_guess[1][k]
    brp_newton = mcintyre_newton.compute_newton_solution(mesh_line, electric_field, initial_guess, 1e-9)
    print(f"Max newton eBreakdownProbability : {np.max(brp_newton)}")
    eBreakdownProbability, hBreakdownProbability = mcintyre_newton.brp_new_ton_to_eh_brp(brp_newton)
    total_breakdown_probability = eBreakdownProbability + hBreakdownProbability - eBreakdownProbability * hBreakdownProbability
    plt.plot(mesh_line, eBreakdownProbability, "-", c="b", label="Electron")
    plt.plot(mesh_line, hBreakdownProbability, "--", c="r",  label="Holes")
    plt.show()