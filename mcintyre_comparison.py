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
from scipy import interpolate

import mcintyre_model
import impact_ionization
import electric_field_profile
import mcintyre_newton


def functorize(x_list, y_list):
    f = interpolate.interp1d(x_list, y_list)
    return f


def main_comparison(nb_points, tolerance, plot=False):
    tolerance = 1e-8
    mesh_line = np.linspace(0.0, 1.8e-4, int(nb_points))
    electric_field = [
        electric_field_profile.function_electric_field(x) for x in mesh_line]

    initial_guess = np.zeros(2 * len(mesh_line))
    x_max_ef = mesh_line[np.argmax(electric_field)]
    init_initial_guess = mcintyre_model.function_initial_guess(
        mesh_line, x_max_ef)
    for k in range(len(init_initial_guess[0])):
        initial_guess[2 * k] = init_initial_guess[0][k]
        initial_guess[2 * k + 1] = init_initial_guess[1][k]

    start_newton = timer()
    brp_newton = mcintyre_newton.compute_newton_solution(
        mesh_line, electric_field, initial_guess, tolerance)
    end_newton = timer()
    time_compute_newton = end_newton - start_newton

    start_solve_bvp = timer()
    sol_scipy = mcintyre_model.solve_mcintyre(
        mesh_line, electric_field, tolerance)
    end_solve_bvp = timer()
    time_compute_bvp = end_solve_bvp - start_solve_bvp

    print(
        f"Time to comput the solution with scipy.integrate.solve_bvp : {time_compute_bvp}")
    print(
        f"Time to comput the solution with a Newton method : {time_compute_newton}")
    print(
        f"Acceleration rate of the Newton method : {time_compute_bvp / time_compute_newton}")

    eBreakdownProbability, hBreakdownProbability = mcintyre_newton.brp_new_ton_to_eh_brp(
        brp_newton)
    x_plot = mesh_line
    y1_plot = sol_scipy.sol(x_plot)[0]
    y2_plot = sol_scipy.sol(x_plot)[1]
    dx = mesh_line[1] - mesh_line[0]
    discret_L2_norm = dx * np.linalg.norm(eBreakdownProbability - y1_plot)
    print(f"Norm of the difference of the two solution : {discret_L2_norm}")

    if (plot):
        fig, axs = plt.subplots(1, figsize=(8, 8))
        axs.plot(x_plot, y1_plot, "-", c="b", alpha=0.75,
                 lw=6, label="Electron Scipy")
        axs.plot(x_plot, y2_plot, "-", c="r",
                 alpha=0.75, lw=6, label="Holes Scipy")
        axs.plot(mesh_line, eBreakdownProbability, lw=4,
                 ls="--", c="g", label="Electron Newton")
        axs.plot(mesh_line, hBreakdownProbability, lw=4,
                 ls="--", c="yellow",  label="Holes Newton")
        axs.ticklabel_format(style="scientific", axis="x")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 2))
        axs.set_xlabel("X (a.u.)")
        axs.set_ylabel("Breakdown Probability")
        axs.legend()
        # fig.tight_layout()
        # axs.set_title(
        #     "Comparsion of Scipy scipy.integrate.solve_bvp\n and Newton method for solving McIntyre equations.")
        # fig.show()
        plt.show()
    return time_compute_bvp, time_compute_newton, discret_L2_norm


def compare_time(min_size, max_size, nb_pt):
    tolerance = 1e-8
    list_size = np.geomspace(min_size, max_size, nb_pt)
    list_time_bvp = []
    list_time_newton = []
    list_acceleration = []
    for nb_point in list_size:
        print(f"Comparsion for size : {nb_point}")
        time_bvp, time_newton, norm_diff = main_comparison(nb_point, tolerance)
        list_time_bvp.append(time_bvp)
        list_time_newton.append(time_newton)
        list_acceleration.append(time_bvp / time_newton)
    fig, axs = plt.subplots(2, figsize=(10, 6))
    axs[0].loglog(list_size, list_time_bvp, ls="-", marker="+",
                  label="scipy.integrate.solve_bvp")
    axs[0].loglog(list_size, list_time_newton, ls="-",
                  marker="*", label="Newton method")
    axs[1].loglog(list_size, list_acceleration, ls="-",
                  marker="+", label="Acceleration of Newton method")
    axs[0].legend()
    axs[1].legend()
    plt.show()


def check_convergence(nb_pt, reference_list_x, reference_ebrp, refernce_hbrp):
    tolerance = 1e-8
    mesh_line = np.linspace(0.0, 1.8e-4, int(nb_pt))
    electric_field = [
        electric_field_profile.function_electric_field(x) for x in mesh_line]

    initial_guess = np.zeros(2 * len(mesh_line))
    x_max_ef = mesh_line[np.argmax(electric_field)]
    init_initial_guess = mcintyre_model.function_initial_guess(
        mesh_line, x_max_ef)
    for k in range(len(init_initial_guess[0])):
        initial_guess[2 * k] = init_initial_guess[0][k]
        initial_guess[2 * k + 1] = init_initial_guess[1][k]
    brp_newton = mcintyre_newton.compute_newton_solution(
        mesh_line, electric_field, initial_guess, tolerance)
    sol_scipy = mcintyre_model.solve_mcintyre(
        mesh_line, electric_field, tolerance)
    eBreakdownProbability, hBreakdownProbability = mcintyre_newton.brp_new_ton_to_eh_brp(
        brp_newton)
    x_plot = mesh_line
    eBrp_solvebvp = sol_scipy.sol(x_plot)[0]
    hBrp_solvebvp = sol_scipy.sol(x_plot)[1]

    function_e_brp_solve_bvp = functorize(x_plot, eBrp_solvebvp)
    function_e_brp_newton = functorize(mesh_line, eBreakdownProbability)
    function_h_brp_solve_bvp = functorize(x_plot, hBrp_solvebvp)
    function_h_brp_newton = functorize(mesh_line, hBreakdownProbability)

    X_conv = reference_list_x
    e_brp_solve_bvp_conv = function_e_brp_solve_bvp(X_conv)
    e_brp_solve_newton_conv = function_e_brp_newton(X_conv)
    h_brp_solve_bvp_conv = function_h_brp_solve_bvp(X_conv)
    h_brp_solve_newton_conv = function_h_brp_newton(X_conv)
    # plt.plot(X_conv, reference_ebrp, label="REF E")
    # plt.plot(x_plot, eBrp_solvebvp, marker="+", label="BVP E")
    # plt.plot(mesh_line, eBreakdownProbability, marker="*",label="NEWTON E")
    # plt.legend()
    # plt.pause(1.0e-2)

    dx = reference_list_x[1] - reference_list_x[0]
    Error_L2_e_brp_bvp_solve = np.linalg.norm(e_brp_solve_bvp_conv - reference_ebrp) * np.sqrt(dx)
    Error_L2_e_brp_newton = np.linalg.norm(e_brp_solve_newton_conv - reference_ebrp) * np.sqrt(dx)
    Error_L2_h_brp_bvp_solve = np.linalg.norm(h_brp_solve_bvp_conv - refernce_hbrp) * np.sqrt(dx)
    Error_L2_h_brp_newton = np.linalg.norm(h_brp_solve_newton_conv - refernce_hbrp) * np.sqrt(dx)

    return Error_L2_e_brp_bvp_solve, Error_L2_e_brp_newton, Error_L2_h_brp_bvp_solve, Error_L2_h_brp_newton


def plot_convergence(min_size, max_size, nb_pt):
    tolerance = 1e-8
    list_size = np.geomspace(min_size, max_size, nb_pt)
    list_error_L2_ebrp_sovlebvp = []
    list_error_L2_ebrp_newton = []
    list_error_L2_hbrp_solvebvp = []
    list_error_L2_hbrp_newton = []

    print("Compute refrence solution ...")
    ref_list_x = np.linspace(0.0, 1.8e-4, int(max_size))
    electric_field = [
        electric_field_profile.function_electric_field(x) for x in ref_list_x]
    sol_scipy = mcintyre_model.solve_mcintyre(ref_list_x, electric_field, tolerance)
    ref_ebrp = sol_scipy.sol(ref_list_x)[0]
    ref_hbrp = sol_scipy.sol(ref_list_x)[1]
    for nb_point in list_size:
        print(f"Comparsion for size : {nb_point}")
        Error_L2_e_brp_bvp_solve, Error_L2_e_brp_newton, Error_L2_h_brp_bvp_solve, Error_L2_h_brp_newton = check_convergence(
            nb_point, ref_list_x, ref_ebrp, ref_hbrp)
        list_error_L2_ebrp_sovlebvp.append(Error_L2_e_brp_bvp_solve)
        list_error_L2_ebrp_newton.append(Error_L2_e_brp_newton)
        list_error_L2_hbrp_solvebvp.append(Error_L2_h_brp_bvp_solve)
        list_error_L2_hbrp_newton.append(Error_L2_h_brp_newton)

    fig, axs = plt.subplots(2, figsize=(10, 6))
    # axs[0].loglog(list_size, list_error_L2_ebrp_sovlebvp, ls="-", marker="+",
    #               label="e_breakdown_probability scipy.integrate.solve_bvp")
    axs[0].loglog(list_size, list_error_L2_ebrp_newton, ls="-",
                  marker="*", label="e_breakdown_probability Newton method")
    # axs[1].loglog(list_size, list_error_L2_hbrp_solvebvp, ls="-",
    #               marker="+", label="h_breakdown_probability scipy.integrate.solve_bvp")
    axs[1].loglog(list_size, list_error_L2_hbrp_newton, ls="-",
                  marker="+", label="h_breakdown_probability Newton method")
    axs[0].legend()
    axs[1].legend()
    plt.show()


if __name__ == "__main__":
    # compare_time(10, 10000, 11)
    # plot_convergence(10, 100000, 10)
    main_comparison(2500, 1e-12, True)
