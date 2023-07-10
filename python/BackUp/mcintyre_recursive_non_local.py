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

plt.style.use('seaborn-talk')


import mcintyre_model
import mcintyre_newton
import impact_ionization
import electric_field_profile



""" Computation of the McIntyre solution with a recursive method.

"""


def compute_mcintyre_recursive_local(x_line, electric_field, tolerance=1e-6, MaxEpoch = 10, boost=1.0, plot=True):
    print(tolerance)
    electric_field = boost * np.array(electric_field)
    alpha_line = np.array([impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        f) for f in electric_field])
    beta_line = np.array([impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        f) for f in electric_field])
    e_brp_line, h_brp_line = mcintyre_model.function_initial_guess(x_line, x_line[np.argmax(electric_field)])
    
    e_no_brp_line = 1 - e_brp_line
    h_no_brp_line = 1 - h_brp_line

    
    
    dead_space_line_electron = [impact_ionization.compute_dead_space(ef, 1.12) for ef in electric_field]
    dead_space_line_hole = [impact_ionization.compute_dead_space(ef, 3*1.8) for ef in electric_field]

    fig1, axs1 = plt.subplots(2, sharex=True)
    axs1[0].plot(x_line, electric_field, label="Electric Field")
    axs1[1].plot(x_line, dead_space_line_electron, label="Electron Dead Space")
    axs1[1].plot(x_line, dead_space_line_hole, label="Hole Dead Space")
    for ax in axs1:
        ax.legend()
    plt.show()
    e_nobrp_line_new = np.zeros_like(x_line)
    h_nobrp_line_new = np.zeros_like(x_line)
    list_epochs = []
    list_differences = []
    list_max_ebrp = []
    epoch = 0
    difference = 1e6
    if plot:
        fig, axs = plt.subplots(2, figsize=(10, 12))
        axs[0].plot(x_line, e_nobrp_line_new, label="Electron")
        axs[0].plot(x_line, h_no_brp_line, label="Hole")
        axsmax = axs[1].twinx()

    size_x_line = len(x_line)
    while difference >= tolerance and epoch <= MaxEpoch:
        total_brp = e_no_brp_line + h_no_brp_line - e_nobrp_line_new * h_nobrp_line_new
        for index, x_pos in enumerate(x_line, start=1):
            if index == size_x_line -1:
                break
            print(index)
            alpha_line_x = [impact_ionization.dead_space_impact_ionization_rate_electron_van_overstraten_silicon(electric_field[k], 
                                                                            x_pos, x_line[k], dead_space_line_electron[k]) for k in range(size_x_line)]
            beta_line_x = [impact_ionization.dead_space_impact_ionization_rate_hole_van_overstraten_silicon(electric_field[k], 
                                                                            x_pos, x_line[k], dead_space_line_hole[k]) for k in range(size_x_line)]
            
            # axs[1].plot(x_line, beta_line, ls="--", marker="+", alpha=0.85)
            # axs[1].plot(x_line, beta_line_x)
            # axs[1].vlines(x_pos, -1, 2e4, alpha=0.5)
            # plt.pause(0.1)
            
            P_se_x_0 = np.exp(- np.trapz(alpha_line_x[:index]))
            p_e_x_xp = alpha_line_x[index] * np.array([np.exp(-np.trapz(alpha_line_x[index_zero:index])) for index_zero in range(index)])
            
            P_sh_x_0 = np.exp(- np.trapz(beta_line_x[index:]))
            p_h_x_xp = beta_line_x[index] * np.array([np.exp(-np.trapz(beta_line_x[index:index_max])) for index_max in range(size_x_line-index)])
            
            new_no_e_brp = P_se_x_0 + np.trapz((p_e_x_xp * e_nobrp_line_new[:index]) ** 2.0 * h_no_brp_line[:index], x_line[:index])
            new_no_h_brp = P_sh_x_0 + np.trapz((p_h_x_xp * h_nobrp_line_new[index:]) ** 2.0 * e_no_brp_line[index:], x_line[index:])
            
            print(f"{P_se_x_0=}")
            print(f"{P_sh_x_0=}")
            e_nobrp_line_new[index] = new_no_e_brp
            h_nobrp_line_new[index] = new_no_h_brp
        difference = np.linalg.norm(e_nobrp_line_new - e_nobrp_line_new) + np.linalg.norm(h_no_brp_line - h_nobrp_line_new)
        print(difference)
        e_no_brp_line = np.copy(e_nobrp_line_new)
        h_no_brp_line = np.copy(h_nobrp_line_new)
        list_max_ebrp.append(np.max(e_nobrp_line_new))
        epoch += 1
        list_epochs.append(epoch)
        list_differences.append(difference)
        print(f"\rEpoch n° {epoch}  ---->   difference = {difference:2e}  with a maximum of {np.max(e_nobrp_line_new):2e} ", end="", flush=True)
        if plot:
            axs[0].plot(x_line, e_nobrp_line_new, label="Electron")
            axs[0].plot(x_line, h_no_brp_line, ls = "--", label="Hole")
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
            # axsmax.legend()
            axsmax.set_ylim(0, 1.0)
            plt.pause(1.0e-3)
            axs[1].clear()
            axsmax.clear()
    print(f"\rEpoch n° {epoch}  ---->   difference = {difference:2e}  with a maximum of {np.max(e_nobrp_line_new):2e} ", end="", flush=True)
    if plot:
        # fig.clear()
        plt.show()
        fig.savefig("results/McIntyreRecursive.pdf")
    return



if __name__ == "__main__":
    MAX_EPOCH = 60
    BOOST = 1.0
    mesh_line = np.linspace(0.0, 3e-4, 200)
    electric_field = np.array([
        electric_field_profile.function_electric_field(x) for x in mesh_line])
    electric_field *= 0.975
    start_slow_method = timer()
    compute_mcintyre_recursive_local(
        mesh_line, electric_field, 1e-6, MAX_EPOCH, BOOST, True)
    end_slow_method = timer()
    time_compute_slow_method = end_slow_method - start_slow_method
    print(f"Time to compute with the slow method : {time_compute_slow_method}")