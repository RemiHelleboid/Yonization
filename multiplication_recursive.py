import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.optimize.zeros import _within_tolerance
from scipy.signal import savgol_filter

import impact_ionization
import electric_field_profile

def compute_multiplication(x_line, electric_field, tolerance, boost=1.0, plot=False):
    electric_field = boost * np.array(electric_field)
    multiplication_line = np.zeros_like(x_line)
    alpha_line = np.array([impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(f) for f in electric_field])
    beta_line = np.array([impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(f) for f in electric_field])
    new_multiplication_line = np.copy(multiplication_line)
    MaxEpoch = 100000
    epoch = 0
    difference = 1e6
    list_max_multiplication = []
    list_difference= []
    max_multiplication = 0
    if plot:
        fig, axs = plt.subplots(2, figsize=(10, 6))
    while difference >= tolerance and epoch <= MaxEpoch and max_multiplication<1e10:
        for idx_pt in range(len(mesh_line)):
            multiplication_x = 1.0 + np.trapz(multiplication_line[:idx_pt:] * alpha_line[:idx_pt:], x_line[:idx_pt:]) + np.trapz(multiplication_line[idx_pt::] * beta_line[idx_pt::], x_line[idx_pt::])
            new_multiplication_line[idx_pt] = multiplication_x
            difference += multiplication_x - multiplication_line[idx_pt]
        difference = np.linalg.norm(multiplication_line - new_multiplication_line)
        multiplication_line = np.copy(new_multiplication_line)
        epoch += 1
        max_multiplication = np.max(multiplication_line)
        list_max_multiplication.append(max_multiplication)
        list_difference.append(difference)

        if (epoch%5 == 0) and plot:
            print(f"Epoch n° {epoch}  ---->   difference = {difference:2e}  with a maximum of {max_multiplication:2e} ")
            axs[0].plot(x_line, multiplication_line)
            axs[0].set_title(f"Multiplication convergence over {epoch} epochs with a maximum of {max_multiplication:2e} !")
            axs[1].plot(list_max_multiplication)
            # axs[1].set_yscale("log")
            # axs[1].set_xscale("log")
            axs[1].set_ylabel("Maximum of the multiplication")
            axs[1].set_xlabel("Itteration")
            # plt.legend()
            plt.pause(0.0001)
            axs[1].clear()
    if plot:
        plt.show()
    return max_multiplication, difference

def boost_critical_find(x_line, electric_field, tolerance):
    NbMaxEpoch= 1000
    list_boost = np.linspace(0.9640, 0.9650, 10)
    list_max_mult = []
    list_final_difference = []
    fig, axs = plt.subplots(2, figsize=(10, 8))
    for boost_factor in list_boost:
        print("Compute for boost = ", boost_factor)
        max_mult, final_diff = compute_multiplication(x_line, electric_field, tolerance, boost_factor)
        list_max_mult.append(max_mult)
        list_final_difference.append(final_diff)
    axs[0].plot(list_boost, list_max_mult, marker="+")
    axs[1].plot(list_boost, list_final_difference, marker="+")
    axs[0].set_title("Maximum of the multiplication average")
    axs[1].set_title("Final difference in the convergence process")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")

    plt.show()

if __name__ == "__main__":
    mesh_line = np.linspace(0.0, 1.8e-4, 120)
    electric_field = [electric_field_profile.function_electric_field(x) for x in mesh_line]

    compute_multiplication(mesh_line, electric_field, 1e-4, 0.965, True)
    # boost_critical_find(mesh_line, electric_field, 1e-4)