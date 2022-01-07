import numpy as np
import matplotlib.pyplot as plt
from math import *

from numpy.core.function_base import geomspace

boltzmann_constant = 1.380649000000000092e-23


def compute_gamma_temperature_dependence(temperature):
    ref_temperature = 300.0
    # reduced planck constant x w_optical_phonon
    h_bar_omega = 0.063
    nominator = tanh(h_bar_omega / (2 * boltzmann_constant * ref_temperature))
    denominator = tanh(h_bar_omega / (2 * boltzmann_constant * temperature))
    gamma = nominator / denominator
    return gamma


def impact_ionization_rate_electron_van_overstraten_silicon(F_ava, gamma_temperature_dependence = 1.0):
    E_threshold = 0.0
    if (F_ava <= E_threshold):
        return 0.0
    else:
        a_e = 7.03e5
        b_e = 1.231e6
        imapact_ionization_e = gamma_temperature_dependence * a_e * exp(-gamma_temperature_dependence * b_e / F_ava)
        return imapact_ionization_e


def impact_ionization_rate_hole_van_overstraten_silicon(F_ava, gamma_temperature_dependence = 1.0):
    E_threshold = 0.0
    E_0 = 4.0e5
    if (F_ava <= E_threshold):
        return 0.0
    elif (F_ava <= E_0):
        a_h = 1.582e6
        b_h = 2.036e6
        imapact_ionization_h = gamma_temperature_dependence * a_h * exp(-gamma_temperature_dependence * b_h / F_ava)
        return imapact_ionization_h
    else:
        a_h = 6.71e5
        b_h = 1.693e6
        imapact_ionization_h = gamma_temperature_dependence * a_h * exp(-gamma_temperature_dependence * b_h / F_ava)
    return imapact_ionization_h


if __name__ == "__main__":
    TEMP = 300
    gamma_temp = compute_gamma_temperature_dependence(TEMP)
    inverse_field_range = 1.0 / np.linspace(1e5, 1e8, 1000)
    electron_ii_coefs = [impact_ionization_rate_electron_van_overstraten_silicon(
        field, gamma_temp) for field in 1.0 / inverse_field_range]
    hole_ii_coefs = [impact_ionization_rate_hole_van_overstraten_silicon(
        field, gamma_temp) for field in 1.0 / inverse_field_range]
    plt.plot(inverse_field_range, electron_ii_coefs, lw=2, c='b',
             label="Electron")
    plt.plot(inverse_field_range, hole_ii_coefs, lw=2, c='r',
             label="Holes")
    plt.xlabel("Inverse Electric Field ($cm\cdot V^{-1}$)")
    plt.ylabel("Impact ionization coefficients ($cm^{-1}$)")
    plt.legend()
    plt.yscale("log")
    plt.ylim(1e1, 5e5)
    plt.xlim(0, 5e-6)
    plt.title(f"Impact ionization coefficients in Silicon at {TEMP}K")
    plt.show()
