import numpy as np
import matplotlib.pyplot as plt
from math import *
import impact_ionization
import electric_field_profile



def pdf_electron_survive(x, electric_field_function, temperature=300):
    gamma_temperature = impact_ionization.compute_gamma_temperature_dependence(
        temperature)
    electric_field_local = electric_field_function(x)
    alpha = impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(electric_field_local, gamma_temperature)
    f = alpha * exp(-alpha * x)
    return f

    
def pdf_hole_survive(x, electric_field_function, temperature=300):
    gamma_temperature = impact_ionization.compute_gamma_temperature_dependence(
        temperature)
    electric_field_local = electric_field_function(x)
    beta = impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(electric_field_local, gamma_temperature)
    f = beta * exp(-beta * x)
    return f



if __name__ == "__main__":
    mesh_line = np.linspace(0.0, 1.8e-4, 2000)
    electric_field = [
        electric_field_profile.function_electric_field(x) for x in mesh_line]
    pdf_electron = [pdf_electron_survive(x, electric_field_profile.function_electric_field) for x in mesh_line]
    pdf_hole = [pdf_hole_survive(x, electric_field_profile.function_electric_field) for x in mesh_line]
    plt.plot(mesh_line, np.array(electric_field)/100.0, c="orange", ls=":", label="EF")
    plt.plot(mesh_line, pdf_electron, c="b", ls="-", label="Electron")
    plt.plot(mesh_line, pdf_hole, c="r", ls="--", label="Hole")
    plt.legend()
    plt.show()
