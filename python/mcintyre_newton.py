import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.matrixlib.defmatrix import matrix
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


import mcintyre_model
import impact_ionization
import electric_field_profile

""" Computation of the McIntyre solution with a Newton Method

"""


def function_dP_e(P_e, P_h, alpha_e):
    return (1 - P_e) * alpha_e * (P_e + P_h - P_e * P_h)


def function_dP_h(P_e, P_h, alpha_h):
    return -(1 - P_h) * alpha_h * (P_e + P_h - P_e * P_h)

def function_initial_guess(x_line_efield, x_max_ef):
    initial_guess_e = np.zeros_like(x_line_efield)
    initial_guess_h = np.zeros_like(x_line_efield)
    for k in range(len(x_line_efield)):
        if x_line_efield[k] > x_max_ef:
            initial_guess_e[k] = 0.9
        else:
            initial_guess_h[k] = 0.4
    return np.vstack((initial_guess_e, initial_guess_h))


def compute_elementary_second_member(index_point, list_x, list_electric_field, breakdownProbability):
    dx = list_x[index_point + 1] - list_x[index_point]
    e_brp_i = breakdownProbability[2 * index_point]
    e_brp_ipp = breakdownProbability[2 * index_point + 2]
    h_brp_i = breakdownProbability[2 * index_point + 1]
    h_brp_ipp = breakdownProbability[2 * index_point + 3]
    y_e = (1.0 / dx) * (e_brp_ipp - e_brp_i)
    y_h = (1.0 / dx) * (h_brp_ipp - h_brp_i)
    alpha_e = impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        list_electric_field[index_point])
    alpha_h = impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        list_electric_field[index_point])
    f_e = 0.5 * (function_dP_e(e_brp_i, h_brp_i, alpha_e) +
                 function_dP_e(e_brp_ipp, h_brp_ipp, alpha_e))
    f_h = 0.5 * (function_dP_h(e_brp_i, h_brp_i, alpha_h) +
                 function_dP_h(e_brp_ipp, h_brp_ipp, alpha_h))
    element_e = y_e - f_e
    element_h = y_h - f_h
    return np.array([element_e, element_h])


def compute_elementary_matrix(index_point, P_e, P_h, list_electric_field):
    alpha_e = impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        list_electric_field[index_point])
    alpha_h = impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        list_electric_field[index_point])
    a_00 = alpha_e * (1 - 2 * P_e - 2 * P_h + 2 * P_e * P_h)
    a_01 = alpha_e * (1 - 2 * P_e + P_e * P_e)
    a_10 = alpha_h * (-1 + 2 * P_h - P_h * P_h)
    a_11 = alpha_h * (-1 + 2 * P_e + 2 * P_h - 2 * P_e * P_h)
    return np.array([a_00, a_01, a_10, a_11])


def assemble_matrix(list_x, list_electric_field, breakdownProbability):
    one_half = 0.5
    nb_points = len(list_x)
    matrix_A_sparse = np.zeros((2 * nb_points, 2 * nb_points))
    for index in range(nb_points - 1):
        x   = list_x[index]
        xpp = list_x[index + 1]
        dx  = xpp - x
        inv_dx = 1.0 / dx
        elementary_second_member_i = compute_elementary_matrix(index, breakdownProbability[2 * index], breakdownProbability[2 * index + 1], list_electric_field)
        elementary_second_member_ipp = compute_elementary_matrix(index + 1, breakdownProbability[2 * index + 2], breakdownProbability[2 * index + 3], list_electric_field)
        matrix_A_sparse[2 * index, 2 * index] = - inv_dx - one_half * elementary_second_member_i[0]
        matrix_A_sparse[2 * index, 2 * index + 1] = - one_half * elementary_second_member_i[1]
        matrix_A_sparse[2 * index + 1, 2 * index] = - one_half * elementary_second_member_i[2]
        matrix_A_sparse[2 * index + 1, 2 * index + 1] = - inv_dx - one_half * elementary_second_member_i[3]
        matrix_A_sparse[2 * index, 2 * index + 2] = inv_dx - one_half * elementary_second_member_ipp[0]
        matrix_A_sparse[2 * index, 2 * index + 3] = - one_half * elementary_second_member_ipp[1]
        matrix_A_sparse[2 * index + 1, 2 * index + 2] = - one_half * elementary_second_member_ipp[2]
        matrix_A_sparse[2 * index + 1, 2 * index + 3] = inv_dx - one_half * elementary_second_member_ipp[3]
    matrix_A_sparse[2 * nb_points - 1, 2 * nb_points - 1] = 1.0
    matrix_A_sparse[2 * nb_points - 2, 0] = 1.0
    matrix_A = sparse.csr_matrix(matrix_A_sparse)
    return matrix_A


def assemble_second_member(list_x, list_electric_field, breakdownProbability):
    N = len(list_x)
    second_member = np.zeros(2 * N)
    for index in range(N - 1):
        q_i = compute_elementary_second_member(index, list_x, list_electric_field, breakdownProbability)
        second_member[2 * index] = - q_i[0]
        second_member[2 * index + 1] = - q_i[1]
    second_member[2 * N - 2] = breakdownProbability[0]
    second_member[2 * N - 1] = breakdownProbability[2 * N - 1]
    return second_member


def brp_new_ton_to_eh_brp(breakdown_probability_newton):
    if (len(breakdown_probability_newton) % 2 != 0):
        print("Error the vector breakdown_probability_newton must havean even size")
    N = len(breakdown_probability_newton) // 2
    e_brp = np.zeros(N)
    h_brp = np.zeros(N)
    for k in range(N):
        e_brp[k] = breakdown_probability_newton[2 * k]
        h_brp[k] = breakdown_probability_newton[2 * k + 1]
    return (e_brp, h_brp)



def compute_newton_solution(list_x, list_electric_field, initial_guess=None, tolerance=1e-6):
    N = len(list_x)
    epoch = 0
    g_0 = 0.0
    if initial_guess is None:
        initial_guess = function_initial_guess(list_x, list_x[np.argmax(np.abs(list_electric_field))])
    breakdownProbability = np.zeros(2 * N)
    for k in range(N):
        breakdownProbability[2 * k] = initial_guess[0][k]
        breakdownProbability[2 * k + 1] = initial_guess[1][k]
    W = np.zeros_like(breakdownProbability)
    W_new = np.zeros_like(breakdownProbability)
    lambda_newton = 0.95
    norm_residu = 1e6
    while norm_residu > tolerance and epoch <= 100:
        # print(f"EPOCH    : {epoch}")
        matrix_A = assemble_matrix(list_x, list_electric_field, breakdownProbability)
        second_member = assemble_second_member(list_x, list_electric_field, breakdownProbability)
        W = sparse.linalg.spsolve(matrix_A, second_member)
        norm_residu = np.linalg.norm(W)
        breakdownProbability = breakdownProbability + lambda_newton * W
        epoch += 1
        print(f"\rEpoch : {epoch}  --->  Current residu norm = {norm_residu}", end="", flush=True)
    # print(f"Final error : {norm_residu}")
    return brp_new_ton_to_eh_brp(breakdownProbability)



if __name__ == "__main__":
    tolerance = 1e-8
    mesh_line = np.linspace(0.0, 1.8e-4, 1000)
    electric_field = np.array([electric_field_profile.function_electric_field(x) for x in mesh_line])
    boost_ef = 1.0
    electric_field = boost_ef * electric_field
    x_max_ef = mesh_line[np.argmax(electric_field)]
    eBreakdownProbability, hBreakdownProbability = compute_newton_solution(mesh_line, electric_field, tolerance=tolerance)
    total_breakdown_probability = eBreakdownProbability + hBreakdownProbability - eBreakdownProbability * hBreakdownProbability
    plt.plot(mesh_line, electric_field/np.max(electric_field))
    plt.plot(mesh_line, eBreakdownProbability, "-", c="b", label="Electron")
    plt.plot(mesh_line, hBreakdownProbability, "--", c="r",  label="Holes")
    plt.plot(mesh_line, total_breakdown_probability, "-", c="g",  label="Total")
    plt.title(f"Max total BrP : {np.max(total_breakdown_probability)}")
    plt.legend()
    plt.show()