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
    matrix_A = sparse.dok_matrix((2 * nb_points, 2 * nb_points))
    matrix_A_sparse = sparse.dok_matrix((2 * nb_points, 2 * nb_points))
    list_blocks = []
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
    matrix_A = matrix_A_sparse.tocsr()
    return matrix_A


def assemble_second_member(list_x, list_electric_field, breakdownProbability):
    N = len(list_x)
    second_member = np.zeros(2 * N)
    for index in range(N - 1):
        q_i = compute_elementary_second_member(index, list_x, list_electric_field, breakdownProbability)
        second_member[2 * index] = - q_i[0]
        second_member[2 * index + 1] = - q_i[1]
    # second_member[2 * N - 2] = breakdownProbability[0]
    # second_member[2 * N - 1] = breakdownProbability[2 * N - 1]
    return second_member


def compute_newton_solution(list_x, list_electric_field, GuessbreakdownProbability, tolerance):
    N = len(list_x)
    epoch = 0
    g_0 = 0.0
    breakdownProbability = GuessbreakdownProbability
    W = np.zeros_like(breakdownProbability)
    W_new = np.zeros_like(breakdownProbability)
    lambda_newton = 1.0
    norm_residu = 1e6
    while norm_residu > tolerance and epoch <= 100:
        # print(f"EPOCH    : {epoch}")
        matrix_A = assemble_matrix(list_x, list_electric_field, breakdownProbability)
        second_member = assemble_second_member(list_x, list_electric_field, breakdownProbability)
        W = sparse.linalg.spsolve(matrix_A, second_member)
        norm_residu = np.linalg.norm(W)
        breakdownProbability = breakdownProbability + lambda_newton * W
        epoch += 1
        print(f"Epoch : {epoch}  --->  Current residu norm = {norm_residu}")
    # print(f"Final error : {norm_residu}")
    return breakdownProbability

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



if __name__ == "__main__":
    tolerance = 1e-12
    mesh_line = np.linspace(0.0, 1.8e-4, 1000)
    electric_field = np.array([electric_field_profile.function_electric_field(x) for x in mesh_line])
    boost_ef = 0.93229
    electric_field = boost_ef * electric_field
    x_max_ef = mesh_line[np.argmax(electric_field)]
    initial_guess = np.zeros(2 * len(mesh_line))
    init_initial_guess = mcintyre_model.function_initial_guess(mesh_line, x_max_ef)
    for k in range(len(init_initial_guess[0])):
        initial_guess[2 * k] = init_initial_guess[0][k]
        initial_guess[2 * k + 1] = init_initial_guess[1][k]
    matrix_A = assemble_matrix(mesh_line, electric_field, initial_guess)
    matrix_A_plain = matrix_A.toarray()
    brp_newton = compute_newton_solution(mesh_line, electric_field, initial_guess, tolerance)
    eBreakdownProbability, hBreakdownProbability = brp_new_ton_to_eh_brp(brp_newton)
    total_breakdown_probability = eBreakdownProbability + hBreakdownProbability - eBreakdownProbability * hBreakdownProbability
    plt.plot(mesh_line, eBreakdownProbability, "-", c="b", label="Electron")
    plt.plot(mesh_line, hBreakdownProbability, "--", c="r",  label="Holes")
    plt.plot(mesh_line, total_breakdown_probability, "-", c="g",  label="Total")
    plt.title(f"Max total BrP : {np.max(total_breakdown_probability)}")
    plt.legend()
    plt.show()