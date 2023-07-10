import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


import mcintyre_model
import impact_ionization
import electric_field_profile
import mcintyre_newton
import mcintyre_recursive
import mcintyre_leti

import scienceplots
plt.style.use(['science', 'grid'])


def get_total_breakdown_probability(eBreakdownProbability, hBreakdownProbability):
    return eBreakdownProbability + hBreakdownProbability - eBreakdownProbability * hBreakdownProbability

def get_all_values():
    tolerance = 1e-12
    mesh_line = np.linspace(0.0, 4.5e-4, 1000)
    electric_field = np.array([electric_field_profile.function_electric_field(x) for x in mesh_line])

    # filename = "VDEVICE.csv"
    # x, y = np.loadtxt(filename, delimiter=',', unpack=True, skiprows=1)
    # mesh_line = np.linspace(np.min(x), np.max(x), int(nb_points))
    # electric_field = functorize(x, y)(mesh_line)
    # mesh_line = x
    # electric_field = y
    
    Newton_eBreakdownProbability, Newton_hBreakdownProbability = mcintyre_newton.compute_newton_solution(mesh_line, electric_field, tolerance=tolerance)
    total_breakdown_probability = get_total_breakdown_probability(Newton_eBreakdownProbability, Newton_hBreakdownProbability)
    Recursive_eBrP, Recursive_hBrP = mcintyre_recursive.fast_compute_mcintyre_recursive_local(mesh_line, electric_field, tolerance=tolerance, plot=False)
    Leti_eBrP, Leti_hBrP, P_total = mcintyre_leti.Compute_Ptotal(mesh_line, electric_field)

    sol = mcintyre_model.solve_mcintyre(mesh_line, electric_field, tolerance)
    SciPy_eBrP = sol.sol(mesh_line)[0]
    SciPy_hBrP = sol.sol(mesh_line)[1]

    alpha_e = [impact_ionization.impact_ionization_rate_electron_van_overstraten_silicon(
        electric_field[index_point]) for index_point in range(len(electric_field))]
    alpha_h = [impact_ionization.impact_ionization_rate_hole_van_overstraten_silicon(
        electric_field[index_point]) for index_point in range(len(electric_field))]

    dict_results = {"mesh_line": mesh_line,
                    "electric_field": electric_field,
                    "Newton_eBrP": Newton_eBreakdownProbability,
                    "Newton_hBrP": Newton_hBreakdownProbability,
                    "Recursive_eBrP": Recursive_eBrP,
                    "Recursive_hBrP": Recursive_hBrP,
                    "SciPy_eBrP": SciPy_eBrP,
                    "SciPy_hBrP": SciPy_hBrP,
                    "Leti_eBrP": Leti_eBrP,
                    "Leti_hBrP": Leti_hBrP,
                    "alpha_e": alpha_e,
                    "alpha_h": alpha_h}
    
    return dict_results


def check_results_values():
    tolerance = 1e-12
    mesh_line = np.linspace(0.0, 4.5e-4, 1000)
    electric_field = np.array([electric_field_profile.function_electric_field(x) for x in mesh_line])
    
    # If using the VDEVICE.csv file
    # filename = "VDEVICE.csv"
    # x, y = np.loadtxt(filename, delimiter=',', unpack=True, skiprows=1)
    # x = x * 1e-4
    # print(f"Solving for {len(x)} points")
    # nb_points = 2000
    # mesh_line = np.linspace(np.min(x), np.max(x), int(nb_points))
    # electric_field = np.interp(mesh_line, x, y)

    boost_ef = 1.0
    electric_field = boost_ef * electric_field
    x_max_ef = mesh_line[np.argmax(electric_field)]

    # Plot electric field
    fig, ax = plt.subplots(1, 1)
    ax.plot(mesh_line*1e4, electric_field, label='Electric field')
    ax.set_xlabel('Position ($\mu$ m)')
    ax.set_ylabel('Electric field (V $\cdot$ cm$^{-1}$)')
    ax.set_title('Electric field profile')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('electric_field_profile.png', dpi=300)
    fig.savefig('electric_field_profile.pdf')



    Newton_eBreakdownProbability, Newton_hBreakdownProbability = mcintyre_newton.compute_newton_solution(mesh_line, electric_field, tolerance=tolerance)
    total_breakdown_probability = get_total_breakdown_probability(Newton_eBreakdownProbability, Newton_hBreakdownProbability)
    Recursive_eBrP, Recursive_hBrP = mcintyre_recursive.fast_compute_mcintyre_recursive_local(mesh_line, electric_field, tolerance=tolerance, plot=False)
    Leti_eBrP, Leti_hBrP, P_total = mcintyre_leti.Compute_Ptotal(mesh_line, electric_field)

    tolerance = 1e-12
    sol = mcintyre_model.solve_mcintyre(mesh_line, electric_field, 1e-6)
    x_plot = mesh_line
    y1_plot = sol.sol(x_plot)[0]
    y2_plot = sol.sol(x_plot)[1]
    SciPy_eBreakdownProbability = y1_plot
    SciPy_hBreakdownProbability = y2_plot

    fig, ax = plt.subplots(2, 2, figsize=(7,4), sharex=True)
    ax[0, 0].plot(mesh_line*1e4, SciPy_eBreakdownProbability, "-", c="k", label="SciPy Electron")
    ax[0, 0].plot(mesh_line*1e4, Newton_eBreakdownProbability, "--", c="b", label="Newton Electron")
    ax[0, 0].plot(mesh_line*1e4, Recursive_eBrP, ":", c="g", label="Recursive Electron")
    ax[0, 0].plot(mesh_line*1e4, Leti_eBrP, "-.", c="r", label="Leti Electron")
    # Inset zoom on the breakdown probability at the end of the mesh
    axins = ax[0, 0].inset_axes([0.69, 0.1, 0.25, 0.35])
    axins.plot(mesh_line*1e4, Newton_eBreakdownProbability, "-", c="k", label="SciPy Electron")
    axins.plot(mesh_line*1e4, Newton_eBreakdownProbability, "--", c="b", label="Newton Electron")
    axins.plot(mesh_line*1e4, Recursive_eBrP, ":", c="g", label="Recursive Electron")
    axins.plot(mesh_line*1e4, Leti_eBrP, "-.", c="r", label="Leti Electron")
    axins.set_xlim(mesh_line[-1]*1e4-0.15, mesh_line[-1]*1e4)
    axins.set_ylim(Newton_eBreakdownProbability[-1]-0.002, Newton_eBreakdownProbability[-1]+0.0015)
    axins.set_xticklabels('')
    # axins.set_yticklabels('')
    ax[0, 0].indicate_inset_zoom(axins, edgecolor="black", linewidth=1.0, alpha=1.0)

    ax[0, 1].plot(mesh_line*1e4, SciPy_hBreakdownProbability, "-", c="k", label="SciPy Hole")
    ax[0, 1].plot(mesh_line*1e4, Newton_hBreakdownProbability, "--", c="r", label="Newton Hole")
    ax[0, 1].plot(mesh_line*1e4, Recursive_hBrP, ":", c="g", label="Recursive Hole")
    ax[0, 1].plot(mesh_line*1e4, Leti_hBrP, "-.", c="b", label="Leti Hole")
    # Inset zoom on the breakdown probability at the end of the mesh
    axins2 = ax[0, 1].inset_axes([0.2, 0.5, 0.25, 0.35])
    axins2.plot(mesh_line*1e4, Newton_hBreakdownProbability, "-", c="k", label="SciPy Electron")
    axins2.plot(mesh_line*1e4, Newton_hBreakdownProbability, "--", c="b", label="Newton Electron")
    axins2.plot(mesh_line*1e4, Recursive_hBrP, ":", c="g", label="Recursive Electron")
    axins2.plot(mesh_line*1e4, Leti_hBrP, "-.", c="r", label="Leti Electron")
    axins2.set_xlim(mesh_line[0]*1e4, mesh_line[0]*1e4+0.15)
    axins2.set_ylim(Newton_hBreakdownProbability[0]-1.5e-3, Newton_hBreakdownProbability[0]+1.5e-3)
    axins2.set_xticklabels('')
    # axins.set_yticklabels('')
    ax[0, 1].indicate_inset_zoom(axins2, edgecolor="black", linewidth=1.0, alpha=1.0)

    ax[0, 0].set_ylabel("Breakdown Probability")
    ax[0, 1].set_ylabel("Breakdown Probability")
    ax[0, 0].legend()
    ax[0, 1].legend()

    ax[0, 0].set_title("Electron")
    ax[0, 1].set_title("Hole")
    ax[0, 0].set_ylim(-0.05, 0.8)
    ax[0, 1].set_ylim(-0.05, 0.8)

    # ax[0, 0].set_yscale("log")

    # Error
    ax[1, 0].plot(mesh_line*1e4, np.abs(SciPy_eBreakdownProbability - Newton_eBreakdownProbability), "-", c="b", label="Newton")
    ax[1, 0].plot(mesh_line*1e4, np.abs(Recursive_eBrP - SciPy_eBreakdownProbability), "--", c="g", label="Recursive")
    ax[1, 0].plot(mesh_line*1e4, np.abs(SciPy_eBreakdownProbability - Leti_eBrP), ":", c="k", label="Leti")
    
    
    ax[1, 1].plot(mesh_line*1e4, np.abs(Newton_hBreakdownProbability - SciPy_hBreakdownProbability), "-", c="r", label="Newton")
    ax[1, 1].plot(mesh_line*1e4, np.abs(Recursive_hBrP - SciPy_hBreakdownProbability), "--", c="g", label="Recursive")
    ax[1, 1].plot(mesh_line*1e4, np.abs(Leti_hBrP - SciPy_hBreakdownProbability), "-.", c="b", label="Leti")
    ax[1, 0].set_ylim(1e-6, 3e0)
    ax[1, 1].set_ylim(1e-6, 3e0)
    ax[1, 0].set_xlabel("x ($\mu$m)")
    ax[1, 0].set_ylabel("Absolute Error")
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[1, 0].set_yscale("log")
    ax[1, 1].set_yscale("log")

    ax[1,1].set_xlabel("x ($\mu$m)")

    fig.tight_layout()
    plt.savefig("McIntyreCheck.pdf")
    plt.savefig("McIntyreCheck.png", dpi=300)

    plt.show()
    return 0

Models = ["SciPy", "Newton", "Recursive", "Leti"]

def check_eq_1_2():
    dict_results = get_all_values()
    mesh_line = dict_results["mesh_line"]
    alpha_e = dict_results["alpha_e"]
    alpha_h = dict_results["alpha_h"]

    fig, ax = plt.subplots(2, 2, figsize=(7,4), sharex=True, sharey=True)

    idx_plot = 0
    for model in Models:
        eBrP = dict_results[model+ "_eBrP"]
        hBrP = dict_results[model+ "_hBrP"]
        Ptotal = eBrP + hBrP - eBrP * hBrP
        
        dPe_dx = np.gradient(eBrP, mesh_line)
        dPh_dx = np.gradient(hBrP, mesh_line)
        RHS_Pe = (1.0 - eBrP) * alpha_e * Ptotal
        RHS_Ph = -(1.0 - hBrP) * alpha_h * Ptotal
        idx_x_plot = idx_plot // 2
        idx_y_plot = idx_plot % 2
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, dPe_dx, "-", c="b", label="dPe_dx")
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, RHS_Pe, "--", c="k", label="RHS_Pe")

        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, dPh_dx, "-", c="r", label="dPh_dx")
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, RHS_Ph, "--", c="k", label="RHS_Ph")

        ax[idx_x_plot, idx_y_plot].set_title(model)
        ax[idx_x_plot, idx_y_plot].legend()
        idx_plot += 1
    
    ax[0, 0].set_ylabel("dP/dx (1/$\mu$m)")
    ax[1, 0].set_ylabel("dP/dx (1/$\mu$m)")
    ax[1, 0].set_xlabel("x ($\mu$m)")
    ax[1, 1].set_xlabel("x ($\mu$m)")
    fig.suptitle("Check Equation 1 and 2")
    fig.tight_layout()
    plt.savefig("McIntyreCheckEq1_2.pdf")
    plt.savefig("McIntyreCheckEq1_2.png", dpi=300)
    plt.show()


def check_eq_7():
    dict_results = get_all_values()
    mesh_line = dict_results["mesh_line"]
    alpha_e = np.array(dict_results["alpha_e"])
    alpha_h = np.array(dict_results["alpha_h"])

    fig, ax = plt.subplots(2, 2, figsize=(7,4), sharex=True, sharey=True)

    f_int = [np.trapz((alpha_e[:k:] - alpha_h[:k:]), mesh_line[:k:]) for k in range(len(mesh_line))]

    idx_plot = 0
    for model in Models:
        eBrP = dict_results[model+ "_eBrP"]
        hBrP = dict_results[model+ "_hBrP"]
        Ptotal = eBrP + hBrP - eBrP * hBrP
        

        lhs = Ptotal / (1.0 - Ptotal)
        rhs = Ptotal[0] / (1.0 - Ptotal[0]) * np.exp(f_int)
        
        
        idx_x_plot = idx_plot // 2
        idx_y_plot = idx_plot % 2
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, lhs, "-", c="b", label="LHS")
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, rhs, "--", c="k", label="RHS")

        ax[idx_x_plot, idx_y_plot].set_title(model)
        ax[idx_x_plot, idx_y_plot].legend()
        idx_plot += 1
    
    ax[0, 0].set_ylabel("u.a.")
    ax[1, 0].set_ylabel("u.a.")
    ax[1, 0].set_xlabel("x ($\mu$m)")
    ax[1, 1].set_xlabel("x ($\mu$m)")
    fig.suptitle("Check Equation 7")
    fig.tight_layout()
    plt.savefig("McIntyreCheckEq7.pdf")
    plt.savefig("McIntyreCheckEq7.png", dpi=300)
    plt.show()


def check_eq_7a():
    dict_results = get_all_values()
    mesh_line = dict_results["mesh_line"]
    alpha_e = np.array(dict_results["alpha_e"])
    alpha_h = np.array(dict_results["alpha_h"])

    fig, ax = plt.subplots(2, 2, figsize=(7,4))

    f_int = [np.trapz((alpha_e[:k:] - alpha_h[:k:]), mesh_line[:k:]) for k in range(len(mesh_line))]

    idx_plot = 0
    for model in Models:
        eBrP = dict_results[model+ "_eBrP"]
        hBrP = dict_results[model+ "_hBrP"]
        Ptotal = eBrP + hBrP - eBrP * hBrP
        print(hBrP[0])
        

        lhs = eBrP[-1] / (1.0 - eBrP[-1])
        rhs = hBrP[0] / (1.0 - hBrP[0]) * np.exp(f_int[-1])

        idx_x_plot = idx_plot // 2
        idx_y_plot = idx_plot % 2
        ax[idx_x_plot, idx_y_plot].bar([0], [lhs], color="b", label="LHS")
        ax[idx_x_plot, idx_y_plot].bar([1], [rhs], color="k", label="RHS")
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[idx_x_plot, idx_y_plot].set_title(model)
        ax[idx_x_plot, idx_y_plot].set_xticks([0, 1])
        ax[idx_x_plot, idx_y_plot].set_xticklabels(["LHS", "RHS"])
        # ax[idx_x_plot, idx_y_plot].legend()
        idx_plot += 1
        
        
        print(f"Model {model:<10} -> Eq 7: {lhs:.7f} = {rhs:.7f}")

def check_eq_7b():
    dict_results = get_all_values()
    mesh_line = dict_results["mesh_line"]
    alpha_e = np.array(dict_results["alpha_e"])
    alpha_h = np.array(dict_results["alpha_h"])

    fig, ax = plt.subplots(2, 2, figsize=(7,4))

    f_int = np.exp(np.array([np.trapz((alpha_e[:k:] - alpha_h[:k:]), mesh_line[:k:]) for k in range(len(mesh_line))]))

    idx_plot = 0
    for model in Models:
        eBrP = dict_results[model+ "_eBrP"]
        hBrP = dict_results[model+ "_hBrP"]
        Ptotal = eBrP + hBrP - eBrP * hBrP
        

        lhs = Ptotal
        rhs = (hBrP[0] * f_int) / (1 - hBrP[0] + hBrP[0] * f_int)

        idx_x_plot = idx_plot // 2
        idx_y_plot = idx_plot % 2
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, lhs, "-", c="b", label="LHS")
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, rhs, "--", c="k", label="RHS")

        ax[idx_x_plot, idx_y_plot].set_title(model)
        ax[idx_x_plot, idx_y_plot].legend()
        idx_plot += 1
        
    fig.suptitle("Check Equation 7b")
    fig.tight_layout()
    plt.savefig("McIntyreCheckEq7b.pdf")
    plt.savefig("McIntyreCheckEq7b.png", dpi=300)
    plt.show()


def check_eq_8():
    dict_results = get_all_values()
    mesh_line = dict_results["mesh_line"]
    alpha_e = np.array(dict_results["alpha_e"])
    alpha_h = np.array(dict_results["alpha_h"])

    fig, ax = plt.subplots(2, 2, figsize=(7,4))

    f_int = np.exp(np.array([np.trapz((alpha_e[:k:] - alpha_h[:k:]), mesh_line[:k:]) for k in range(len(mesh_line))]))

    idx_plot = 0
    for model in Models:
        eBrP = dict_results[model+ "_eBrP"]
        hBrP = dict_results[model+ "_hBrP"]
        Ptotal = eBrP + hBrP - eBrP * hBrP
        

        lhs = 1.0 - hBrP
        integral = np.array([np.trapz((alpha_h[:k:] * Ptotal[:k:]), mesh_line[:k:]) for k in range(len(mesh_line))])
        rhs = (1.0 - hBrP[0]) * np.exp(integral)

        idx_x_plot = idx_plot // 2
        idx_y_plot = idx_plot % 2
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, lhs, "-", c="b", label="LHS")
        ax[idx_x_plot, idx_y_plot].plot(mesh_line*1e4, rhs, "--", c="k", label="RHS")

        ax[idx_x_plot, idx_y_plot].set_title(model)
        ax[idx_x_plot, idx_y_plot].legend()
        idx_plot += 1
        
        
    fig.suptitle("Check Equation 8")
    fig.tight_layout()
    plt.savefig("McIntyreCheckEq8.pdf")
    plt.savefig("McIntyreCheckEq8.png", dpi=300)
    plt.show()

def check_eq_9():
    dict_results = get_all_values()
    mesh_line = dict_results["mesh_line"]
    alpha_e = np.array(dict_results["alpha_e"])
    alpha_h = np.array(dict_results["alpha_h"])

    fig, ax = plt.subplots(2, 2, figsize=(7,4))

    f_int = np.exp(np.array([np.trapz((alpha_e[:k:] - alpha_h[:k:]), mesh_line[:k:]) for k in range(len(mesh_line))]))

    idx_plot = 0
    for model in Models:
        eBrP = dict_results[model+ "_eBrP"]
        hBrP = dict_results[model+ "_hBrP"]
        Ptotal = eBrP + hBrP - eBrP * hBrP
        
        print(f"Model: {model:<15}     hBrP[0]   = {hBrP[0]}")
        print(f"Model: {model:<15}     Ptotal[0] = {Ptotal[0]}")
        lhs = -np.log(1.0 - hBrP[0])
        rhs1 = np.trapz((alpha_h[::] * Ptotal[::]), mesh_line[::])
        rhs2 = np.trapz((hBrP[0] * alpha_h * f_int) / (hBrP[0]*f_int + 1 - hBrP[0]), mesh_line[::])

        idx_x_plot = idx_plot // 2
        idx_y_plot = idx_plot % 2
        ax[idx_x_plot, idx_y_plot].bar([0], [lhs], color="b", label="LHS")
        ax[idx_x_plot, idx_y_plot].bar([1], [rhs1], color="k", label="RHS1")
        ax[idx_x_plot, idx_y_plot].bar([2], [rhs2], color="r", label="RHS2")
       # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[idx_x_plot, idx_y_plot].set_title(model)
        ax[idx_x_plot, idx_y_plot].set_xticks([0, 1, 2])
        ax[idx_x_plot, idx_y_plot].set_xticklabels(["LHS", "RHS1", "RHS2"])

        idx_plot += 1
        
    fig.suptitle("Check Equation 9")
    fig.tight_layout()
    plt.savefig("McIntyreCheckEq9.pdf")
    plt.savefig("McIntyreCheckEq9.png", dpi=300)
    plt.show()





if __name__ == "__main__":
    check_results_values()
    check_eq_1_2()
    check_eq_7()    
    check_eq_7a()
    check_eq_7b()
    check_eq_8()
    check_eq_9()
    