include("electric_fields.jl")
module ImpactIonizations
export αₑ, αₕ 

using ..ElectricFields


function compute_dead_space(electric_field, impact_ionization_energy_threshold) 
    return impact_ionization_energy_threshold / electric_field
end

function local_to_dead_space_model(local_coefficient, dead_space)
    dead_space_coefficient = 1.0 / ((1.0 / local_coefficient) - 2.0 * dead_space)
    return dead_space_coefficient
end

function αₑ(electric_field::Float64)
    a_e::Float64 = 7.03e5
    b_e::Float64 = 1.231e6
    imapact_ionization_e::Float64 = a_e * exp(-b_e / electric_field)
    return imapact_ionization_e
end

function αₕ(electric_field::Float64)
    E_threshold::Float64 = 0.0
    E_0::Float64 = 4.0e5
    if (electric_field <= E_threshold)
        return 0.0
    elseif (electric_field <= E_0)
        a_h::Float64 = 1.582e6
        b_h::Float64 = 2.036e6
        imapact_ionization_h::Float64 = a_h * exp(-b_h / electric_field)
        return imapact_ionization_h
    else
        a_h = 6.71e5
        b_h = 1.693e6
        imapact_ionization_h = a_h * exp(-b_h / electric_field)
    return imapact_ionization_h
    end
end






# using Plots
using LaTeXStrings
using PyPlot


function plot_ionization_coefficients(x_min::Float64, x_max::Float64, number_points::Int64)
    x_line, electric_filed_line = ElectricFields.electric_field_profile(x_min, x_max, number_points)
    line_αₑ = map(αₑ, electric_filed_line)
    line_αₕ = map(αₕ, electric_filed_line)
    # plotlyjs()
    # PyPlot()
    # Plots.PlotlyJSBackend()
    plt = PyPlot.plot(x_line, electric_filed_line,  
                    # fontfamily = "computer modern",
                    #  title= "Ionization coefficients Profile",
                     label= "Electric Field",
                    #  xlabel = L"\textrm{Depth\;\;} (u.a.)",
                    #  ylabel = L"\textrm{Ionization\; coefficient\;\;} (cm^{-1})",
                    #  xformatter = :scientific,
                     )
    plot(x_line, line_αₑ, label= "Electron")
    plot(x_line, line_αₕ, label= "Hole")
    legend()
    tight_layout()
    savefig("figures/plot_ionization_coef_profile.png")
    savefig("figures/plot_ionization_coef_profile.svg")
    savefig("figures/plot_ionization_coef_profile.pdf")
end

function plot_dead_space()
    number_points = 1000
    list_electric_field = collect(LinRange(1.0e3, 1.0e6, number_points))
    list_dead_space = [compute_dead_space(ef, 1.12) for ef in list_electric_field]    
end

end # End of ImpactIonizations module

using .ImpactIonizations

if abspath(PROGRAM_FILE) == @__FILE__
ImpactIonizations.plot_ionization_coefficients(0.0, 3e-4, 1000)
end