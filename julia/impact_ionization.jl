include("electric_fields.jl")
module ImpactIonizations
export αₑ, αₕ 

using ..ElectricFields

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

using Plots
using LaTeXStrings


function plot_ionization_coefficients(x_min::Float64, x_max::Float64, number_points::Int64)
    x_line, electric_filed_line = ElectricFields.electric_field_profile(x_min, x_max, number_points)
    line_αₑ = map(αₑ, electric_filed_line)
    line_αₕ = map(αₕ, electric_filed_line)
    # plotlyjs()
    plot_ef = plot(x_line, electric_filed_line, show = true, 
                    fontfamily = "computer modern",
                     title= "Ionization coefficients Profile",
                     label= "Electric Field",
                     xlabel = L"\textrm{Depth\;\;} (u.a.)",
                     ylabel = L"\textrm{Ionization\; coefficient\;\;} (cm^{-1})",
                     xformatter = :scientific,
                     dpi=600)
    plot!(x_line, line_αₑ, show=true, label= "Electron")
    plot!(x_line, line_αₕ, show=true, label= "Hole")
    savefig(plot_ef, "plot_ionization_coef_profile.png")
    savefig(plot_ef, "plot_ionization_coef_profile.svg")
    savefig(plot_ef, "plot_ionization_coef_profile.pdf")
end

end # End of ImpactIonizations module

using .ImpactIonizations

 if abspath(PROGRAM_FILE) == @__FILE__
    ImpactIonizations.plot_ionization_coefficients(0.0, 3e-4, 1000)
 end