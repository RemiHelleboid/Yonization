include("electric_fields.jl")
include("impact_ionization.jl")


module McIntyres 

using Plots
using LaTeXStrings
using LinearAlgebra
using NumericalIntegration
using Distributed
using SharedArrays

using  ..ImpactIonizations
using ..ElectricFields

function mcintyre_initial_guess(x_line::Vector{Float64}, electric_field_profile::Vector{Float64})
    line_size = length(x_line)
    maxval, maxindx = findmax(electric_field_profile);
    guess_e_brp::Vector{Float64} = zeros(line_size)
    guess_h_brp::Vector{Float64} = zeros(line_size)
    for index in eachindex(x_line)
        if x_line[index] >= x_line[maxindx]
            guess_e_brp[index] = 0.7
        else
            guess_h_brp[index] = 0.2
        end
    end
    return guess_e_brp, guess_h_brp
end

function compute_recursive_mcintyre(x_line::Vector{Float64}, electric_field_profile::Vector{Float64}, tolerance::Float64, boost::Float64=1.0)
    line_size = length(x_line)
    line_αₑ::Vector{Float64} = map(αₑ, boost .* electric_field_profile)
    line_αₕ::Vector{Float64} = map(αₕ, boost .* electric_field_profile)
    line_e_brp::Vector{Float64}, line_h_brp::Vector{Float64} = mcintyre_initial_guess(x_line, electric_field_profile)
    line_e_brp_new::Vector{Float64} = zeros(line_size)
    line_h_brp_new::Vector{Float64} = zeros(line_size)

    max_epoch = 100
    epoch = 0
    difference = 1.0e6
    p1 = plot()
    while difference >= tolerance && epoch <= max_epoch
        total_brp_line = line_e_brp .+ line_h_brp - line_e_brp .* line_h_brp
        for index in eachindex(x_line[1:line_size])
            new_e_brp = line_e_brp[1] + integrate(x_line[1:index],line_αₑ[1:index] .* (ones(index) - line_e_brp[1:index]) .* total_brp_line[1:index], TrapezoidalEvenFast())
            new_h_brp = line_h_brp[line_size] + integrate(x_line[index:line_size], line_αₕ[index:line_size] .* (ones(line_size - index + 1) - line_h_brp[index:line_size]) .* total_brp_line[index:line_size], TrapezoidalEvenFast())
            line_e_brp_new[index] = new_e_brp
            line_h_brp_new[index] = new_h_brp
        end
        difference = norm(line_e_brp_new - line_e_brp)
        line_e_brp = copy(line_e_brp_new)
        line_h_brp = copy(line_h_brp_new)
        println("Epoch : $epoch")
        println("Difference : $difference")
        epoch = epoch + 1
    end
    plot!(x_line, line_e_brp, ylims=(0, 1.0), label="Electron")
    plot!(x_line, line_h_brp, ylims=(0, 1.0), label="Hole")
    savefig("McIntyreRecursive.svg")
    savefig("McIntyreRecursive.png")
end

function fast_compute_recursive_mcintyre(x_line::Vector{Float64}, electric_field_profile::Vector{Float64}, tolerance::Float64, boost::Float64=1.0)
    line_size = length(x_line)
    line_αₑ::Vector{Float64} = map(αₑ, boost .* electric_field_profile)
    line_αₕ::Vector{Float64} = map(αₕ, boost .* electric_field_profile)
    line_e_brp::Vector{Float64}, line_h_brp::Vector{Float64} = mcintyre_initial_guess(x_line, electric_field_profile)
    line_e_brp_new::Vector{Float64} = zeros(line_size)
    line_h_brp_new::Vector{Float64} = zeros(line_size)

    dx = x_line[2] - x_line[1]

    max_epoch = 2000
    epoch = 0
    difference = 1.0e6
    p1 = plot()
    while difference >= tolerance && epoch <= max_epoch
        total_brp_line = line_e_brp .+ line_h_brp - line_e_brp .* line_h_brp
        sum_integral_electron = 0.0
        sum_integral_hole = 0.0
        integral_hole_line::Vector{Float64} = zeros(line_size)
        for index in eachindex(x_line)
            sum_integral_hole += dx * (line_αₕ[index] * (1.0 - line_h_brp[index]) * total_brp_line[index])
            integral_hole_line[index] = sum_integral_hole
        end
        integral_hole_line = integral_hole_line[line_size] * ones(line_size) - integral_hole_line

        for index in eachindex(x_line[1:line_size])
            sum_integral_electron += dx * line_αₑ[index] * (1.0 - line_e_brp[index]) * total_brp_line[index]
            line_e_brp_new[index] = sum_integral_electron
            line_h_brp_new[index] = line_h_brp[line_size] + integral_hole_line[index]
        end
        difference = norm(line_e_brp_new - line_e_brp)
        line_e_brp = copy(line_e_brp_new)
        line_h_brp = copy(line_h_brp_new)
        println("Epoch : $epoch")
        println("Difference : $difference")
        epoch = epoch + 1
    end
    # plot!(x_line, line_e_brp, ylims=(0, 1.0), label="Electron")
    # plot!(x_line, line_h_brp, ylims=(0, 1.0), label="Hole")
    # gui()
    # savefig("McIntyreRecursiveFast.svg")
    # savefig("McIntyreRecursiveFast.png")
end

x_min = 0.0
x_max = 3.0e-4
number_points = 10000000
boost = 0.90
x_line, electric_filed_line = ElectricFields.electric_field_profile(x_min, x_max, number_points)
# gr()
fast_compute_recursive_mcintyre(x_line, electric_filed_line, 1.0e-9, boost)
# gui()

end
