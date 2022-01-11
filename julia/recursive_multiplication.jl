include("electric_fields.jl")
include("impact_ionization.jl")

module RecursiveMultiplications

using Plots
using LaTeXStrings
using LinearAlgebra
using NumericalIntegration

using  ..ImpactIonizations
using ..ElectricFields

function compute_recursive_multiplication(x_line::Vector{Float64}, electric_field_profile::Vector{Float64}, tolerance::Float64, boost::Float64=1.0)
    line_size = length(x_line)
    line_αₑ::Vector{Float64} = map(αₑ, boost .* electric_field_profile)
    line_αₕ::Vector{Float64} = map(αₕ, boost .* electric_field_profile)
    line_multiplication::Vector{Float64} = zeros(line_size)
    line_multiplication_new::Vector{Float64} = zeros(line_size)

    max_epoch = 1000
    epoch = 0
    difference = 1.0e6
    p1 = plot()
    while difference >= tolerance && epoch <= max_epoch
        for index in eachindex(x_line[1:line_size])

            additional_electron = integrate(x_line[1:index], line_multiplication[1:index] .* line_αₑ[1:index], TrapezoidalEvenFast()) 
            additional_hole = integrate(x_line[index:line_size], line_multiplication[index:line_size] .* line_αₕ[index:line_size], TrapezoidalEvenFast())
            line_multiplication_new[index] = 1.0 + additional_electron + additional_hole
            if index > (line_size - 2)
                println(index)
            end
        end
        difference = norm(line_multiplication_new - line_multiplication)
        line_multiplication = copy(line_multiplication_new)
        println("Epoch : $epoch")
        println("Difference : $difference")
        epoch = epoch + 1
    end
    plot!(x_line, line_multiplication, label = "Epoch $epoch", ylims=(0, Inf))
    savefig("MultiplicationRecursiveEpochsBest2.svg")
    savefig("MultiplicationRecursiveEpochsBest2.png")
    
end

x_min = 0.0
x_max = 3.0e-4
number_points = 1000
boost = 0.95
x_line, electric_filed_line = ElectricFields.electric_field_profile(x_min, x_max, number_points)
gr()
compute_recursive_multiplication(x_line, electric_filed_line, 1.0e-4, boost)
gui()

end
