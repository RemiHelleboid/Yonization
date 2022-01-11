# Electric field profile functions for tests.


function electric_field_raw_at_position(x_position::Float64, boost_factor::Float64=1.2)::Float64
    low_field::Float64 = 1.0e4
    electric_field::Float64 = 0.0
    if x_position <= 1.0e-4
        electric_field = low_field
    elseif x_position > 1.0e-4 && x_position <= 1.36e-4
        electric_field = 1624982.38 * x_position * 1e4 - 1614982.38
    elseif  x_position > 1.36e-4 && x_position <= 1.49e-4
        electric_field = -4656341.3 * x_position * 1e4 + 6946969.7
    else
        electric_field = low_field
    end
    return boost_factor * electric_field
end


function electric_field_profile(x_min::Float64, x_max::Float64, number_points::Int64)::Tuple{Vector{Float64},Vector{Float64}}
    x_line = collect(LinRange(x_min, x_max, number_points))
    electric_field_profile::Vector{Float64} = map(electric_field_raw_at_position, x_line)
    return x_line, electric_field_profile
end
