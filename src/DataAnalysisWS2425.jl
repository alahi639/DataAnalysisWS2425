module DataAnalysisWS2425
using QuadGK, Random, Statistics, Optim, Plots, StatsBase
export greet, gaussian_scaled, polynomial_scaled, breit_wigner_scaled, voigt_scaled, sample_rejection, sample_inversion, extended_nll, fit_enll, plot_data_fit_with_pulls

greet() = print("Hello World!")

"""
    gaussian_scaled(x; μ, σ, a=1.0)

Computes the value of a Gaussian function with flexible normalization at `x`, given the mean `μ`, standard deviation `σ`, and scaling factor `a`.

The form of the Gaussian is:

    a * exp(-((x - μ)^2) / (2 * σ^2))

# Example
```julia
julia> y = gaussian_scaled(2.0; μ=0.0, σ=1.0, a=3.0)
```
"""
function gaussian_scaled(x; μ, σ, a)
	return a * exp(-(x-μ)^2 / (2*(σ^2)))
end

"""
    polynomial_scaled(x; coeffs)

Evaluates a polynomial at `x`, given the coefficients in `coeffs`.
The `coeffs` is an iterable collection of coefficients, where the first element corresponds to the lowest degree term.

The polynomial has the form:

    coeffs[1] * x^0 + coeffs[1] * x^(2) + ... + coeffs[n] * x^(n-1)

where `n-1` is the degree of the polynomial, determined by the length of `coeffs`.

# Example
```julia
julia> y = polynomial_scaled(2.0; coeffs=[1.0, -3.0, 2.0])
```
"""
function polynomial_scaled(x; coeffs)
	return sum([coeffs[n]*x^(n-1) for n in 1:length(coeffs)])
end

"""
    breit_wigner_scaled(x; M, Γ, a=1.0)

Computes the value of a Breit-Wigner function with flexible normalization at `x`, given the mass `m`, width `Γ`, and scaling factor `a`.

The form of the Breit-Wigner is:

    a / |m^2 - x^2 - imΓ|^2

# Example
```julia
julia> y = breit_wigner_scaled(2.0; m=1.0, Γ=0.5, a=2.0)
```
"""
function breit_wigner_scaled(x; M, Γ, a)
	return a / ((x^2 - M^2)^2 + (M^2 * Γ^2))
end
	
"""
    voigt_scaled(x; M=0.0, Γ=1.0, σ=1.0, a=1.0)

Computes the value of a Voigt profile with flexible normalization at `x`, given the peak position `m`, Breit-Wigner width `Γ`, Gaussian width `σ`, and scaling factor `a`.

The Voigt profile is a convolution of a non-relativistic Breit-Wigner function and a Gaussian, commonly used to describe spectral lineshapes.

# Example
```julia
julia> y = voigt_scaled(2.0; M=1.3, Γ=0.15, σ=0.3, a=3.0)
```
"""
function voigt_scaled(x; M, Γ, σ, a)
	integrand(Ep; x, M, Γ, σ, a) = breit_wigner_scaled(x-Ep; M, Γ, a)* gaussian_scaled(Ep; μ=0, σ, a=1)*(1/(σ*sqrt(2*pi))) #* exp(-(Ep^2) / (2*(σ^2)))
	result = quadgk(Ep -> integrand(Ep; x, M, Γ, σ, a), -Inf, Inf)[1]
	return result
end

"""
    sample_rejection(f, n, support; nbins=1000)

Generates `n` samples using the rejection sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional keyword argument. The number of equidistant points in `support` used to search for the maximum of `f`

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
```julia
julia> data = sample_rejection(exp, 10, (0, 4))
```
"""
function sample_rejection(f, n, support; nbins=1000)
    samples = Float64[]
    
    # Find the maximum of the function over the support for scaling
    x_vals = range(support[1], support[2], length=nbins)
    f_max = maximum(f.(x_vals))
    
    for _ in 1:n
        while true
            # Generate a random x within the support range
            x = support[1] + (support[2] - support[1]) * rand()
            # Generate a random y within the range [0, f_max]
            y = rand() * f_max
            
            # Accept or reject the sample
            if y <= f(x)
                push!(samples, x)
                break
            end
        end
    end
    return samples
end


"""
    sample_inversion(f, n, support; nbins=1000)

Generates `n` samples using the inversion sampling method for a given function `f` over a specified `support` range.

# Arguments
- `f::Function`: The function to sample from.
- `n::Int`: The number of samples to generate.
- `support::Tuple{T, T}`: A tuple specifying the range `(a, b)` where the function `f` will be sampled.
- `nbins::Int`: Optional keyword argument. The number of equidistant points in `support` for which the c.d.f. is pre-computed.

# Returns
- An array of `n` samples generated from the distribution defined by `f`.

# Example
```julia
julia> data = sample_inversion(exp, 10, (0, 4))
```
"""

function sample_inversion(f, n, support; nbins=1000)
    # Define the PDF by normalizing `f` over the given `support`
    normalization_factor = quadgk(f, support[1], support[2])[1]
    mypdf(x) = f(x) / normalization_factor
    
    # Compute the CDF by integrating the PDF
    mycdf(x) = quadgk(mypdf, support[1], x)[1]
    
    # Precompute the CDF over a grid for efficient sampling
    precomputed_cdf = let
        grid = range(support[1], support[2], length=nbins)
        cdf_values = mycdf.(grid)
        (; grid, cdf_values)
    end
    
    # Function to generate a sample using the inverse CDF method
    function generate_with_inv_cdf(two_rand, precomputed_cdf) 
        grid = precomputed_cdf.grid
        cdf_values = precomputed_cdf.cdf_values
        r1, r2 = two_rand  # Random values between 0 and 1
        
        # Find the index of the bin where the random value `r1` falls
        binind = findfirst(cdf_values .> r1)
        if isnothing(binind) || binind == 1
            return grid[1]  # Edge case: return the first grid point
        elseif binind == length(grid)
            return grid[end]  # Edge case: return the last grid point
        end
        
        # Interpolate within the bin to find the sample point
        x_left, x_right = grid[binind-1], grid[binind]
        cdf_left, cdf_right = cdf_values[binind-1], cdf_values[binind]
        
        # Linear interpolation within the bin
        x = x_left + (r1 - cdf_left) / (cdf_right - cdf_left) * (x_right - x_left)
        return x
    end
    
    # Generate `n` samples using the inversion method
    data = let
        random_input = rand(2, n)  # Generate random pairs (r1, r2) for sampling
        pairs_of_numbers = eachslice(random_input, dims=2)
        generate_with_inv_cdf.(pairs_of_numbers, Ref(precomputed_cdf))
    end
    
    return data  # Return the generated samples
end 

"""
    extended_nll(model, parameters, data; support = extrema(data), normalization_call = _quadgk_call)

Calculate the extended negative log likelihood (ENLL) for a given model and dataset.

# Arguments
- `model`: A function that represents the model. It should take two arguments: observable and parameters.
- `parameters`: The parameters for the model.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated. Defaults to the range of the data.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `extended_nll_value`: The extended negative log likelihood value for the given model and data.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
parameters = [1.0, 0.1]
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

enll_value = extended_nll(model, parameters, data; support=support)
```
"""
function extended_nll(model, parameters, data; support = extrema(data), normalization_call = quadgk)
    # Calculate the negative log likelihood
    minus_sum_log = -sum(data) do x
        value = model(x, parameters)
        value > 0 ? log(value) : -1e10  # Penalize non-positive values
    end

    # Calculate the normalization constant
    normalization, _ = normalization_call(support...) do x
        model(x, parameters)
    end

    # Calculate the extended negative log likelihood
    enll = minus_sum_log + normalization
    return enll
end

"""
    fit_enll(model, init_pars, data; support = extrema(data), alg=BFGS, normalization_call = _quadgk_call)

Fit the model parameters using the extended negative log likelihood (ENLL) method.

# Arguments
- `model`: A function that represents the model to be fitted. It should take two arguments: observable and parameters.
- `init_pars`: Initial parameters for the model.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated.
- `alg`: (Optional) Optimization algorithm to be used. Default is `BFGS`.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `result`: The optimization result that minimizes the extended negative log likelihood.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
init_pars = [1.0, 0.1]
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

fit_result = fit_enll(model, init_pars, data; support=support)
```
"""
function fit_enll(model, init_pars, data; support = extrema(data), alg = BFGS(), normalization_call = quadgk)
    # Define objective function for optimizer
    objective(p) = extended_nll(model, typeof(init_pars)(p), data; support = support, normalization_call = normalization_call)
    
    # Run the optimization
    result = optimize(objective, collect(init_pars), alg)
    return result
end

"""
    plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlbl="", ylbl="")

Plot a histogram of data with a fit using model overlaid and a pull distribution.

# Arguments
- `data`: A collection of data points.
- `model`: A function that represents the model to be fitted. It should take two arguments: data points and parameters.
- `binning`: The bin edges for the histogram.
- `best_fit_pars`: The best-fit parameters for the model.
- `xlbl`: (Optional) Label for the x-axis. Default is an empty string.
- `ylbl`: (Optional) Label for the y-axis. Default is an empty string.

# Example
```julia
data = [0.1, 0.2, 0.3, 0.4, 0.5]
model(x, p) = p[1] * exp(-p[2] * x)
binning = 0:0.1:1.0
best_fit_pars = [1.0, 0.1]
plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlbl="X-axis", ylbl="Y-axis")
```
"""
function plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlbl="", ylbl="")
    # Create histogram of data
    hist_plot = histogram(data, bins=binning, normalize=true, label="Data", xlabel=xlbl, ylabel=ylbl)

    # Overlay model fit
    x_vals = range(binning[1], stop=binning[end], length=100)
    y_vals = [model(x, best_fit_pars) for x in x_vals]
    plot!(x_vals, y_vals, label="Model", color="red")

    # Compute and plot pull distribution
    model_vals = [model(x, best_fit_pars) for x in binning]
    data_counts, _ = fit(Histogram, data, binning).weights
    pulls = (data_counts .- model_vals) ./ sqrt.(model_vals .+ 1e-6)  # Avoid divide by zero

    pull_plot = bar(binning[1:end-1], pulls, label="Pulls", xlabel=xlbl, ylabel="Pulls", color="blue")

    # Combine into a single layout with subplots
    plot(hist_plot, pull_plot, layout=(2, 1), size=(800, 600))
end

end # module DataAnalysisWS2425
