### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 80161b0b-4b97-4b47-bb3d-0d19c5fcc479
using Test
using DataAnalysisWS2425
using DataAnalysisWS2425.QuadGK
using DataAnalysisWS2425.Random

# test the implementation of gaussian_scaled
@testset "gaussian" begin
    @test gaussian_scaled(1.1; μ = 0.4, σ = 0.7, a = 1.0) ≈ 0.6065306597126333
    @test gaussian_scaled(2268.1; μ = 2286.4, σ = 7.0, a = 1.0) ≈ 0.03280268530267093
end

# test the implementation of polynomial_scaled
@testset "polynomials" begin
    @test polynomial_scaled(1.3; coeffs = (1.1, 0.5)) ≈ 1.75
    @test polynomial_scaled(1.3; coeffs = (0.0, -0.5, 0.3, 1.7)) ≈ 3.5919
end

# test the implementation of breit_wigner_scaled
@testset "Relativistic Breit-Wigner" begin
    @test breit_wigner_scaled(1530.0; M = 1532.0, Γ = 9.0, a = 1532.0^2) ≈ 0.010311498077081241
    @test breit_wigner_scaled(11.3; M = 12.0, Γ = 0.3, a = 144.0) ≈ 0.5161732492496677
end

# test the implementation of voigt_scaled
@testset "Voigt profile" begin
    @test voigt_scaled(1.53; M = 1.532, Γ = 9.0, σ = 6.0, a = 1) ≈ 0.002641789064996706
    @test voigt_scaled(4.2; M = 4.3, Γ = 0.1, σ = 0.05, a = 3.0) ≈ 4.674318379761549
end

# # code for visual instpection of convolution method
# using Plots
# let
#     plot()
#     pars = (; M = 1.532, Γ = 0.03, a = 2)
#     plot!(x->voigt_scaled(x; pars..., σ = 0.05), 1.1, 2.5)
#     plot!(x->breit_wigner_scaled(x; pars...), 1.1, 2.5)
# end

# code for visual instpection of simpling methods
# using Plots
# let
#     f(x) = exp(-x^4)
#     data = sample_inversion(f, 100_000, (-2.0, 2.0); nbins=20)
#     bins = range(-2, 2, 400)
#     stephist(data; bins)
# end

@testset "Rejection sampling" begin
    Random.seed!(1234)
    #
    g(x) = gaussian_scaled(x; μ = 2286.4, σ = 7.0, a = 1.0)
    data_g = sample_rejection(g, 3, (2240.0, 2330.0))
    @test data_g isa Vector
    @test length(data_g) == 3
    #
    v(x) = voigt_scaled(x; M = 1532.0, Γ = 9.0, σ = 6.0, a = 1532)
    data_v = sample_rejection(v, 2, (1500.0, 1560.0))
    @test data_v isa Vector
    @test length(data_v) == 2
end

@testset "Inversion sampling" begin
    data_g = sample_inversion(4, (-4.0, 4.0)) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    @test data_g isa Vector
    @test length(data_g) == 4
end

@testset "Simple fitting" begin
    init_pars = (; μ = 0.3, σ = 1.0, a = 1.0)
    support = (-4.0, 4.0)
    data = sample_inversion(400, support) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    model(x, pars) = gaussian_scaled(x; pars.μ, pars.σ, pars.a)
    ext_unbinned_fit = fit_enll(model, init_pars, data; support)
    best_pars_extnll = typeof(init_pars)(ext_unbinned_fit.minimizer)
    @test ext_unbinned_fit.ls_success
    @test 0.36 < best_pars_extnll.μ < 0.44
end

@testset "Plot Data Fit with Pulls" begin
    # Generate synthetic data
    init_pars = (; μ = 0.3, σ = 1.0, a = 1.0)
    support = (-4.0, 4.0)
    data = sample_inversion(400, support) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    model(x, pars) = gaussian_scaled(x; pars.μ, pars.σ, pars.a)

    # Fit model to data
    ext_unbinned_fit = fit_enll(model, init_pars, data; support)
    best_pars_extnll = typeof(init_pars)(ext_unbinned_fit.minimizer)
    @test ext_unbinned_fit.ls_success  # Test that fitting succeeded

    # Check that fitted parameters are within expected range (e.g., μ)
    @test 0.36 < best_pars_extnll.μ < 0.44

    # Define binning and plot labels
    binning = -4.0:0.5:4.0
    xlbl = "Observable X"
    ylbl = "Counts"

    # Generate the plot
    plt = plot_data_fit_with_pulls(data, model, binning, best_pars_extnll; xlbl=xlbl, ylbl=ylbl)
    
    # Basic test to ensure plot object is created successfully
    @test plt isa Plot

    # Optional: save plot to a temporary file for manual inspection
    # savefig(plt, "temp_plot.png")  # Uncomment if you want to check the plot manually

    # Test that the plot contains the expected series (data histogram, model, and pulls)
    @test length(plt.series_list) >= 3  # Check there are at least three series (data, model, pulls)
end

# ╔═╡ Cell order:
# ╠═80161b0b-4b97-4b47-bb3d-0d19c5fcc479
