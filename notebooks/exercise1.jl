### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ c40a2614-8620-11ef-0a0f-d77c4d42f1cc
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(joinpath(@__DIR__, ".."))
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    # You do not need all packages that you have added to your project earlier
   # You can however add more as we move along. Just re-execute this cell and they will be there.
    using DataAnalysisWS2425, Random, Plots
end

# ╔═╡ Cell order:
# ╠═c40a2614-8620-11ef-0a0f-d77c4d42f1cc
