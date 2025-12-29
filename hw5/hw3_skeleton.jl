using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters
using StatsBase

@with_kw mutable struct POMDPscenario
    F::Matrix{Float64} # 2x2
    Σw::Matrix{Float64} # 2x2
    Σv::Matrix{Float64} # 2x2
    rng::MersenneTwister
    beacons::Matrix{Float64} # nx2 (n is number of beacons)
    d::Float64
    rmin::Float64
end


function SampleMotionModel(𝒫::POMDPscenario, x::Vector{Float64}, a::Vector{Float64})
    # add your code here...
end

# function PropagateBelief(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64})::FullNormal
#     # add your code here...
#     return MvNormal(μp, Σp)
# end

function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Vector{Float64})::Union{NamedTuple, Nothing}
    distances = # add your code here...
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            obs = # add your code here...
            return (obs=obs, index=index)
        end
    end
    return nothing
end

function GenerateRangedObservationFromBeacons(𝒫::POMDPscenario, x::Vector{Float64})::Union{NamedTuple, Nothing}
    distances = # add your code here...
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            obs = # add your code here...
            return (obs=z, index=index) 
        end
    end
    return nothing
end

function PosteriorBeliefBeacons(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64}, z::NamedTuple)::FullNormal
    # add your code here...
    return MvNormal(μb′, Σb′)
end

function PosteriorBeliefRangedBeacons(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64}, z::NamedTuple)::FullNormal
    # add your code here...
    return MvNormal(μb′, Σb′)
end

function TerminalCost(b::FullNormal)
    # add your code here...
end


function main()
    # definition of the random number generator with seed 
    ID = # add you ID
    rng = MersenneTwister(ID)

    # parameters
    F = [1.0 0.0;
         0.0 1.0]
    Σw = 0.1^2 * [1.0 0.0;
                  0.0 1.0]
    Σv =  0.1^2 * [1.0 0.0;
                  0.0 1.0]
    d = 1.0
    rmin = 0.1

    # set beacons locations
    beacons = # add your code here...

    # initialize prior belief
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0;
          0.0 1.0]
    b0 = MvNormal(μ0, Σ0)

    xgt0 = [-0.5, -0.2] # ground truth initial location
    a_i = [1.0, 1.0] # action of the action sequence
    T = 10 # time steps

    # initialize the scenario
    𝒫 = POMDPscenario(F=F, Σw=Σw, Σv=Σv, rng = rng, beacons=beacons, d=d, rmin=rmin)

    # generate motion trajectory
    τ = [xgt0] # trajectory
    for i in 0:T-1
        push!(τ, SampleMotionModel(𝒫, τ[end], a_i))
    end

    @show τ

    # generate observation trajectory
    τ_obs_beacons_ranged = []
    for i in 1:T
        push!(τ_obs_beacons_ranged, GenerateRangedObservationFromBeacons(𝒫, τ[i+1]))
    end

    @show τ_obs_beacons_ranged

    pl_gt =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle, markersize=6, aspect_ratio=:equal, size=(700,700), legend=:topleft)
    # Use covellipse to draw circles of radius d around beacons
    beacon_range_cov = d^2 * I(2)
    for i in 1:size(beacons, 1)
        covellipse!(pl_gt, beacons[i,:], beacon_range_cov, n_std=1,
                           lw=0.5, c=:blue, linecolor=:black, fillalpha=0.2, label=false)
    end
    scatter!(pl_gt, [x[1] for x in τ],
                    [x[2] for x in τ],
                    label="gt", markershape=:diamond, markersize=6)
    scatter!(pl_gt, [beacons[obs.index, 1] + obs.obs[1] for obs in τ_obs_beacons_ranged if !isnothing(obs)],
                    [beacons[obs.index, 2] + obs.obs[2] for obs in τ_obs_beacons_ranged if !isnothing(obs)],
                    label="obs", markershape=:star5, markersize=6, color=:orange)
    savefig(pl_gt,"ground_truth_$ID.png")

    # add your code here...
end

main()