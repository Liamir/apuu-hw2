module GaussianFilter

export PropagateBelief, PosteriorBeliefRangedBeacons

using apuu.RobotEnv2D: POMDPscenario
using Distributions
using Random
using Parameters
using LinearAlgebra
using Plots
using StatsPlots


function PropagateBelief(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64})::FullNormal
    F = 𝒫.F
    μp = F * b.μ + a
    Σp = F * b.Σ * F' + 𝒫.Σw
    return MvNormal(μp, Σp)
end

# function PosteriorBeliefBeacons(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64}, z::NamedTuple)::FullNormal
#     # add your code here...
#     return MvNormal(μb′, Σb′)
# end

"""
    PosteriorBeliefRangedBeacons(𝒫, b, a, z)

Compute the posterior belief given a prior belief, action, and ranged beacon observation.

# Arguments:
- `𝒫::POMDPscenario`: The POMDP scenario
- `b::FullNormal`: The prior belief (Gaussian)
- `a::Vector{Float64}`: The action taken
- `z::NamedTuple`: The observation, containing `obs` and `beacon_index`
"""
function PosteriorBeliefRangedBeacons(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64}, z::NamedTuple)::FullNormal
    # predicted mean and variance:
    μp = b.μ + a
    Σp = b.Σ + 𝒫.Σw

    # observation covariance:
    r = norm(μp - 𝒫.beacons[z.beacon_index, :])
    Σv = 0.1^2 * max(r, 𝒫.rmin)^2 * I(2)

    # shift relative observation to beacon position:
    z = z.obs + 𝒫.beacons[z.beacon_index, :]
    
    # Kalman gain:
    K = Σp * inv(Σp + Σv)

    # println("Kalman Gain K: ", K)

    # updated mean and covariance:
    μb′ = (I(2) - K) * μp + K * z
    Σb′ = (I(2) - K) * Σp

    return MvNormal(μb′, Σb′)
end


"""
    PosteriorBeliefRangedBeacons(𝒫, b, a, z)

Compute the posterior belief given a prior belief, action, and ranged beacon observation.

# Arguments:
- `𝒫::POMDPscenario`: The POMDP scenario
- `b::FullNormal`: The prior belief (Gaussian)
- `a::Vector{Float64}`: The action taken
- `z::NamedTuple`: The observation, containing `obs` and `beacon_index`
"""
function PosteriorBeliefBeacons(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64}, z::NamedTuple)::FullNormal
    # predicted mean and variance:
    μp = b.μ + a
    Σp = b.Σ + 𝒫.Σw

    # observation covariance:
    r = norm(μp - 𝒫.beacons[z.beacon_index, :])
    Σv = 0.1^2 * I(2)

    # shift relative observation to beacon position:
    z = z.obs + 𝒫.beacons[z.beacon_index, :]
    
    # Kalman gain:
    K = Σp * inv(Σp + Σv)

    # println("Kalman Gain K: ", K)

    # updated mean and covariance:
    μb′ = (I(2) - K) * μp + K * z
    Σb′ = (I(2) - K) * Σp

    return MvNormal(μb′, Σb′)
end


end