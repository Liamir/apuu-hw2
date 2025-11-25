module RobotEnv2D

export POMDPscenario, InitParticleBelief, SampleMotionModel, GenerateObservation, ObsLikelihood

using apuu.ParticleFilter: ParticleBelief
using Distributions
using Random
using Parameters
using LinearAlgebra
using Plots
using StatsPlots


"""
    POMDPscenario(F=F, Σw=Σw, Σv=Σv, rng = rng, beacons=beacons, d=d)

POMDP scenario for a 2D robot environment

# Arguments:
- `F::Matrix{Float64}`: State transition matrix (2x2)
- `Σw::Matrix{Float64}`: Process noise covariance matrix (2x2)
- `Σv::Matrix{Float64}`: Observation noise covariance matrix (2x2)
- `rng::MersenneTwister`: Random number generator
- `beacons::Matrix{Float64}`: Locations of beacons (nx2 matrix, n is number of beacons)
- `d::Float64`: Some distance parameter
"""
@with_kw mutable struct POMDPscenario
    F::Matrix{Float64} # 2x2
    Σw::Matrix{Float64} # 2x2
    Σv::Matrix{Float64} # 2x2
    rng::MersenneTwister
    beacons::Matrix{Float64} # nx2 (n is number of beacons)
    d::Float64
end


"""
    InitParticleBelief(𝒫, n_particles, μ0, Σ0)
    
Initialize a particle belief based on a Gaussian prior.

# Arguments:
- `𝒫::POMDPscenario`: The POMDP scenario
- `n_particles::Int`: Number of particles
- `μ0::Vector{Float64}`: Initial mean of the Gaussian prior
- `Σ0::Matrix{Float64}`: Initial covariance of the Gaussian prior

# Returns:
- `ParticleBelief`: The initialized particle belief
"""
function InitParticleBelief(𝒫::POMDPscenario, n_particles::Int, μ0::Vector{Float64}, Σ0::AbstractMatrix)::ParticleBelief
    particles = [rand(MvNormal(μ0, Σ0)) for _ in 1:n_particles]
    weights = fill(1.0 / n_particles, n_particles)

    return ParticleBelief(particles, weights)
end

"""
Help function, scatters the samples of the given belief, with respect to their weights, on the given plot.
- `pl`: Plot to add scatter particles.
- `belief`: Particle Belief with `particles` and `weights`.
- `label`: Text label of the particles to add to the plot.
"""
function scatterParticles!(pl, belief::ParticleBelief, label::String)
    n_particles = length(belief.particles)
    x = [particle[1] for particle in belief.particles]
    y = [particle[2] for particle in belief.particles]
    w = belief.weights
    scatter!(pl, x, y, markershape=:circle, markersize=w .* n_particles*5, markerstrokewidth=0, markercolor=:auto, alpha=0.5, label=label)
end


"""

"""
function SampleMotionModel(𝒫::POMDPscenario, a::Vector{Float64}, x::Vector{Float64})::Vector{Float64}
    
    # deterministic forward step:
    fx = 𝒫.F * x + a

    # add noise:
    w = rand(𝒫.rng, MvNormal(zeros(2), 𝒫.Σw)) # w ~ N(0, Σw)
    next_x = fx + w

    return next_x
end

"""

Use the POMDP scenario to generate an observation given the state `x`.

"""
function GenerateObservation(𝒫::POMDPscenario, x::Vector{Float64})::Vector{Float64}
    v = rand(𝒫.rng, MvNormal(zeros(2), 𝒫.Σv)) # v ~ N(0, Σv)
    z = x + v
    return z
end


"""

Likelihood of an obseration given a state.

"""
function ObsLikelihood(𝒫::POMDPscenario, z::Vector{Float64}, x::Vector{Float64})::Float64
    # input observation z and state x
    # output likelihood of the observation given the state
    return pdf(MvNormal(x, 𝒫.Σv), z)
end


end # module RobotEnv2D