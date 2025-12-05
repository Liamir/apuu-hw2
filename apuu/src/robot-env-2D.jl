module RobotEnv2D

export POMDPscenario, InitParticleBelief, SampleMotionModel, GenerateObservation, ObsLikelihood, GenerateObservationFromBeacons
export GenerateRangedObservationFromBeacons, GenerateSingleRangedBeaconObservation
export beacon_grid

using apuu.ParticleFilter: ParticleBelief, PosteriorParticleBelief
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
    rmin::Union{Float64, Nothing} = nothing # optional: minimum range for observation deterioration
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
function InitParticleBelief(n_particles::Int, μ0::Vector{Float64}, Σ0::AbstractMatrix)::ParticleBelief
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
Generate a relative observation from a single beacon.
"""
function GenerateSingleBeaconObservation(
    𝒫::POMDPscenario, 
    robot_x::Vector{Float64}, 
    beacon_x::Vector{Float64}
    )::Vector{Float64}
    v = rand(𝒫.rng, MvNormal(zeros(2), 𝒫.Σv)) # v ~ N(0, Σv)
    z_rel = (robot_x - beacon_x) + v
    return z_rel
end


function GetFirstBeaconWithinDistance(
    𝒫::POMDPscenario,
    robot_x::Vector{Float64},
    )::Union{Tuple{Vector{Float64}, Int}, Nothing}

    # return index of the first beacon under distance threshold d:
    for i in 1:size(𝒫.beacons, 1)
        beacon_x = 𝒫.beacons[i, :]
        if norm(robot_x - beacon_x) <= 𝒫.d
            return (beacon_x, i)
        end
    end
    
    return nothing
end


"""
Generate a relative observation from the first beacon under distance d.
"""
function GenerateObservationFromBeacons(
    𝒫::POMDPscenario,
    robot_x::Vector{Float64},
    )::Union{Tuple{Vector{Float64}, Int}, Nothing}

    # return an observation from the first beacon under distance threshold d:
    result = GetFirstBeaconWithinDistance(𝒫, robot_x)
    if result !== nothing
        (beacon_x, i) = result
        z_rel = GenerateSingleBeaconObservation(𝒫, robot_x, beacon_x)
        return (z_rel, i)
    end
    
    return nothing
end



"""
Likelihood of an obseration given a state.
"""
function ObsLikelihood(𝒫::POMDPscenario, z::Vector{Float64}, x::Vector{Float64})::Float64
    # input observation z and state x
    # output likelihood of the observation given the state
    return pdf(MvNormal(x, 𝒫.Σv), z)
end

function ObsFromBeaconsLikelihood(𝒫::POMDPscenario; z_rel::Vector{Float64}, robot_x::Vector{Float64})::Union{Float64, Nothing}

    result = GetFirstBeaconWithinDistance(𝒫, robot_x)
    if result !== nothing
        (beacon_x, i) = result
        x_rel = robot_x - beacon_x
        return ObsLikelihood(𝒫, z_rel, x_rel)
    end

    return 0. # zero chance of receiving an observation if no beacon is close enough, can say this robot_x is impossible given z_rel
end

function ObsFromBeaconsLikelihood(𝒫::POMDPscenario, z_rel::Vector{Float64}, robot_x::Vector{Float64})::Union{Float64, Nothing}
    return ObsFromBeaconsLikelihood(𝒫, z_rel=z_rel, robot_x=robot_x)
end

function PosteriorParticleBeliefBeacons(𝒫::POMDPscenario, b::ParticleBelief, a::Vector{Float64}, z′::Vector{Float64})::ParticleBelief
    return PosteriorParticleBelief(𝒫, b, a, z′, SampleMotionModel, ObsFromBeaconsLikelihood)
end






"""
HW3
"""


"""
9x9 grid of beacons at (0,0), (0,4.5), (0,9), (4.5,0), ..., (9,9)
"""
function beacon_grid()
    # define beacon locations:
    _beacon_grid = [0, 4.5, 9]
    return hcat([[x, y] for x in _beacon_grid, y in _beacon_grid]...)'
end


"""
Generate a relative observation from a single beacon (assumes robot within distance d)
"""
function GenerateSingleRangedBeaconObservation(
    𝒫::POMDPscenario,
    robot_x::Vector{Float64},
    beacon_x::Vector{Float64}
    )::Vector{Float64}

    # noise increases with distance:
    rmin = 𝒫.rmin
    r = norm(robot_x - beacon_x)
    Σv = (0.1*max(r,rmin))^2 * I(2)

    v = rand(𝒫.rng, MvNormal(zeros(2), Σv)) # v ~ N(0, Σv)
    z_rel = (robot_x - beacon_x) + v
    return z_rel
end


"""
Generate a relative observation from the first beacon under distance d, with signal deterioration (minimum range rmin).
"""
function GenerateRangedObservationFromBeacons(
    𝒫::POMDPscenario,
    robot_x::Vector{Float64},
    )::Union{Tuple{Vector{Float64}, Int}, Nothing}

    # return an observation from the first beacon under distance threshold d:
    result = GetFirstBeaconWithinDistance(𝒫, robot_x)
    if result !== nothing
        (beacon_x, i) = result
        z_rel = GenerateSingleRangedBeaconObservation(𝒫, robot_x, beacon_x)
        return (z_rel, i)
    end
    
    return nothing
end


end # module RobotEnv2D