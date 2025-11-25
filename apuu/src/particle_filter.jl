module ParticleFilter

# exports:
export ParticleBelief, PropagateParticleBelief, PosteriorParticleBelief, ResampleParticles

# dependencies:
# using Revise
# using Distributions
# using Random
# using LinearAlgebra
# using Plots
# using StatsPlots
using Parameters
using StatsBase


@with_kw struct ParticleBelief
    particles::Vector{Vector{Float64}} # list of size N (N is number of particles) of 2x1 vectors (x-y position states)
    weights::Vector{Float64} # list of size N of weights
end


"""

Applies the motion model to each particle in the belief given action `a`.

# Note:
Doesn't update weights (this requires an observation, see PosteriorParticleBelief).

"""
function PropagateParticleBelief(𝒫, b::ParticleBelief, a::Vector{Float64}, particle_motion::Function)::ParticleBelief
    new_particles = Vector{Vector{Float64}}(undef, length(b.particles))
    for (i, particle) in enumerate(b.particles)
        new_particles[i] = particle_motion(𝒫, a, particle)
    end
    new_weights = copy(b.weights)
    return ParticleBelief(new_particles, new_weights)
end



"""
TODO - use PropagateParticleBelief instead of duplicating code
"""
function PosteriorParticleBelief(
    𝒫, 
    b::ParticleBelief,
    a::Vector{Float64}, 
    z′::Vector{Float64}, 
    particle_motion::Function, 
    obs_likelihood::Function,
    )::ParticleBelief

    new_particles = Vector{Vector{Float64}}(undef, length(b.particles))
    new_weights = Vector{Float64}(undef, length(b.particles))
    
    for (i, particle) in enumerate(b.particles)
        new_particles[i] = particle_motion(𝒫, a, particle)
        new_weights[i] = obs_likelihood(𝒫, z′, new_particles[i]) * b.weights[i]
    end
    
    # normalize weights
    weight_sum = sum(new_weights)
    new_weights = new_weights ./ weight_sum
    
    return ParticleBelief(new_particles, new_weights)
end



"""

Computes the effective sample size and resamples particles if below threshold.

"""
function ResampleParticles(𝒫, b::ParticleBelief, ess_threshold::Float64)::ParticleBelief
    # compute effective sample size
    ess = 1.0 / sum(w^2 for w in b.weights)
    if ess >= ess_threshold
        return b
    end
    new_particles = Vector{Vector{Float64}}(undef, length(b.particles))
    indices = sample(𝒫.rng, 1:length(b.particles), Weights(b.weights), length(b.particles); replace=true)
    for (i, idx) in enumerate(indices)
        new_particles[i] = b.particles[idx]
    end
    new_weights = fill(1.0 / length(b.particles), length(b.particles))
    
    # hint: use function `StatsBase.sample`
    return ParticleBelief(new_particles, new_weights)
end


end # module ParticleFilter