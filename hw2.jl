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
end

@with_kw struct ParticleBelief
    particles::Vector{Vector{Float64}} # list of size N (N is number of particles) of 2x1 vectors (x-y position states)
    weights::Vector{Float64} # list of size N of weights
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

function InitParticleBelief(𝒫::POMDPscenario, n_particles::Int, μ0::Vector{Float64}, Σ0::Matrix{Float64})::ParticleBelief
    particles = [rand(MvNormal(μ0, Σ0)) for _ in 1:n_particles]
    weights = fill(1.0 / n_particles, n_particles)

    return ParticleBelief(particles, weights)
end

function SampleMotionModel(𝒫::POMDPscenario, a::Vector{Float64}, x::Vector{Float64})
    # w ~ N(0, Σw)
    w = rand(𝒫.rng, MvNormal(zeros(2), 𝒫.Σw))
    x_next = 𝒫.F * x + a + w
    return x_next
end

function GenerateObservation(𝒫::POMDPscenario, x::Vector{Float64})
    # v ~ N(0, Σv)
    v = rand(𝒫.rng, MvNormal(zeros(2), 𝒫.Σv))
    z = x + v
    return z
end

function PropagateParticleBelief(𝒫::POMDPscenario, b::ParticleBelief, a::Vector{Float64})::ParticleBelief
    new_particles = Vector{Vector{Float64}}(undef, length(b.particles))
    for (i, particle) in enumerate(b.particles)
        new_particles[i] = SampleMotionModel(𝒫, a, particle)
    end
    new_weights = copy(b.weights)
    return ParticleBelief(new_particles, new_weights)
end

function ObsLikelihood(𝒫::POMDPscenario, z::Vector{Float64}, x::Vector{Float64})::Float64
    # input observation z and state x
    # output likelihood of the observation given the state
    return pdf(MvNormal(x, 𝒫.Σv), z)
end

function PosteriorParticleBelief(𝒫::POMDPscenario, b::ParticleBelief, a::Vector{Float64}, z′::Vector{Float64})::ParticleBelief
    new_particles = Vector{Vector{Float64}}(undef, length(b.particles))
    new_weights = Vector{Float64}(undef, length(b.particles))
    
    for (i, particle) in enumerate(b.particles)
        new_particles[i] = SampleMotionModel(𝒫, a, particle)
        new_weights[i] = ObsLikelihood(𝒫, z′, new_particles[i]) * b.weights[i]
    end
    
    # normalize weights
    weight_sum = sum(new_weights)
    new_weights = new_weights ./ weight_sum
    
    return ParticleBelief(new_particles, new_weights)
end

function ResampleParticles(𝒫::POMDPscenario, b::ParticleBelief, ess_threshold::Float64)::ParticleBelief
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

function ObsFromBeaconsLikelihood(𝒫::POMDPscenario, z::Vector{Float64}, x::Vector{Float64})::Float64
    # add your code here...
end

function PosteriorParticleBeliefBeacons(𝒫::POMDPscenario, b::ParticleBelief, a::Vector{Float64}, z::Vector{Float64})::ParticleBelief
    # add your code here...
    return ParticleBelief(new_particles, new_weights)
end


function main()
    # definition of the random number generator with seed
    ID = 318803129
    rng = MersenneTwister(ID)

    # parameters
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0;
          0.0 1.0]
    F = [1.0 0.0;
         0.0 1.0]
    Σw = 0.1^2 * [1.0 0.0;
                  0.0 1.0]
    Σv = [1.0 0.0;
          0.0 1.0]
    d = 1.0

    # set beacons locations
    beacons = Matrix{Float64}(undef, 0, 2)  # 0 rows, 2 columns (empty matrix)

    # initialize the scenario
    𝒫 = POMDPscenario(F=F, Σw=Σw, Σv=Σv, rng = rng, beacons=beacons, d=d)

    # initialize particle belief
    n_particles = 10
    b0 = InitParticleBelief(𝒫, n_particles, μ0, Σ0)

    xgt0 = [-0.5, -0.2] # ground truth initial location
    ai = [1.0, 1.0] # action of the action sequence
    T = 10 # time steps

    # generate motion trajectory
    τ = [xgt0] # trajectory
    for i in 0:T-1
        push!(τ, SampleMotionModel(𝒫, ai, τ[end]))
    end

    @show τ

    # generate observation trajectory
    τ_obs = []
    for i in 1:T
        push!(τ_obs, GenerateObservation(𝒫, τ[i+1]))
    end

    @show τ_obs

    pl_gt = plot([x[1] for x in τ], [x[2] for x in τ], label="gt", markershape=:diamond, title="Trajectory and Observations", color=:black, aspect_ratio=:equal)
    scatter!(pl_gt, [z[1] for z in τ_obs], [z[2] for z in τ_obs], label="obs", markershape=:star5, color=:orange)
    savefig(pl_gt,"ground_truth_$ID.png")

    # iii. belief propagation (without observation update)
    beliefs = Vector{ParticleBelief}(undef, T+1)
    beliefs[1] = b0
    for t in 1:T
        beliefs[t+1] = PropagateParticleBelief(𝒫, beliefs[t], ai)
    end

    # plot the actual trajectory and the propagated particle beliefs (with scatterParticles)
    pl_beliefs = plot([x[1] for x in τ], [x[2] for x in τ], label="gt", markershape=:diamond, title="Trajectory and Particle Beliefs", color=:black, aspect_ratio=:equal)
    for t in 1:T+1
        scatterParticles!(pl_beliefs, beliefs[t], "t=$(t-1)")
    end
    savefig(pl_beliefs,"particle_beliefs_$ID.png")

    # iv. belief update with observations
    beliefs_upd = Vector{ParticleBelief}(undef, T+1)
    beliefs_upd[1] = b0
    for t in 1:T
        beliefs_upd[t+1] = PosteriorParticleBelief(𝒫, beliefs_upd[t], ai, τ_obs[t])
    end
    # plot the actual trajectory and the updated particle beliefs (with scatterParticles)
    pl_beliefs_upd = plot([x[1] for x in τ], [x[2] for x in τ], label="gt", markershape=:diamond, title="Trajectory and Updated Particle Beliefs", color=:black, aspect_ratio=:equal)
    for t in 1:T+1
        scatterParticles!(pl_beliefs_upd, beliefs_upd[t], "t=$(t-1)")
    end
    savefig(pl_beliefs_upd,"updated_particle_beliefs_$ID.png")

    # v. belief update with observations and resampling
    ess_threshold = n_particles / 2.0  # change to Inf to always resample
    beliefs_resamp = Vector{ParticleBelief}(undef, T+1)
    beliefs_resamp[1] = b0
    for t in 1:T
        b_upd = PosteriorParticleBelief(𝒫, beliefs_resamp[t], ai, τ_obs[t])
        beliefs_resamp[t+1] = ResampleParticles(𝒫, b_upd, ess_threshold)
    end
    # plot the actual trajectory and the resampled particle beliefs (with scatterParticles)
    pl_beliefs_resamp = plot([x[1] for x in τ], [x[2] for x in τ], label="gt", markershape=:diamond, title="Trajectory and Resampled Particle Beliefs", color=:black, aspect_ratio=:equal)
    for t in 1:T+1
        scatterParticles!(pl_beliefs_resamp, beliefs_resamp[t], "t=$(t-1)")
    end
    savefig(pl_beliefs_resamp,"resampled_particle_beliefs_$ID.png")

end

main()