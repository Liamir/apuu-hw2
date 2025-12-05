module Paths

# Project root (apuu/src -> apuu -> root)
const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))

# plot directories:
const PLOTS = joinpath(PROJECT_ROOT, "plots")
const HW2_PLOTS = joinpath(PLOTS, "hw2")
const HW3_PLOTS = joinpath(PLOTS, "hw3")

export HW2_PLOTS
export HW3_PLOTS

end