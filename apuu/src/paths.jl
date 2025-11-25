module Paths

# Project root (apuu/src -> apuu -> root)
const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))

# plot directories:
const PLOTS = joinpath(PROJECT_ROOT, "plots")
const HW2_PLOTS = joinpath(PLOTS, "hw2")

export HW2_PLOTS

end