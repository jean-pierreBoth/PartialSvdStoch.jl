module PartialSvdStoch

using Match

using Logging
using Base.CoreLogging

using Printf
using LinearAlgebra
using LowRankApprox
using Base.CoreLogging
using Statistics

debug_log = stdout
logger = ConsoleLogger(stdout, CoreLogging.Info)
global_logger(logger)

export vrPCA,
        SvdMode,
        LowRankApproxMc,
        reduceToRank,
        reduceToEpsil,
        iterateVrPCA,
        getindex,
        getResidualError,
        SvdApprox,
        isConverged


include("vrpca.jl")
include("lowRankApproxMc.jl")

end