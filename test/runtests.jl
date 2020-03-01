using Test, PartialSvdStoch, Serialization, Distributions


using Logging
using Base.CoreLogging

logger = ConsoleLogger(stdout, CoreLogging.Debug)
global_logger(logger)

include("testvrpca.jl")
include("testifsvd.jl")

# a data file for a matrix(11520, 22499) can be found in ./data/plimat.serialized
# mass spectrometry image

plimat = Some(Matrix{Float64})
dataf = "data/plimat.serialized"
if isfile(dataf)
    plimat = Some(Serialization.deserialize(open(dataf, "r")))
end



# This test compares singular values from exact svd with those extracted from
# incremental svd from LowRankApproxMc

@testset "ifsvd_versus_svd" begin
    @info "========================================================="
    @info "\n\n in test ifsvd_versus_svd"
    d = 500
    n = 5000
    k = 50
    mat = rand(d,n)
    for j in 1:k
    mat[:,j] = mat[:,j] * 20.
    end
    for j in 1:5
        mat[:,j+k] = mat[:,j+k] * 100.
    end
    @test testifastsvd_S(mat, 200)
end


# This test shows how in presence high variability of data
# it is possible to get a competitive approximation with incremental svd.

@testset "ifsvd_lowRankApproxEpsil" begin
    @info "========================================================="
    @info "\n\n in test ifsvd_lowRankApproxEpsil"
    d = 10000
    n = 20000
    mat = zeros(d,n)
    #
    Xlaw = Normal(0, 1000.)
    # multiply some columns
    blocsize = 100
    for bloc in 1:round(Int,n/blocsize)
        val = abs(rand(Xlaw))
        for i in 1:d
            if rand() < 0.01
                for j in 1:blocsize        
                    mat[i,(bloc-1)*blocsize+j] = val * (1. - rand()/10.)
                end
            end
        end
    end
    @test testifastsvd_epsil(mat, 100, 0.05)
end


#
# test vrpca_epsil

# This tests asks for an approx at 0.01
# It extracts a rank 100 matrix at 0.005 precision within 3s (after first compilation).
# With LowRankApprox.prange alone we need to go up to rank=500 (initialized with 
# opts = LowRankApprox.LRAOptions(rank=500, rtol = 0.0001) to achieve a relative 
# Froebonius error of  0.005. It then runs in 1.15s.

 
# So we see that LowRankApprox is really fast but needs many more singular vectors
# and that VrPCA gives a clear reduction of relative error.
# Of course VrPCA initialized with LowRankApprox would give an analogous reduction
# in relative error. Cf testset "vrpca_withLoxwRankApprox"
#

@testset "vrpca_epsil" begin
    @info "========================================================="
    @info "\n\n in test vrpca_epsil"
    d = 5000
    n = 10000
    k = 100
    maxval = 1000
    mat = rand(d,n)
    # multiply some columns
    for i in 1:k
        mat[:,i] = mat[:,i] * maxval
    end
    @test testvrpca_withepsil(mat, 0.01)
end


@testset "vrpca_withLowRankApprox" begin
    @info "========================================================="
    @info "\n\n in test vrpca_withLowRankApprox"
    d = 5000
    n = 10000
    k = 100
    maxval = 1000
    mat = rand(d,n)
    # multiply some columns. but do not sample them( some will be sampled twice and will affect sketching)
    for i in 1:k
        mat[:,i] = mat[:,i] * maxval
    end
    @test testvrpca_withLowRankApprox(mat, 100)
end