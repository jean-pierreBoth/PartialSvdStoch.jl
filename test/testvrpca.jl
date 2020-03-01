using PartialSvdStoch, Test

using LowRankApprox


function testvrpca_withLowRankApprox(mat, k)
    #
    normMat = norm(mat)
    # get initial spectrum from psvd
    opts = LRAOptions(rank=k, rtol=0.001)
    @info "testvrpca_withLowRankApprox : initialization"
    @time Uguess, Sguess, Vguess = psvd(mat,opts)
    matGuess = Uguess * diagm(Sguess) * transpose(Vguess)
    deltaMatGuess = norm(matGuess - mat)
    @info "rank obtained : " size(Uguess)[2]
    @info " residual Error , initial L2 norm " deltaMatGuess normMat
    @info " relative error at initialization " deltaMatGuess/normMat
    #
    eta = 0.01
    batchSize = 20
    @info "doing vrpca svd  with rank initialization..." eta batchSize
    pcaPb = VrPCA(mat, k) 
    @time iterateVrPCA(pcaPb, eta, batchSize)
    U = PartialSvdStoch.getindex(pcaPb,:U)
    S = PartialSvdStoch.getindex(pcaPb,:S)
    Vt = PartialSvdStoch.getindex(pcaPb,:Vt)
    # it happens that the obtained rank is less than asked for
    obtainedRank = pcaPb.k
    matVrpca = U.value * diagm(S.value) * Vt.value
    deltaMatIter = norm(matVrpca - mat)
    normMat = norm(mat)
    @info "deltaS after vr"  norm(matVrpca - mat) normMat
    relErr = deltaMatIter/normMat
    @info " relative error after vrpca iterations : " deltaMatIter/normMat
    relErr < 0.05 ? true : false
end


"""
# function testvrpca_withepsil(mat, epsil)

This test extracts the rank to get approximation at given precision epsil
see test : vrpca_epsil
"""
function testvrpca_withepsil(mat, epsil)
    #
    normMat = norm(mat)
    @info "testvrpca_withepsil : doing vrpca initialization with epsil" epsil
    pcaPb = VrPCA(mat, epsil) 
    #
    eta = 0.01
    batchSize = 20
    @info "doing vrpca svd iteration" eta batchSize
    @time iterateVrPCA(pcaPb, eta, batchSize)
    #
    U = PartialSvdStoch.getindex(pcaPb,:U)
    S = PartialSvdStoch.getindex(pcaPb,:S)
    Vt = PartialSvdStoch.getindex(pcaPb,:Vt)
    matVrpca = U.value * diagm(S.value) * Vt.value
    relErr = norm(matVrpca - mat)/normMat
    @info "residual error after vr"  norm(matVrpca - mat) normMat relErr
    relErr < 0.05 ? true : false
end