using PartialSvdStoch, LinearAlgebra, Test


# reduceToRank comparison with exact svd
# we check that the reconstructed matrix has nearly the same singular values
function testifastsvd_S(mat, rank)
    @info "doing exact svd ..."
    @time Uexact,Sexact,Vtexact = svd(mat)
    # 
    # 
    svdpb = LowRankApproxMc(mat) 
    @time reduceToRank(svdpb, rank, nbiter = 2)
    U = PartialSvdStoch.getindex(svdpb,:U)
    S = PartialSvdStoch.getindex(svdpb,:S)
    Vt = PartialSvdStoch.getindex(svdpb,:Vt)
    rankReached = svdpb.currentRank
    matApprox = U.value * diagm(S.value) * Vt.value
    @info "relative residual error LowRankApproxMc " norm(matApprox - mat)/norm(mat)
    # doing exact svd of matApprox
    @time Urecons,Srecons,Vtrecons = svd(matApprox)
    deltaRecons  = norm(Srecons[1:rankReached]-Sexact[1:rankReached])
    relerr = deltaRecons/norm(Sexact[1:rankReached])
    @info "relative error on eigen values after reconstruction : " relerr
    relerr < 0.1 ? true : false
end



# reduceToEpsil comparison with exact svd
function testifastsvd_epsil(mat)
    @debug "doing exact svd ..."
    @time Uexact,Sexact,Vtexact = svd(mat)
    #
    @info "reducing to epsil " 0.1
    svdpb = LowRankApproxMc(mat) 
    @time reduceToEpsil(svdpb, 50, 0.01)
    U = PartialSvdStoch.getindex(svdpb,:U)
    S = PartialSvdStoch.getindex(svdpb,:S)
    Vt = PartialSvdStoch.getindex(svdpb,:Vt)
    rankReached = svdpb.currentRank
    deltaS  = norm(S.value[1:rankReached]-Sexact[1:rankReached])/norm(Sexact[1:rankReached])
    matApprox = U.value * diagm(S.value) * Vt.value
    relerrMat = norm(matApprox - mat)/norm(mat)
    @info "delta mat after LowRankApproxMc reduce" deltaS  relerrMat
    relerr < 0.05 ? true : false
end



# ifsvd and epsil iterations
function testifastsvd_epsil(mat,  samplingSize, epsil)  
    #
    @debug "doing LowRankApproxMc reduceToRank"
    svdpb = LowRankApproxMc(mat) 
    @time reduceToEpsil(svdpb, samplingSize , epsil)
    U = PartialSvdStoch.getindex(svdpb,:U)
    S = PartialSvdStoch.getindex(svdpb,:S)
    Vt = PartialSvdStoch.getindex(svdpb,:Vt)
    rankReached = svdpb.currentRank
    matApprox = U.value * BLAS.gemm('T','N', U.value, mat)
    relerrMat = norm(matApprox - mat)/norm(mat)
    @info "delta mat after LowRankApproxMc reduce"  norm(matApprox - mat) norm(mat)
    @info "relative residual error" norm(matApprox - mat)/norm(mat)
    relerrMat < 0.05 ? true : false
end


# ifsvd and rank control
function testifastsvd_rank(mat,  k)  
    #
    @info "\n doing LowRankApproxMc reduceToRank"
    svdpb = LowRankApproxMc(mat) 
    @time reduceToRank(svdpb, k)
    U = PartialSvdStoch.getindex(svdpb,:U)
    S = PartialSvdStoch.getindex(svdpb,:S)
    Vt = PartialSvdStoch.getindex(svdpb,:Vt)
    rankReached = svdpb.currentRank
    matApprox = U.value * BLAS.gemm('T','N', U.value, mat)
    relerrMat = norm(matApprox - mat)/norm(mat)
    @info "delta mat after LowRankApproxMc reduce"  norm(matApprox - mat) norm(mat)
    @info "relative residual error" norm(matApprox - mat)/norm(mat)
    relerr < 0.05 ? true : false
end