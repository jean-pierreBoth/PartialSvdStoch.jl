using LinearAlgebra

"""
# enum SvdMode

. leftsv means that alogrithm operates on columns of data
passed in LowRankApproxMc and thus  giving
a orthonormal basis of left singular vector.

. rightsv means that algorithm stores the transpose of data passed
in LowRankApproxMc and thus runs on row of original data , thus giving
a orthonormal basis of right singular vector.

"""
@enum SvdMode begin
    leftsv
    rightsv
end


"""
# LowRankApproxMc structure 


as described in:

1. Vempala Spectral Book (2009) 
    paragraph on Iterative Fast Svd P 85 and Th 7.2 7.3

2. Lectures in Randomized Linear Algebra. Mahoney 2016 P121 and seq.

Cf also Deshpande Rademacher Vempala Wang 
Matrix Approximation and Projective Clustering via Volume Sampling 2006

For a data matrix X(d,n) compute a orthogonal set of left singular vectors in a matrix W(rank, n) 
Right singular vector are computed as W'*X.

FIELDS
------

 - X  :  the (d,n) matrix we want to reduce.
 - targetRank :   the rank of the data matrix we want to extract

 - currentRank : the rank reached at current iteration

 - U : Union{ Some{Matrix{Float64}}, Nothing} the matrix of left singular vectors if any
 - S : Union{ Some{Vector{Float64}}, Nothing} the vecor of singular values if any
 - Vt: Union{ Some{Matrix{Float64}}, Nothing} the transposed matrix of right singular vectors if any
     
CONSTRUCTORS
-----------

1.    `function LowRankApproxMc(data::Matrix{Float64}, targetRank::Int64)`
       initialize structure and rank of the approximation

"""
mutable struct LowRankApproxMc
    X::Matrix{Float64}
    #
    leftOrRight::SvdMode
    # rank we want in output
    targetRank::Int64
    targetEpsil::Float64
    # number of column sampling in a pass
    samplingSize::Int64
    # 
    currentRank::Int64
    # store relative error
    residualErr::Float64
    #
    currentIter::Int64
    U::Union{ Some{Matrix{Float64}}, Nothing}
    S::Union{ Some{Vector{Float64}}, Nothing}
    Vt::Union{ Some{Matrix{Float64}}, Nothing}
    #
    function LowRankApproxMc(data::Matrix{Float64}; mode = leftsv)
        if mode == leftsv
            @info "LowRankApproxMc in left singular vector mode"
            new(data, mode, 0, 1. , 0,  0, 1. , 0,
                nothing, nothing, nothing)
        else
            @info "LowRankApproxMc in right singular vector mode, transposing data"
            new(transpose(data), mode, 0, 1. , 0,  0, 1. , 0,
                nothing, nothing, nothing)
        end
    end
end

#
# The data matrix are supposed to be stored in columns of length data dimension.
# We operate on columns so we ouptut Left singular vectors.
# So our implementation conforms to Mahoney "Lecture Notes on Randomized Algebra"
# paragraph 15.4 Th 21 as opposed to Kannan-Vempala "Spectral Algorithms"
# which operates on rows and output right singular vectors.
#

function reduceIfast(datapb::LowRankApproxMc, stop::Function)
    @debug "in LowRankApproxMc reduceIfast v2"
    epsil = 1.E-5
    d,n = size(datapb.X)
    datapb.currentIter = 1
    datapb.currentRank = 0
    E = Matrix(datapb.X)
    acceptanceRatio = 1.
    normX = norm(E,2)
    residualAtIter= Vector{Float64}()
    meanL2NormC = sqrt(normX * normX / n)
    @debug "meanL2NormC" meanL2NormC
    # We could use c as in Deshpande Vempala and let c decrease with iterations
    # to accelerate ?
    probaC = Vector{Float64}(undef,n)
    normC = Vector{Float64}(undef,n)
    A = Matrix{Float64}(undef, d, 0)
    more = true
    while more
        @debug "\n\n iteration " datapb.currentIter
        # sampled columns at this iter
        tSet = Set{Int64}()
        keptIdxNonZero = Vector{Int64}()
        # sample columns according to L2-norm of each columns
        normE = 0
        Threads.@threads for i = 1:n
            normC[i] = norm(E[:,i]) * norm(E[:,i])
        end
        for i = 1:n
            probaC[i] = i > 1 ? probaC[i-1] + normC[i] : normC[i]            
            normE += normC[i]
        end
        if probaC[end] <= 0
            @debug "\n non positive proba for last slot=  :  " probaC[end]
            error("cannot sample column")
        end
        map!(x -> x/probaC[end], probaC, probaC)
        # get minimum non null column norm at iter t
        meanL2NormIter = mean(sqrt.(normC))
        @debug "\n c meanL2Norm at Iter  :  " meanL2NormIter
        for j = 1:datapb.samplingSize
            xsi = rand(Float64)
            s = searchsortedfirst(probaC, xsi)
            if s > length(probaC)
                @warn "\n sampling column failed xsi =  :  " xsi
                exit(1)
            end
            # avoid selecting some already sampled column or already a
            if !(s in tSet)
                push!(tSet, s)
            end
        end
        acceptanceRatio = length(tSet)/datapb.samplingSize
        @debug "acceptanceRatio" acceptanceRatio
        # get index of selected columns
        idxc = collect(tSet)
        At = datapb.X[:, idxc]
        # orthogonalize ... with preceding columns in A. Compute transpose(At)*A
        AttA = BLAS.gemm('T', 'N', At, A)
        Threads.@threads for i = 1:length(idxc)
            # with preceding columns in A. At[:,i] not normalized, A[:,j] is
            @simd for j = 1:size(A)[2]
                At[:,i] = At[:,i] - A[:,j] * AttA[i,j]
            end
        end
        @debug "QR orthogonalizing new block size At" size(At)
        At,tau  = LAPACK.geqrf!(At)
        Q=LAPACK.orgqr!(At, tau, length(tau))
        for i = 1:size(Q)[2]
            # within this block of columns in A.
            normi = norm(Q[:,i])
            # how much remains of At[:,i]. Should check with initial norm of column i
            if abs(normi - 1.) < 1.E-5
                push!(keptIdxNonZero, i)
#                @debug "adding a column vector of norm" normi idxc[i]
                A = hcat(A, Q[:,i])
                if size(A)[2] >=  min(d,n)
                    @debug "max dim reached"
                    more = false
                    break
                end
            else
                @debug "column not normalized" normi
            end 
        end
        @info "A size" size(A)
        #       
        datapb.currentRank = size(A)[2]
        # stopping criteria
        if stop(datapb)
            more = false
        else
            @debug "updating E"
            if length(keptIdxNonZero) < length(idxc)
                E = E - At[:, keptIdxNonZero] * BLAS.gemm('T', 'N', At[:, keptIdxNonZero], datapb.X)
            else
                E = E - At * BLAS.gemm('T', 'N', At, datapb.X)
            end
            datapb.residualErr = norm(E,2)/normX
            push!(residualAtIter, datapb.residualErr)
            # protection against stationarity
            if datapb.currentIter > 3 && residualAtIter[end] > 0.95 * residualAtIter[end-1]
                @warn "stationary iterations , stopping. perhaps increase rank"
                more = false
            end
            @info " iteration  , ||E|| set size "  datapb.currentIter datapb.residualErr  datapb.currentRank
            # if in reduceToEpsil mode we have to test now that we have datapb.residualErr
            if stop(datapb)
                more = false
            else
                datapb.currentIter += 1
            end
        end
    end
    # go from left singular to right singular vectors
    #
    @debug "computing right vectors"
    Vt = Matrix{Float64}(undef, size(A)[2], n)
    BLAS.gemm!('T', 'N', 1., A, datapb.X, 0., Vt)
    eigenValues = zeros(size(A)[2])
    Threads.@threads for i in 1:size(A)[2]
        eigenValues[i] = norm(Vt[i,:])
        Vt[i,:] = Vt[i,:] / eigenValues[i] 
    end
    @debug "exiting LowRankApproxMc reduceIfast"
    datapb.U = Some(A)
    datapb.S = Some(eigenValues)
    datapb.Vt = Some(Vt)
end


"""
# function getResidualError(lrmc::LowRankApproxMc)

if X is the data matrix and U computation succeeded 
    return the 2-uple (error, relerr) with :
```math
        error = norm(X - U * transpose(U) * X)
```
and :
```math
        relerr = Error/norm(X)
``` 

else throws an error
"""
function getResidualError(lrmc::LowRankApproxMc)
    U = something(lrmc.U)
    resErr = norm(lrmc.X - U* BLAS.gemm('T', 'N', U, lrmc.X))
    return resErr , resErr/norm(lrmc.X)
end


"""
# function getindex(LowRankApproxMc, Symbol)

returns U,S,Vt,V or rank according to Symbol :S , :U, :Vt , V or :k

"""
function getindex(datapb::LowRankApproxMc, d::Symbol)
    @match d begin
    :S    =>  datapb.S
    :U    =>  datapb.U
    :V    =>  datapb.Vt'
    :Vt   =>  datapb.Vt
    :k    =>  length(datapb.S)
    _     =>  throw(KeyError(d))
    end
end



"""
# function reduceToRank(datapb::LowRankApproxMc, expectedRank::Int64; nbiter = 2)


Implements incremental range approximation as described in 
1: Vempala Spectral Book (2009) 
    paragraph on Iterative Fast Svd P 85 and Th 7.2 7.3.



## Args
 -  expectedRank asked for. 
    It must be noted that the returned rank can be different
   from expectedRank, especially if some columns are dominant in the data matrix. In this case
   The sampling size used to select colmuns is set to ``\\frac{expectedRank}{nbiter}``

 - number of iterations. Default to 2. Can be set to 1 ito get get 
   a faster result at the expense of precision.

 ## Output
 
 The function does not return any output, instead it fills the fields U, S, Vt of the structure LowRankApproxMc.
 The fields can be retrieved by the function getIndex.
    If the rank extracted from the algorithm extractedRank we get as output :
  - U: a (d,extractedRank) matrix of orthonormalized left singular vector

  - S : a vector of singular values up to extractedRank

  - Vt : a (extractedRank , n) matrix , transposed of right singular vector. 
    Nota the vector are not orthonormal.

    The svd approximation of data is:

    `` B = U * S * transpose(V) `` 
    
    and verifies:

  `` E(|| data - B||^{2}) <=  1/(1-\\epsilon) * E(|| data - A_{k}|| ^{2}) + \\epsilon^{t} E(|| data ||^{2}) ``

 where ``A_{k}`` is the L2 best rank k approximation of data.
 B is of the form U * transpose(U) * data with U orthogonal made of left singular vectors.

 To get the corresponding result with right singular vector use a transpose Matrix as input.

 ## NOTA:

 **The Singular values are NOT sorted**

"""
function reduceToRank(datapb::LowRankApproxMc, rank::Int64; nbiter = 2)
    if rank > size(datapb.X)[1]
        error("reduceToRank reduction to rank greater than dimension")
    end
    maxdim = size(datapb.X)[2]
    datapb.currentIter = 1
    datapb.targetRank = rank
    # reset Epsil target in case of successive calls
    datapb.targetEpsil = 1.
    # we have rank target, 2 iterations should do
    if nbiter > 1
        datapb.samplingSize = floor(Int64,(1.05 * rank /nbiter))
        datapb.samplingSize = min(size(datapb.X)[2], datapb.samplingSize) 
        @debug "LowRankApproxMc.reduceToRank setting samplingSize: " datapb.samplingSize
        maxiter = min(nbiter, 5)
        @warn "LowRankApproxMc.reduceToRank using maxiter " maxiter
        fstop2(datapb) = datapb.currentIter >= maxiter || datapb.currentRank >= datapb.targetRank ? true : false
        reduceIfast(datapb, fstop2)
    else
        # only one iter
        maxiter = 1
        datapb.samplingSize = floor(Int64,(1.10 * rank /nbiter))
        datapb.samplingSize = min(size(datapb.X)[2], datapb.samplingSize) 
        @debug "LowRankApproxMc.reduceToRank setting samplingSize: " datapb.samplingSize
        fstop1(datapb) = datapb.currentIter >= maxiter ? true : false
        reduceIfast(datapb, fstop1)
    end
    # check we got rank
    if isnothing(datapb.U)
        @warn "reduceToRank  got not get left singular vectors"
    elseif size(datapb.U.value)[2] < rank
        @warn "reduceToRank  got only up to rank" size(datapb.U.value)[2]
    end
    # return rank approx
end


"""
# function reduceToEpsil(datapb::LowRankApproxMc, samplingSize::Int64, epsil::Float64;maxiter = 5)


This function computes an approximate svd up to a precision epsil.
    It computes U, S and Vt so that the matrix B = U*S*Vt verifies : 
    `` E(|| data - B||_{2}) / E(|| data ||_{2}) <=  epsil ``    
    It iterates until either the precision asked for is reached or *maxiter* iterations are done
        or rank reached is maximal.
## Args
------

-  samplingSize: the number of columns tentatively sampled by iteration.

- epsil : precision asked for. It is clear that a very small epsil will cause a full to be done
        with the corresponding costs. In this case possibly, a call to LAPACK.svd will be better.

- maxiter : the maximum number of iterations. default to 5

## Output:

The function does not return any output, instead it fills the fields U, S, Vt of the structure LowRankApproxMc.

The fields can be retrieved by the function getIndex.
If the rank extracted from the algorithm extractedRank we get as output :
- U: a (d,extractedRank) matrix of orthonormalized left singular vector
   
- S : a vector of singular values up to extractedRank
   
- Vt : a (extractedRank , n) matrix , transposed of right singular vector. 
       Nota the vector are not orthonormal.

The svd approximation of data is: `` B = U * S * transpose(V) `` 

and verifies:

`` E(|| data - B||^{2}) <=  1/(1-\\epsilon) * E(|| data - A_{k}|| ^{2}) + \\epsilon^{t} E(|| data ||^{2}) ``

where ``A_{k}`` is the L2 rank k approximation of data.
B is of the form U * transpose(U) * data with U orthogonal made of left singular vectors.

## NOTA:

 **The Singular values are NOT sorted**

"""
function reduceToEpsil(datapb::LowRankApproxMc, samplingSize::Int64, epsil::Float64; maxiter = 5)
    datapb.targetEpsil = epsil
    # reset targetRank in case of successive calls
    # rank is set to full dimension
    datapb.targetRank = min(size(datapb.X)[1], size(datapb.X)[2])
    if samplingSize <= 0
        throw("non strict positive samplingSize")
    end
    samplingSize = min(size(datapb.X)[2], samplingSize)
    datapb.samplingSize = samplingSize
    # epsil target
    # avoid doing one more iteration just for 1% of epsil
    fstop(datapb) = datapb.currentIter >= maxiter || datapb.residualErr <= epsil*(1.01) || 
            datapb.currentRank >= datapb.targetRank ? true : false
    reduceIfast(datapb, fstop)
end



# the following is faster as we do not ask for Vt

function getRangeApprox(data::Matrix{Float64}, rank::Int64)
    @debug " in getRangeApprox"
    svdpb = LowRankApproxMc(data)
    reduceToRank(svdpb, rank)
    U = something(svdpb.U)
    @debug "reduceToRank returned size" size(U)
    # compute all U columns and 0 columns of Vt 
    U,S,Vt = LAPACK.gesvd!('A', 'O', U)
    @debug "lapack returned size U,S,Vt"  size(U) size(S) size(Vt)
    # return first k columns of U and first k values of diagonal of S
    U[:, (1:rank)],S[1:rank]
end

