

export VrPCA,
        reduceToRank,
        getindex,
        SvdApprox,
        isConverged




"""
    # enum SvdApprox
                
. Lrapprox encodes for the initialization of vrpca with the LowRankApprox package
                
. Ifsvd  encodes for the initialization of vrpca with incremental svd of Vempala-Kunnan
                
"""        
@enum SvdApprox begin
    lrapprox
    ifsvd
end

"""
# VrPCA

returns an orthonormal matrix W with columns vectors the first
k left singular vectors  spanning the range of X.
**k is supposed to be small compared to d**.  

If X = U S Vt, the methode computes in W the matrix U[:, 1:k]
This structure gathers data to do stochastic svd as described in :
***Fast Stochastic Algorithms for SVD and PCA : convergences Properties and Convexity
Ohad Shamir 2015***


FIELDS
------

 - X  the (d,n) matrix we want to reduce.
 - k     the dimension into which we want to reducte the data Matrix
         ***Note :  If the obtained rank is less than asked for value of k is reset!***

 - W     orthogonal (d,k) matrix 
 - eta   step size
 - m     epoch length

 - U : Union{ Some{Matrix{Float64}}, Nothing} the matrix of left singular vectors if any
 - S : Union{ Some{Vector{Float64}}, Nothing} the vecor of singular values if any
 - Vt: Union{ Some{Matrix{Float64}}, Nothing} the transposed matrix of right singular vectors if any

"""
mutable struct VrPCA
    X::Matrix{Float64}
    k::Int64
    # orthogonal matrix (d,k)
    W::Matrix{Float64}
    converged::Bool
    relerr::Float64
    #
    U::Union{ Some{Matrix{Float64}}, Nothing}
    S::Union{ Some{Vector{Float64}}, Nothing}
    Vt::Union{ Some{Matrix{Float64}}, Nothing}
end

"""
# function VrPCA(data::Matrix{Float64}, k::Int64)


This function initialize a VrPCA problem with the LowRankApprox package

## Args
. data : the data matrix with data vectors in column
. k : the rank we want as output 
"""
function VrPCA(data::Matrix{Float64}, k::Int64)
        @debug  "VrPCA initialized by LowRankApprox and rank target"
        # get an initial W0, we do not need very good precision
        opts = LowRankApprox.LRAOptions(rank=k, rtol = 0.001)
        # get W such that Mdata ~ W*W'*Mdata  in Froebenius norm
        W = prange(data, opts)
        if size(W)[2] < k
            @warn "VrPCA initialization got less than asked rank, got : " size(W)[2]
        end
        @debug "got rank " size(W)[2]
        VrPCA(data, size(W)[2] , W, false, 1., nothing,nothing,nothing)
end



"""
# function VrPCA(data::Matrix{Float64}, epsil::Float64)


This function initialize a VrPCA problem with the LowRankApprox package

## Args
 - data : the data matrix with data vectors in column
 - epsil: the precision we ask at initialisation, the VrPCA iteration will further reduce error
         when iterating with the rank obtained at initialization
"""
function VrPCA(data::Matrix{Float64}, epsil::Float64)
    @debug "VrPCA initialized by Vempala's LowRankApproxMc and epsil target"
    lrmc = LowRankApproxMc(data)
    samplingSize = 100
    reduceToEpsil(lrmc, samplingSize, epsil)
    U = PartialSvdStoch.getindex(lrmc, :U)
    if U === nothing
        throw("initialization of VrPCA with epsil failed")
    end
    W = something(U)
    @debug "initialization got rank " size(W)[2]
    VrPCA(data, size(W)[2], W, false, 1. , nothing, nothing, nothing)
end


"""
# function iterateVrPCA(pcaPb::VrPCA, eta::Float64, m::Int64)



## Args
 - eta : the gradient step. Should be sufficiently small to ensure inversibility of the (k,k) matrix
         W*transpose(W) obtained during iterations on W(d,k)

 - m   : the batch size. Roughly inversely proportional to eta.

    eta = 0.01 and m = 20 is a good compromises. m does not need to be large
        as sampling is weighted by L2-norm of data column.

 Output : The function does not return any output.
    It fills the fields U, S, Vt of the structure VrPCA.
    The fields can be retrieved by the function getIndex.
"""
function iterateVrPCA(pcaPb::VrPCA, eta::Float64, m::Int64)
    @debug "entering VrPCA reduce eta m block version" eta m
    small = 1.E-5
    d,n = size(pcaPb.X)
    k = pcaPb.k
    # allocates once and for all 3 temproray matrices
    Us = zeros(Float64, d, k)  # is (d,k)
    B_t = zeros(Float64, k, k)   # Bt_1 is Wt*W so it is a (k,k) matrix
    W_tmp = zeros(Float64, d, k)
    pcaPb.converged = false
    s = 1
    W_s = pcaPb.W
    W_t = pcaPb.W
    deltaWIter = Vector{Float64}()
    # get L2 proba per column
    probaC = Vector{Float64}(undef,n)
    normC = Vector{Float64}(undef,n)
    normX = 0
    Threads.@threads for i = 1:n
        normC[i] = norm(pcaPb.X[:,i]) * norm(pcaPb.X[:,i])
    end
    for i = 1:n
        probaC[i] = i > 1 ? probaC[i-1] + normC[i] : normC[i]            
        normX += normC[i]
    end
    if probaC[end] <= 0
        @debug "\n non positive proba for last slot=  :  " probaC[end]
        error("cannot sample column")
    end
    map!(x -> x/probaC[end], probaC, probaC)
    #
    @debug " W_s dim" size(W_s)
    more = true
    while more 
        fill!(Us,0.)
        # split in block the update of Us to get speed from BLAS.gemm
        blockSize = 1000
        y = zeros(k, blockSize)
        nbblocs = floor(Int64, n / blockSize)
        for numbloc = 1:nbblocs
            blocfirst = 1 + (numbloc -1) * blockSize
            bloclast = min(blockSize * numbloc, n)
            # get  W_s' * pcaPb.X[:,i] in y
            BLAS.gemm!('T', 'N', 1., W_s , pcaPb.X[:,blocfirst:bloclast], 0., y)
            # Us = Us + pcaPb.X[:,i] * y'  (= transpose(pcaPb.X[:,i]) * W_s)
            for i in 1:blockSize
                j = (numbloc - 1) * blockSize + i
                BLAS.ger!(1.,pcaPb.X[:,j], y[:,i], Us)
            end
            # Us = Us + pcaPb.X[:,i] * (transpose(pcaPb.X[:,i]) * W_s)
        end
        # do not forget the residual part of block splitting!!
        if n % blockSize > 0
            y = zeros(k)
            for i in 1 + nbblocs * blockSize:n
                BLAS.gemm!('T', 'N', 1., W_s , pcaPb.X[:,i], 0., y)
                BLAS.ger!(1.,pcaPb.X[:,i], y, Us)
            end
        end
        Us = Us / n
        # stochastic update pass
        # we need to store Wt W_{s-1} (Ws1 in the code) W_{t-1} (Wt in the code)
        for t = 1:m
            # compute in place to avoid allocator transpose(W_{t_1}) * W_{s-1}, B_t is (k,k)
            BLAS.gemm!('T', 'N', 1., W_t, W_s, 0., B_t)
            # compute svd of transpose(W_{t_1}) * W_{s-1}    i.e do a svd of a (k,k) matrix
            U,S,Vt = LAPACK.gesvd!('A', 'A', B_t)
            # compute B_t as V*Ut
            BLAS.gemm!('T', 'T', 1., Vt, U, 0., B_t)
            #
            # biased sampling instead of it = rand(1:n)
            xsi = rand(Float64)
            it = searchsortedfirst(probaC, xsi)
            s_weight = 1. / (n * probaC[it])     
            # update Wt, vaux1 is  (1,k)
            vaux1 = transpose(pcaPb.X[:,it]) * W_t - (transpose(pcaPb.X[:,it]) * W_s) * B_t
            # vaux2 is (d,k)
            vaux2 = s_weight * pcaPb.X[:,it] * vaux1 + BLAS.gemm('N', 'N', Us, B_t)
            # @debug "norm vaux2" norm(vaux2)
            # W_s1 * B_t1 is (d,k), so W_t1 and W_s1. W_tmp _s (d,k)
            W_tmp = W_t + eta * vaux2
            #
            # final update of W_t, compute sqrt(transpose(W_t1)*W_{t}),  (k,k)
            # in fact we want to orthonormalize U and Vt are (k,k)
            U,S,Vt = LAPACK.gesvd!('A', 'A', transpose(W_tmp) * W_tmp) 
            # @debug "\n size U , Vt" size(U), size(Vt)
            # check for null values of S
            nbsmall = count( x -> x < small, S)
            if nbsmall > 0
                @debug "too small values " S
                @warn "could not do W update, ||W|| " norm(W_tmp, 2)
            end
            map!(x -> 1/sqrt(x) , S, S)
            # compute the inverse, BLAS.gemm does not seem to spped up things here
            Wnorminv = transpose(Vt) * diagm(0 => S) * transpose(U)
            # orthonormalize W. compute:  W_t / sqrt(transpose(W_t)*W_t)
            # @debug "\n size W_tmp Wnorminv" size(W_tmp) size(Wnorminv)
            W_t = BLAS.gemm('N', 'N', W_tmp, Wnorminv)
            # Cshould check W_t orthogonality : OK!
            # @debug "\n W orthogonality check " norm(transpose(W_t) * W_t - Matrix{Float64}(I,k,k))
        end
        #
        # convergence detection
        normWt = norm(W_t) * norm(W_t)
        deltaW = norm(W_t - W_s) * norm(W_t - W_s)
        lastDeltaW = 0
        if length(deltaWIter) > 0
            lastDeltaW = deltaWIter[end]
        end
        push!(deltaWIter, deltaW)
        @debug " iter ,  ||W_t - W_s||_2 , ||W_t||_2" s deltaW normWt
        W_s = W_t
        # 
        # do a convergence test
        if normWt > 0 && deltaW/normWt < 1.E-3
            pcaPb.W = W_t
            pcaPb.relerr = deltaW/normWt
            more = false
            pcaPb.converged = true
        else
            if s >= 3
                if deltaW >= 0.8 * lastDeltaW || deltaW <= 0.05 * deltaWIter[1]
                    @warn "\n reached stationarity  s deltaW  ||W|| ,  exiting :  " s  deltaW  normWt
                    pcaPb.relerr =  deltaW/normWt
                    pcaPb.W = W_t
                    pcaPb.converged = true
                    more  = false 
                elseif s > 5
                    # we stop without setting pcaPb.converged...
                    @warn "\n stopping iteration" s  deltaW  normWt
                    pcaPb.relerr =  deltaW/normWt
                    pcaPb.W = W_t                    
                    more = false
                end
            end
            s = s+1
        end
    end # end of while
    # now we have a converged matrix W (d,k) consisting in k left singular vectors
    # compute singular values and right singular vectors
    #                 = (k,d) * (d,n)
    #      Vt = transpose(W_t) * pcaPb.X
    Vt = Matrix{Float64}(undef, k, n)
    BLAS.gemm!('T', 'N', 1., W_t, pcaPb.X, 0., Vt)
    S = map(i -> norm(Vt[i,:]), (1:size(Vt)[1]))
    for i in 1:length(S)
        Vt[i,:] /= S[i]
    end
    pcaPb.U = Some(W_t)
    pcaPb.S = Some(S)
    pcaPb.Vt = Some(Vt)
end




"""
# function getindex(VrPCA, Symbol)

returns U,S,Vt,V or rank according to Symbol :S , :U, :Vt , V or :k

"""
function getindex(pcaPb::VrPCA, d::Symbol)
    @match d begin
    :S    =>  pcaPb.S
    :U    =>  pcaPb.U
    :V    =>  pcaPb.Vt'
    :Vt   =>  pcaPb.Vt
    :k    =>  length(pcaPb.S)
    _     =>  throw(KeyError(d))
    end
end



"""
# function getResidualError(vrpca::VrPCA)

if X is the data matrix and U computation succeeded 
    return the 2-uple (error, relerr) with:
```math
        error = norm(X - U * transpose(U) * X)
```  
and
```math
        relerror = norm(X - U * transpose(U) * X)/norm(X)
```

else throws an error
"""
function getResidualError(vrpca::VrPCA)
    U = something(vrpca.U)
    resErr = norm(vrpca.X - U* BLAS.gemm('T', 'N', U, vrpca.X))
    return resErr,resErr/norm(vrpca.X)
end


"""
# function isConverged(pcaPb::VrPCA)

return true if stationarity criteria were satisfied during iterations.
"""
function isConverged(pcaPb::VrPCA)
    return pcaPb.converged
end