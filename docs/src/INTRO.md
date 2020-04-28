
# PartialSvdStoch

This package provides approximate partial SVD using stochastic methods.

It implements algorithms descrided in :

1. The paper by Ohad Shamir: **Fast Stochastic Algorithms for SVD and PCA : convergences Properties and Convexity(2015)**.

The algorithm combines a stochastic gradient approach and iterative techniques. It requires
a correct (see the paper) initialization of the rank k matrix of first left singular vectors, then exponential convergence to the exact rank-k approximation is proven in the paper.
The first left singular vectors can be initialized using the *impressive* LowRankApprox package if the rank of approximation required is known, or by the algorithm of Vempala-Kannan (Cf below) if the requirement is a given precision.  
The structure related to the implementation of the algorithm is **VrPCA**.

**The implementation deviates from the original article on one point**: \
We sample column vectors from the data matrix in the iteration pass proportionally to their L2-norm (as in the Vempala book see below) and do a weight correction to keep an unbiaised estimator. Tests shows a faster numerical convergence.

The test *vrpca_epsil* shows a case where we can get the same relative error with a rank
5 times less than without the VrPCA iterations.

2. The chapter on fast incremental Monte-Carlo Svd from **Spectral Algorithms. S. Vempala and R. Kannan(2009)**
(chapter 7 P 85 and Th 7.2 7.3).

The algorithm consists in sampling columns a fixed number *samplingSize* of data vectors proportionally
to their L2-norm without replacement.
Vectors sampled are orthogonalised thus constructing a first pass of the range approximation.  
To get a more precise estimation another iteration can be done. Then the first approximation is substracted from the range of the data and another set of columns vectors are sampled from the residual to get smaller components of the range. The process can be iterated providing exponential convergence to the rank-k approximation as proved in Vempala.  \
Our implementation deviates slightly from the paper as we let the final rank obtained fluctuate depending on the nature of data. As some columns can be rejected if alredy sampled in a pass, we can get a variable rank at the end of the algorithm.
This method is not as fast as the *LowRankApprox* package but can provide a competitive approximation when the matrix of data has large variations of L2-norm between columns as the sampling construct the approximation with the largest contribution of the L2-norm of the data matrix.
(Cf tests for examples).

The function *reduceToEpsil* computes an approximation B at a given precision $\epsilon$ :  $\frac{E(|| data - B||_{2})}{E(|| data ||_{2})} <=  \epsilon$
and determines the rank accordingly.

The function *reduceToRank* needs usually 2 iterations to achieve a good compromise and that is the default used in the code. Run times are of the order of 4.5s for a (10000, 20000) matrix with 2 iterations for a rank = 100 approximation.

## Miscellaneous

There is a logger installed in the code which is in the Info mode (See beginning of PartialSvdStoch.jl).
It can be set in Debug mode and will then provide some log on variables monitoring convergence.

It must be noted that our package computes residual errors with the Froebonius norm and that
the package LowRankApprox uses the spectral norm.
