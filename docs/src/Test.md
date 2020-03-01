# Tests

To run test, **in the julia REPL** go to the test directory, and run :
**include("runtests.jl")**.  
In fact to avoid compilation times, it is best to execute twice the inclusion of the *runtests.jl* file.
The logger is by default (during tests) in Debug configuration and will output time,
and intermediary convergence results (See file runtests.jl)

All the tests uses randomly generated matrices, moreover the algorithms
use random sampling so the cpu times needed, and precisions obtained can fluctuate but the order of magnitude should be consistent with the results announced for a laptop with 2 cores i7@2.70GHz , 32 Gb of Memory and
Julia running with 8 threads.

## ifsvd\_versus\_svd

This compares LowRankApproxMc with exact svd,
computing the L2 norm of the difference of singular values.

## ifsvd_lowRankApproxEpsil

This test runs on a matrix of size (10000, 20000).
It has the structure encountered in hyperspectral images where a column is a pixel, and rows represent
a canal (a mass in mass spectrometry imaging). The matrix has some blocks of related pixels 
which have expressions in some common rows.  
The iterations of incremental svd with sampling size 100, we can reach in 14s a 5% relative error with an adjusted rank=391,
or depending on random data, in 18s to reach a 3.8% relative error and an adjusted rank= 495

## vrpca_epsil

 This tests asks for an approx at 0.01.
 It extracts a rank 100 matrix at 0.005 precision within 1.8s.
 With LowRankApprox.prange we need to go up to rank=500 (initialized with
 *opts = LowRankApprox.LRAOptions(rank=500, rtol = 0.0001)* to achieve a relative Froebonius error of 0.005.
 It then runs in 1.15s.

So we see that LowRankApprox is really fast but needs many more singular vectors
to achieve the same relative error, so that VrPCA gives a clear reduction of relative error.
Of course VrPCA initialized with LowRankApprox give an analogous reduction
in relative error as in the following test.

## vrpca_withLowRankApprox

This test runs on the same data as vrpca_epsil. It initialize VrPCA
with *LRAOptions(rank=100, rtol=0.001)* and does a svd of order 100. This pass
runs in 0.7s and gives a relative error of 0.02.
The VrPCA iterations run in 3.2s and take us to a relative error of 0.0049.
Once again LowRankApprox is really fast and VrPCA gives a division by 4 of the relative error.
