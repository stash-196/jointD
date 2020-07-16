# Disclaimer: This code is a python implementation of the extended Jacobi
# technique for simultaneous diagonalization in the jadeR(CM, m) function from 
# https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/23ed4823-a05f-47da-99e9-4e84f74c955d/c77c8388-dd0e-4c1a-9067-edfa8989fef4/previews/iPPG_dataset_and_Matlab_package/jadeR.m/index.html
# by stash-196
# The technique is discussed in "Cardoso and Souloumiac, 1996, Jacobi Angles for Simultaneous Diagonalization" 
# at https://epubs.siam.org/doi/abs/10.1137/S0895479893259546

import numpy as np
import math

verbose = 1
def jointD(Matrices):

    CM = np.hstack(tuple(Matrices))

    m, n = CM.shape
    nbcm = int(n/m)

    if 0:
    ## Init by diagonalizing a *single* cumulant matrix.  It seems to save
    ## some computation time `sometimes'.  Not clear if initialization is really worth
    ## it since Jacobi rotations are very efficient.  On the other hand, it does not
    ## cost much...
        if verbose:
            print('Initialization of the diagonalization')
            D, V = np.linalg.eig(CM[:, list(range(m))])
            for u in range(0, m*nbcm, m):
                CM[:, u:u+m] = CM[:, u:u+m] @ V
            CM = V.T @ CM

    else: ##The dont-try-to-be-smart init //lol
        V = np.eye(m)

    ## Computing the initial value of the contrast
    Diag = np.zeros(m)
    On = 0
    Range = np.array(range(m))

    for im in range(nbcm):
        Diag =  np.diag(CM[:, Range])
        On = On + np.sum(Diag * Diag)
        Range += m

    Off = np.sum(CM * CM) - On

    seuil = 1e-7 # threshold on small angles. //Should be scaled... but idk
    encore = 1
    sweep = 0 # sweep number
    updates = 0 # Total number of rotations
    upds = 0 # Number -f rotations in a given sweep
    g = np.zeros((2, nbcm))
    gg = np.zeros((2, 2))
    G = np.zeros((2, 2))
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    ## Joint diagonalization proper
    if verbose: print('Contrast optimization by joint diagonalization\n')

    while encore:
        encore = 0
        if verbose: print('Sweep', sweep)
        sweep += 1
        upds = 0
        Vkeep = V

        for p in range(m-1):
            for q in range(p+1, m):
                Ip = list(range(p, m*nbcm, m))
                Iq = list(range(q, m*nbcm, m))

                # Computation of Givens angle
                g = np.array([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                # print('g = \n', g)
                gg = g @ g.T
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * math.atan2(toff, ton + np.sqrt(ton * ton + toff * toff))
                Gain = (np.sqrt(ton * ton + toff * toff) - ton) / 4

                if abs(theta) > seuil:
                    encore = 1
                    upds += 1
                    c = math.cos(theta)
                    s = math.sin(theta)
                    G = np.array([[c, -s], [s, c]])

                    pair = np.array([p, q]).T
                    V[:, pair] = V[:, pair] @ G
                    CM[pair, :] = G.T @ CM[pair, :]
                    CM[:, Ip + Iq] = np.hstack([c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq]])

                    On = On + Gain
                    Off = Off - Gain

                    # print('  {}, {}, {}'.format(p, q, Off/On))


        if verbose: print('  completed in {} rotations'.format(upds))
        updates += upds
    #returns a list of diagonal matrices and eigen vector matrix
    return list(np.array_split(CM, nbcm, 1)), V


# % Copyright (c) 2013, Jean-Francois Cardoso
# % All rights reserved.
# %
# %
# % BSD-like license.
# % Redistribution and use in source and binary forms, with or without modification, 
# % are permitted provided that the following conditions are met:
# %
# % Redistributions of source code must retain the above copyright notice, 
# % this list of conditions and the following disclaimer.
# %
# % Redistributions in binary form must reproduce the above copyright notice,
# % this list of conditions and the following disclaimer in the documentation 
# % and/or other materials provided with the distribution.
# %
# %
# % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
# % OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# % AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER 
# % OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# % DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
# % DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
# % IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
# % OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



if __name__ == "__main__":
    # Checking to see if it works with Random Symmetric Commutative Matrices
    # n = 5
    # A = np.random.rand(n, n)
    # A = (A + A.T)/2
    # w, v = np.linalg.eig(A)
    # B = np.random.rand(n)
    # B = np.diag(B)
    # B =  v @ B @ v.T

    # Or explicitly pick them
    A = np.array([
        [-0.1332,   -0.5041,    0.8295,   -0.7784,   -0.0687],
        [-0.5041,   -0.8479,   -1.1886,    0.0981,    0.7626],
        [0.8295,   -1.1886,   -0.8655,   -0.8128,    0.4876],
        [-0.7784,    0.0981,   -0.8128,    0.3335,   -0.0424],
        [-0.0687,    0.7626,    0.4876,   -0.0424,    0.8620],
    ])

    B = np.array([
        [0.4531,    0.1442,    0.0488,   -0.1342,   -0.0728],
        [0.1442,   -0.6669,   -0.6133,    0.3468,    0.2286],
        [0.0488,   -0.6133,   -0.6185,   -0.5847,    0.1302],
        [-0.1342,    0.3468,   -0.5847,   -0.2836,   -0.0181],
        [-0.0728,    0.2286,    0.1302,   -0.0181,   -0.4216],
    ])

    # Input list or tuple
    newDs, newV = jointD((A, B))

    # Test reproducibility
    newA = newV @ newDs[0] @ newV.T
    newB = newV @ newDs[1] @ newV.T
    print(newA)
    print(newB)

