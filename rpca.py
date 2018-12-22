import numpy as np
import librosa
import sys
from fast_griffin_lim import fast_griffin_lim
from plotspec import plotspec
import matplotlib.pyplot as plt


def rpca(X, lam, n_iter=100, tolerance=1e-9):
    def floor(_X):
        _X[_X < 0.0] = 0.0
        return _X

    def shrink(_X, eps):
        _X[np.abs(_X) <= eps] = 0.0
        _X[_X > eps] -= eps
        _X[_X < -eps] += eps
        return _X
        
    X_L2 = np.linalg.norm(X)
    invlam_X_Linf = np.max(np.abs(X)) / lam 
    Y = X / np.maximum(X_L2, invlam_X_Linf)
    S, L = np.zeros(X.shape), np.zeros(X.shape)
    mu, rho = 1.25 / X_L2, 1.5
    
    print("Iter\tRank\tError")
    for iter in range(0, n_iter):
        U, s, VH = np.linalg.svd((X - S + Y/mu), full_matrices=False)
        L = floor(np.dot(U * shrink(s, 1.0/mu), VH))
        S = floor(shrink(X - L + Y/mu, lam/mu))
        Y += mu*(X - L - S)
        mu *= rho
        err = np.abs(X - L - S).sum()
        rank = np.linalg.matrix_rank(S)
        print(f"{iter:04}:\t{rank}\t{err}")
        if err < tolerance:
            break
    return L, S
