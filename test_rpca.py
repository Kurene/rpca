import numpy as np
import librosa
import sys
from fast_griffin_lim import fast_griffin_lim
from plotspec import plotspec
import matplotlib.pyplot as plt

from rpca import rpca 

    
if __name__ == "__main__":
    path = sys.argv[1]
    lam = float(sys.argv[2])
    sc = 1.0
    
    mono = True
    sr = 16000
    n_fft = 2048
    hop_length = n_fft//4
    window = "hann"
    offset = 0
    duration = 55
    

    y, sr = librosa.core.load(path, sr=sr, offset=offset, duration=duration, mono=mono)
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    X_abs, X_phase = np.abs(X), np.angle(X)
    X_gain = np.max(X_abs)
    X_abs /= X_gain

    L, S = rpca(X_abs**sc, lam)
    L, S = L**(1.0/sc), S**(1.0/sc)
    plotspec(X_abs, L, S, context="plotspec.png")
    
    # Time-frequency masking (wiener filter)
    librosa.output.write_wav('__Source.wav', y, sr)
    WF_S, WF_L = (S**2)/(L**2 + S**2), (L**2)/(L**2 + S**2)
    y_est = fast_griffin_lim(X_gain*WF_S*X_abs, X_phase, n_fft=n_fft, hop_length=hop_length, window=window)
    librosa.output.write_wav('__Sparse.wav', y_est, sr)
    #y_est = fast_griffin_lim(X_gain*WF_L*X_abs, X_phase, n_fft=n_fft, hop_length=hop_length, window=window)
    #librosa.output.write_wav('__LowRank.wav', y_est, sr)     