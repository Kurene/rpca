import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def plotspec(X, L, S, y_axis='log', context=""):
    plt.clf()
    plt.subplot(2,2,1)
    librosa.display.specshow(librosa.amplitude_to_db(X, ref=np.max),
                             y_axis=y_axis, x_axis='time', cmap='jet')

    plt.title('X')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,2,2)
    librosa.display.specshow(librosa.amplitude_to_db(L, ref=np.max),
                             y_axis=y_axis, x_axis='time', cmap='jet')
    plt.title('Low-rank')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,2,4)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis=y_axis, x_axis='time', cmap='jet')
    plt.title('Sparsity-matrix')
    plt.colorbar(format='%+2.0f dB')
        
    plt.tight_layout()
    if context == "":
        plt.show()
    else:
        plt.savefig('plotspec_'+context+'.png')
