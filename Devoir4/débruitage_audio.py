#################################################
# débruitage_audio.py
# Authors : Charles Van Hees and Brieuc Dallemagne
#
# This script compresses an audio signal using the SVD method.
#################################################

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import argparse


def denoise_signal(filename, output_file, k=70) :
    """
    ARGS:
    filename : str, name of the file with the signal
    output_file : str, name of the file with the denoised signal
    k : int, number of singular values to keep

    RETURNS:
    None
    """
    y, sr = librosa.load(filename)
    D = librosa.stft(y)
    magnitude = np.abs(D)
    U, S, Vh = np.linalg.svd(magnitude, full_matrices=False)
    # Reconstruction du signal avec les k premières composantes principales
    new_magnitude = U[:,:k] @ np.diag(S[:k]) @ Vh[:k,:]
    D_reconstructed = new_magnitude * np.exp(np.angle(D)*1j)
    y_reconstructed = librosa.istft(D_reconstructed)
    sf.write(output_file, y_reconstructed, sr)

if __name__ == '__main__' :
    #parse arguments and call denoise_signal
    parser = argparse.ArgumentParser(description="Débruitage de signaux audio")
    parser.add_argument('-f', '--file', type=str, help="Nom du fichier avec le signal")
    parser.add_argument('-o', '--output_file', type=str, help="Nom du fichier du signal débruité")
    parser.add_argument('-k', '--components', type=int, help="Nombre de composantes principales à conserver", default=70)
    args = parser.parse_args()
    denoise_signal(args.file, args.output_file, args.components)