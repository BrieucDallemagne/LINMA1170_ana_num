import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

filename = 'a.wav'
y, sr = librosa.load(filename)

n_fft = 2048
hop_length = 512
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

magnitude = np.abs(D)
U, S, Vt = np.linalg.svd(magnitude, full_matrices=False)

magnitude_reconstructed = np.dot(U, np.dot(np.diag(S), Vt))

D_reconstructed = magnitude_reconstructed * np.exp(1j * np.angle(D))

y_reconstructed = librosa.istft(D_reconstructed, hop_length=hop_length)

output_filename = 'outputs/normal.wav'
sf.write(output_filename, y_reconstructed, sr)
