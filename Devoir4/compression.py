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

k = 50 # Example
S_compressed = np.zeros_like(S)
S_compressed[:k] = S[:k]

magnitude_compressed = np.dot(U, np.dot(np.diag(S_compressed), Vt))

D_compressed = magnitude_compressed * np.exp(1j * np.angle(D))

y_compressed = librosa.istft(D_compressed, hop_length=hop_length)

sf.write('outputs/compressed_audio.wav', y_compressed, sr)
