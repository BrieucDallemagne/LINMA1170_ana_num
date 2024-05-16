import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

filename = 'elton.wav'
y, sr = librosa.load(filename)

n_fft = 2048
hop_length = 512
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

magnitude = np.abs(D)
U, S, Vt = np.linalg.svd(magnitude, full_matrices=False)

S_feature_extracted = np.zeros_like(S)
start, end = 10, 20  # Example
S_feature_extracted[start:end] = S[start:end]

magnitude_feature_extracted = np.dot(U, np.dot(np.diag(S_feature_extracted), Vt))

D_feature_extracted = magnitude_feature_extracted * np.exp(1j * np.angle(D))

y_feature_extracted = librosa.istft(D_feature_extracted, hop_length=hop_length)

sf.write('feature_extracted_audio.wav', y_feature_extracted, sr)
