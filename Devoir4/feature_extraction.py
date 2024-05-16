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

S_feature_extracted = np.zeros_like(S)
start, end = 10, 1024  # Example
S_feature_extracted[start:end] = S[start:end]

magnitude_feature_extracted = np.dot(U, np.dot(np.diag(S_feature_extracted), Vt))

D_feature_extracted = magnitude_feature_extracted * np.exp(1j * np.angle(D))

y_feature_extracted = librosa.istft(D_feature_extracted, hop_length=hop_length)

sf.write('outputs/feature_extracted_audio.wav', y_feature_extracted, sr)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(magnitude, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Original magnitude spectrogram')
plt.subplot(2, 1, 2)
plt.imshow(magnitude_feature_extracted, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()

plt.title('Feature extracted magnitude spectrogram')

plt.tight_layout()
#plt.show()
plt.savefig('img/feature_extraction.pdf')

