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

k = 10 # Example
U_reduced = U[:, :k]
S_reduced = S[:k]
Vt_reduced = Vt[:k, :]

magnitude_reduced = np.dot(U_reduced, np.dot(np.diag(S_reduced), Vt_reduced))

D_reduced = magnitude_reduced * np.exp(1j * np.angle(D))

y_reduced = librosa.istft(D_reduced, hop_length=hop_length)

sf.write('outputs/reduced_dimensionality_audio.wav', y_reduced, sr)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(magnitude, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Original magnitude spectrogram')
plt.subplot(2, 1, 2)
plt.imshow(magnitude_reduced, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Reduced dimensionality magnitude spectrogram')

plt.tight_layout()
#plt.show()
plt.savefig('img/reduced_dimensionality.pdf')
