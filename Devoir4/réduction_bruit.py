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

threshold = 0.1 * np.max(S) # Example
S_denoised = np.where(S > threshold, S, 0)

magnitude_denoised = np.dot(U, np.dot(np.diag(S_denoised), Vt))

D_denoised = magnitude_denoised * np.exp(1j * np.angle(D))

y_denoised = librosa.istft(D_denoised, hop_length=hop_length)

sf.write('outputs/denoised_audio.wav', y_denoised, sr)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(magnitude, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Original magnitude spectrogram')
plt.subplot(2, 1, 2)
plt.imshow(magnitude_denoised, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Denoised magnitude spectrogram')

plt.tight_layout()
#plt.show()
plt.savefig('img/denoised.pdf')
