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

S[10:] = 0  # example 

magnitude_reconstructed = np.dot(U, np.dot(np.diag(S), Vt))

D_reconstructed = magnitude_reconstructed * np.exp(1j * np.angle(D))

y_reconstructed = librosa.istft(D_reconstructed, hop_length=hop_length)

output_filename = 'outputs/reconstructed_audio.wav'
sf.write(output_filename, y_reconstructed, sr)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(magnitude, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Original magnitude spectrogram')
plt.subplot(2, 1, 2)
plt.imshow(magnitude_reconstructed, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Reconstructed magnitude spectrogram')

plt.tight_layout()
#plt.show()
plt.savefig('img/annulation.pdf')
