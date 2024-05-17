#################################################
# comparaison_audio.py
# Authors : Charles Van Hees and Brieuc Dallemagne
#
# This script compares the compression of an audio signal using the SVD method.
#################################################

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# import audio
filename = 'input/a.mp3'
y, sr = librosa.load(filename)

# compute STFT
n_fft = 2048
hop_length = 512
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
magnitude = np.abs(D)
phase = np.angle(D)

# apply SVD
U, S, Vt = np.linalg.svd(magnitude, full_matrices=False)

k = 10  # Par exemple, pour la musique et les paroles, vous pourriez tester plusieurs valeurs

# Reconstruction du signal avec les k premières composantes principales
S1 = np.zeros_like(S)
S1[:k] = S[:k]
magnitude_music = np.dot(U, np.dot(np.diag(S1), Vt))

# Reconstruction du signal avec les k dernières composantes principales
S2 = np.zeros_like(S)
S2[k:] = S[k:]
magnitude_vocals = np.dot(U, np.dot(np.diag(S2), Vt))

D_music = magnitude_music * np.exp(1j * phase)
D_vocals = magnitude_vocals * np.exp(1j * phase)

# Reconstruction des signaux audio grâce à la STFT inverses
y_music = librosa.istft(D_music, hop_length=hop_length)
y_vocals = librosa.istft(D_vocals, hop_length=hop_length)

output_music = 'outputs/10th.wav'
output_vocals = 'outputs/rest.wav'
#sf.write(output_music, y_music, sr)
#sf.write(output_vocals, y_vocals, sr)

# plot spectrograms
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.imshow(magnitude, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('Original magnitude spectrogram')
plt.subplot(3, 1, 2)
plt.imshow(magnitude_music, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('10 first singular values')
plt.subplot(3, 1, 3)
plt.imshow(magnitude_vocals, aspect='auto', origin='lower', vmin=0, vmax=20)
plt.colorbar()
plt.title('rest of singular values')
plt.tight_layout()
#plt.show()
plt.savefig('img/comparaison_audio.pdf')
