import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft
from sklearn.decomposition import FastICA

filename1 = "BC1.mp3"
filename2 = "BC2.mp3"
y1, sr1 = librosa.load(filename1)
mean1 = np.mean(y1)
y1 -= mean1
y2, sr2 = librosa.load(filename2)
mean2 = np.mean(y2)
y2 -= mean2

if len(y1) > len(y2) : y1 = y1[:len(y2)]
else : y2 = y2[:len(y1)]

X = np.vstack((y1, y2))
U, S, V = np.linalg.svd(X, full_matrices=False)

print(np.shape(V))

sf.write("test.mp3", V[0,:], sr1)