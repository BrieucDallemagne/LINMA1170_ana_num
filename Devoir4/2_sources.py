import numpy as np
import librosa
import librosa.display
import soundfile as sf
from sklearn.decomposition import FastICA

def cocktail_party(data):

    U, s, Vt = np.linalg.svd(data, full_matrices=False)

    k = min(data.shape)
    S = np.diag(s[:k])
    U = U[:, :k]
    Vt = Vt[:k, :]

    sources = np.dot(U, np.dot(S, Vt))
    
    return sources

#filename1 = "BC1.mp3"
#filename2 = "BC2.wav"

#y1, sr1 = librosa.load(filename1)
#y2, sr2 = librosa.load(filename2)

s1 = [[0.11,0.4,0.51,1.3] ,[0.92,0.23,0.6,0.2]]

s1 = np.array(s1)

A = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        A[i][j] = np.random.uniform(0, 1)

X = s1 @ A
print(X)
"""
len1 = len(y1)
len2 = len(y2)
if len1 > len2:
    y1 = y1[:len2]
else:
    y2 = y2[:len1]

y = np.vstack((y1, y2)).T
"""

sources = cocktail_party(X)

print(sources)

print(cocktail_party(np.identity(4) ))
# Sauvegarde des sources séparées
#sf.write('outputs/source1.wav', sources[:, 0], sr1)
#sf.write('outputs/source2.wav', sources[:, 1], sr2)







