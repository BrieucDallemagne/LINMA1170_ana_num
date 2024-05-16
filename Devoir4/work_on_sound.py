import numpy as np
import scipy as sp
from audio2numpy import open_audio
 
def load_mp3(file_path):
    signal, sampling_rate = open_audio(file_path)
    return signal,sampling_rate
 
def save_wav(file_path, data, sample_rate):
 
    sp.io.wavfile.write(file_path, sample_rate, data.astype(np.float32))
 
def blind_source_separation(audio_data):
 
    X = np.array(audio_data, dtype=float)
 
    U, s, V = sp.linalg.svd(X, full_matrices=False)
 
    num_sources = 2  # Nombre de sources à séparer
    sources = U[:, :num_sources] @ np.diag(s[:num_sources])
 
    return sources
 
input_mp3_file = "elton.wav"
 
audio_data, sampling_rate = load_mp3(input_mp3_file)
print(audio_data, sampling_rate)
 
separated_sources = blind_source_separation(audio_data)
 
for i, source in enumerate(separated_sources.T):
    output_wav_file = f"source_{i+1}.wav"
    save_wav(output_wav_file, source, sampling_rate)