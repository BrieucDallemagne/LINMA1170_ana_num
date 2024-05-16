import numpy as np
import scipy as sp
from pydub import AudioSegment
 
def load_mp3(file_path,sample_rate=40):
 
    audio = AudioSegment.from_mp3(file_path)
 
    audio = audio.set_frame_rate(sample_rate)
 
    audio_array = np.array(audio.get_array_of_samples())
 
   
    audio_array = audio_array / np.max(np.abs(audio_array))
 
    return audio_array
 
 
 
 
def save_wav(file_path, data, sample_rate):
 
    sp.io.wavfile.write(file_path, sample_rate, data.astype(np.int16))
 
def blind_source_separation(audio_data):
 
    X = np.array(audio_data, dtype=float)
 
    X = np.outer(X, X)
   
 
    U, s, V = sp.linalg.svd(X, full_matrices=False)
 
    num_sources = 2  # Nombre de sources à séparer
    sources = U[:, :num_sources] @ np.diag(s[:num_sources])
 
    return sources
 
input_mp3_file = "son_MP3/a.wav"
 
audio_data = load_mp3(input_mp3_file)
 
separated_sources = blind_source_separation(audio_data)
 
for i, source in enumerate(separated_sources.T):
    output_wav_file = f"source_{i+1}.wav"
    save_wav(output_wav_file, source, 44100)