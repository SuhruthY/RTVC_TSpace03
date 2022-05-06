import inference as encoder
from preprocess import *

arr = librosa.load("../data/84-121123-0000.flac")[0]
print(arr.shape)

arr = preprocess_wav(arr)
print(arr.shape)

arr = wav_to_mel_spectrogram(arr)
print(arr.shape)

encoder = encoder.load_model("encoder.pt")
print(encoder)

