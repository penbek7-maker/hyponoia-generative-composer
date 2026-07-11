import json
import soundfile as sf
import numpy as np
import librosa

REFERENCE = "output/alpha_memory_drone.wav"

audio, sr = sf.read(REFERENCE)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

audio = audio.astype(np.float32)

rms = librosa.feature.rms(y=audio)[0]
centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
flatness = librosa.feature.spectral_flatness(y=audio)[0]
onset = librosa.onset.onset_strength(y=audio, sr=sr)

profile = {
    "energy": float(np.mean(rms)),
    "brightness": float(np.mean(centroid)),
    "noise": float(np.mean(flatness)),
    "attack": float(np.mean(onset))
}

with open("alpha_profile.json", "w") as f:
    json.dump(profile, f, indent=4)

print("Created alpha_profile.json")
print(profile)
