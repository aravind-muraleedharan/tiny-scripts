# seamless m4T- code for audio splitting and transcription

import torch
import torchaudio
from seamless_communication.models.inference import Translator
from pydub import AudioSegment
import wave
import math
import os

audio_name = "audio.wav"
target_lang = "mal"

# function to calculate the duration of the input audio clip
def get_duration_wave(file_path):
   with wave.open(file_path, 'r') as audio_file:
      frame_rate = audio_file.getframerate()
      n_frames = audio_file.getnframes()
      duration = n_frames / float(frame_rate)
      return duration


duration = get_duration_wave(audio_name)
print(f"Duration: {duration:.2f} seconds")

resample_rate = 16000
t1 = 0
t2 = 20000

# Generating 'n' number of audio samples each with 20seconds duration. This is to avoid issue with the maximum sequence length
num_samples = math.ceil(duration/20)
print("number of samples ", num_samples)

# Initializing the translator model
translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:0"), torch.float16)

for i in range(num_samples):
    newAudio = AudioSegment.from_wav(audio_name)
    newAudio = newAudio[t1:t2]
    new_audio_name = "new_" + str(t1) + ".wav"
    newAudio.export(new_audio_name, format="wav")
    waveform, sample_rate = torchaudio.load(new_audio_name)
    resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    torchaudio.save("resampled.wav", resampled_waveform, resample_rate)

    translated_text, _, _ = translator.predict("resampled.wav", "s2tt", target_lang)
    print(translated_text)
    t1 = t2
    t2 += 20000
    os.remove(new_audio_name)
os.remove("resampled.wav")


