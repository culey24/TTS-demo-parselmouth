import parselmouth
import numpy as np
import soundfile as sf
import tempfile
import os
from dtaidistance import dtw

# === Change gender function ===
def change_gender(
    input: np.ndarray,
    sampling_rate: int,
    pitch_min: float,
    pitch_max: float,
    formant_shift_ratio: float,
    new_pitch_median: float = 0,
    pitch_range_factor: float = 1,
    duration_factor: float = 1,
) -> np.ndarray:
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_file.name, input, sampling_rate)

    sound = parselmouth.Sound(tmp_file.name)

    tuned_sound = parselmouth.praat.call(
        sound,
        "Change gender",
        pitch_min,
        pitch_max,
        formant_shift_ratio,
        new_pitch_median,
        pitch_range_factor,
        duration_factor,
    )

    tmp_file.close()
    return np.array(tuned_sound.values.T)

# === Extract pitch (F0) track ===
def get_pitch_track(sound, time_step=0.01):
    pitch = sound.to_pitch(time_step)
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    return np.nan_to_num(pitch_values, nan=0.0)

# === DTW F0 distance between 2 sounds ===
def calculate_dtw_distance(sound1, sound2):
    f0_1 = get_pitch_track(sound1)
    f0_2 = get_pitch_track(sound2)
    distance = dtw.distance(f0_1, f0_2)
    return distance

# === Load audios ===
if not os.path.exists("tts.wav") or not os.path.exists("human.wav"):
    raise FileNotFoundError("Missing tts.wav or human.wav")

tts_sound = parselmouth.Sound("tts.wav")
human_sound = parselmouth.Sound("human.wav")

tts_array = tts_sound.values.T.astype(np.float32)
sr = int(tts_sound.sampling_frequency)

# === Random Search ===
np.random.seed(42)
pitch_min, pitch_max = 75, 600

best_distance = float('inf')
best_params = None
best_audio = None
N_TRIALS = 50

print("Starting random search...\n")

for i in range(N_TRIALS):
    fsr = np.random.uniform(0.6, 1.8)
    new_pitch = np.random.uniform(70, 300)
    pitch_factor = np.random.uniform(0.5, 1.5)
    duration_factor = np.random.uniform(0.8, 1.2)

    try:
        transformed = change_gender(
            tts_array, sr, pitch_min, pitch_max,
            fsr, new_pitch, pitch_factor, duration_factor
        )
        transformed_sound = parselmouth.Sound(transformed.flatten(), sr)
        distance = calculate_dtw_distance(transformed_sound, human_sound)

        print(f"[{i+1:02d}] DTW F0 Distance: {distance:.2f} | Params: fsr={fsr:.2f}, new_pitch={new_pitch:.1f}, pitch_factor={pitch_factor:.2f}, duration_factor={duration_factor:.2f}")

        if distance < best_distance:
            best_distance = distance
            best_params = (fsr, new_pitch, pitch_factor, duration_factor)
            best_audio = transformed

    except Exception as e:
        print(f"[{i+1:02d}] Error: {e}")

# === Save best transformed audio ===
if best_audio is not None:
    sf.write("best_transformed_tts.wav", best_audio, sr)
    print("\nBest audio saved as 'best_transformed_tts.wav'")
    print("Best parameters:")
    print(f"  Formant Shift Ratio:   {best_params[0]:.2f}")
    print(f"  New Pitch Median:      {best_params[1]:.1f}")
    print(f"  Pitch Range Factor:    {best_params[2]:.2f}")
    print(f"  Duration Factor:       {best_params[3]:.2f}")
    print(f"  DTW F0 Distance:       {best_distance:.2f}")
else:
    print("No valid transformation found.")
