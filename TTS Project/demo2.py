import parselmouth
import numpy as np
from scipy.optimize import minimize
from fastdtw import fastdtw
import librosa
import soundfile as sf
import tempfile
import os

# === Function to change gender using Praat ===
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
    assert pitch_min <= pitch_max, "pitch_min should be less than or equal to pitch_max"
    assert duration_factor <= 3.0, "duration_factor cannot be larger than 3.0"

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


# === Load audio ===
if not os.path.exists("tts.wav") or not os.path.exists("human.wav"):
    raise FileNotFoundError("Missing tts.wav or human.wav")

tts_sound = parselmouth.Sound("tts.wav")
real_sound = parselmouth.Sound("human.wav")


# === Helper: extract MFCCs ===
def get_mfcc(sound):
    samples = sound.values[0]
    sr = int(sound.sampling_frequency)
    if len(samples) < 2048:
        samples = np.pad(samples, (0, 2048 - len(samples)))
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
    return mfcc.T  # (time, features)


# === Cost function for optimization ===
def objective(params):
    fsr, new_pitch, pitch_factor, duration_factor = params
    pitch_min = 75
    pitch_max = 600

    if fsr <= 0 or new_pitch <= 0 or pitch_factor < 0 or not (0.5 <= duration_factor <= 3.0):
        return np.inf

    try:
        # Convert original TTS audio to np.ndarray
        tts_array = tts_sound.values.T.astype(np.float32)
        sr = int(tts_sound.sampling_frequency)

        transformed_audio = change_gender(
            tts_array, sr, pitch_min, pitch_max,
            fsr, new_pitch, pitch_factor, duration_factor
        )
        transformed_sound = parselmouth.Sound(transformed_audio.flatten(), sr)

        t_mfcc = get_mfcc(transformed_sound)
        r_mfcc = get_mfcc(real_sound)
        distance, _ = fastdtw(t_mfcc, r_mfcc)
        print(f"Params: {params} -> DTW distance: {distance}")
        return distance

    except Exception as e:
        print(f"Error with params {params}: {e}")
        return np.inf


# === Optimization setup ===
init_params = [1.0, 150.0, 1.0, 1.0]
bounds = [(0.5, 2.0), (50.0, 400.0), (0.0, 2.0), (0.5, 3.0)]

print("Starting optimization...\n")
result = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B', options={'maxiter': 30})

print("\n--- Optimization Complete ---")
print("Best parameters:", result.x)
print("Minimum DTW Distance:", result.fun)


# === Apply best transformation and save ===
best_params = result.x
tts_array = tts_sound.values.T.astype(np.float32)
sr = int(tts_sound.sampling_frequency)

final_audio = change_gender(
    tts_array, sr, pitch_min=75, pitch_max=600,
    formant_shift_ratio=best_params[0],
    new_pitch_median=best_params[1],
    pitch_range_factor=best_params[2],
    duration_factor=best_params[3]
)

sf.write("transformed_tts.wav", final_audio, sr)
print("Saved transformed voice as 'transformed_tts.wav'")
