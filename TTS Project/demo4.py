import parselmouth
import numpy as np
from scipy.optimize import minimize
from fastdtw import fastdtw
import soundfile as sf
import tempfile
import os
import matplotlib.pyplot as plt

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


# === Helper: extract pitch track (F0) ===
def get_pitch_track(sound, time_step=0.01):
    pitch = sound.to_pitch(time_step)
    pitch_values = pitch.selected_array['frequency']  # in Hz
    pitch_values[pitch_values == 0] = np.nan  # Replace unvoiced with NaN
    return pitch_values


# === Cost function for optimization ===
def objective(params):
    fsr, new_pitch, pitch_factor, duration_factor = params
    pitch_min = 75
    pitch_max = 600

    if fsr <= 0 or new_pitch <= 0 or pitch_factor < 0 or not (0.5 <= duration_factor <= 3.0):
        return np.inf

    try:
        tts_array = tts_sound.values.T.astype(np.float32)
        sr = int(tts_sound.sampling_frequency)

        transformed_audio = change_gender(
            tts_array, sr, pitch_min, pitch_max,
            fsr, new_pitch, pitch_factor, duration_factor
        )
        transformed_sound = parselmouth.Sound(transformed_audio.flatten(), sr)

        # Extract pitch tracks
        t_pitch = get_pitch_track(transformed_sound)
        r_pitch = get_pitch_track(real_sound)

        # Replace NaNs with 0 for DTW
        t_pitch_clean = np.nan_to_num(t_pitch, nan=0.0)
        r_pitch_clean = np.nan_to_num(r_pitch, nan=0.0)

        distance, _ = fastdtw(
            t_pitch_clean.reshape(-1, 1),
            r_pitch_clean.reshape(-1, 1)
        )

        print(f"Params: {params} -> DTW F0 distance: {distance}")
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
print("Minimum DTW F0 Distance:", result.fun)


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


# === Plot F0 pitch tracks of original TTS, transformed TTS, and real voice ===
def plot_pitch_tracks(original_sound, transformed_sound, reference_sound):
    t_pitch_orig = get_pitch_track(original_sound)
    t_pitch_trans = get_pitch_track(transformed_sound)
    r_pitch = get_pitch_track(reference_sound)

    max_len = max(len(t_pitch_orig), len(t_pitch_trans), len(r_pitch))
    t_pitch_orig = np.pad(t_pitch_orig, (0, max_len - len(t_pitch_orig)), constant_values=np.nan)
    t_pitch_trans = np.pad(t_pitch_trans, (0, max_len - len(t_pitch_trans)), constant_values=np.nan)
    r_pitch = np.pad(r_pitch, (0, max_len - len(r_pitch)), constant_values=np.nan)

    time = np.arange(max_len) * 0.01  # assuming time_step = 0.01s

    plt.figure(figsize=(12, 6))
    plt.plot(time, r_pitch, label="Real Voice", color='green', linewidth=2)
    plt.plot(time, t_pitch_orig, label="Original TTS", color='blue', linestyle='dashed')
    plt.plot(time, t_pitch_trans, label="Transformed TTS", color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Fundamental Frequency (Hz)")
    plt.title("Pitch Tracks Comparison (F0)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pitch_comparison_plot.png")
    plt.show()

# Create parselmouth.Sound object from final audio
transformed_sound = parselmouth.Sound(final_audio.flatten(), sr)

# Plot pitch tracks
plot_pitch_tracks(tts_sound, transformed_sound, real_sound)
