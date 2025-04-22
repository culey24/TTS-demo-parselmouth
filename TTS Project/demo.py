import parselmouth
import numpy as np
from scipy.optimize import minimize
from fastdtw import fastdtw
import librosa

import tempfile

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call


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
    """
    Changes the gender of the input audio using Praat's 'Change gender' algorithm.

    Args:
        input (np.ndarray): The input audio data as a NumPy array.
        sampling_rate (int): The sampling rate of the input audio.
        pitch_min (float): Minimum pitch (Hz) below which pitch candidates will not be considered.
        pitch_max (float): Maximum pitch (Hz) above which pitch candidates will be ignored.
        formant_shift_ratio (float): Ratio determining the frequencies of formants in the newly created audio.
            A ratio of 1.0 indicates no frequency shift, while 1.1 approximates female formant characteristics.
            A ratio of 1/1.1 approximates male formant characteristics.
        new_pitch_median (float): Median pitch (Hz) of the new audio. The pitch values in the new audio
            are calculated by multiplying them by new_pitch_median / old_pitch_median.
            Default: 0.0 (same as original).
        pitch_range_factor (float): Scaling factor for the new pitch values around the new pitch median.
            A factor of 1.0 implies no additional pitch modification (except for the median adjustment).
            A factor of 0.0 monotonizes the new sound to the new pitch median.
            Default: 1.0.
        duration_factor (float): Factor by which the sound will be lengthened.
            Values less than 1.0 result in a shorter sound, while values larger than 3.0 are not supported.
            Default: 1.0.

    Returns:
        np.ndarray: The processed audio data as a NumPy array with the gender changed.

    Raises:
        AssertionError: If pitch_min is greater than pitch_max or if duration_factor is larger than 3.0.
    """
    assert pitch_min <= pitch_max, "pitch_min should be less than or equal to pitch_max"
    assert duration_factor <= 3.0, "duration_factor cannot be larger than 3.0"

    # Save the input audio to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_file, input, sampling_rate)

    # Load the source audio
    sound = parselmouth.Sound(tmp_file.name)

    # Tune the audio
    tuned_sound = call(
        sound,
        "Change gender",
        pitch_min,
        pitch_max,
        formant_shift_ratio,
        new_pitch_median,
        pitch_range_factor,
        duration_factor,
    )

    # Remove the temporary file
    tmp_file.close()

    return np.array(tuned_sound.values.T)

# === Load your TTS and real human audio ===
tts_sound = parselmouth.Sound("tts.wav")
real_sound = parselmouth.Sound("human.wav")

# === Helper: apply "Change gender" using Praat ===
def apply_voice_transform(sound, formant_shift_ratio, new_pitch_median, pitch_range_factor, duration_factor):
    try:
        return parselmouth.praat.call(
            sound, "Change gender", 
            formant_shift_ratio, 
            new_pitch_median, 
            pitch_range_factor, 
            duration_factor
        )
    except Exception as e:
        print(f"Transformation failed: {e}")
        return None

# === Helper: extract MFCCs for DTW comparison ===
def get_mfcc(sound):
    samples = sound.values[0]
    sr = int(sound.sampling_frequency)
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
    return mfcc.T  # Shape: (time, features)

# === Cost function for optimization ===
def objective(params):
    fsr, new_pitch, pitch_factor, duration_factor = params

    if fsr <= 0 or new_pitch <= 0 or pitch_factor < 0 or not (0.5 <= duration_factor <= 3.0):
        return np.inf  # Invalid params

    try:
        transformed = apply_voice_transform(
            tts_sound, fsr, new_pitch, pitch_factor, duration_factor
        )
        if transformed is None:
            return np.inf
        t_mfcc = get_mfcc(transformed)
        r_mfcc = get_mfcc(real_sound)
        distance, _ = fastdtw(t_mfcc, r_mfcc)
        print(f"Params: {params} -> DTW distance: {distance}")
        return distance
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return np.inf

# === Initial guess and bounds ===
init_params = [1.0, 150.0, 1.0, 1.0]  # [formant_shift_ratio, new_pitch, pitch_range_factor, duration_factor]
bounds = [(0.5, 2.0), (50.0, 400.0), (0.0, 2.0), (0.5, 3.0)]

# === Run optimization ===
print("Starting optimization...\n")
result = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B')

# === Show result ===
print("\n--- Optimization Complete ---")
print("Best parameters:", result.x)
print("Minimum DTW Distance:", result.fun)

# === Save the final transformed output ===
final_transformed = apply_voice_transform(
    tts_sound, *result.x
)
if final_transformed:
    final_transformed.save("transformed_tts.wav", "WAV")
    print("Saved transformed voice as 'transformed_tts.wav'")
else:
    print("Final transformation failed.")


