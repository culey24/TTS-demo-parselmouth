import parselmouth
import numpy as np
import soundfile as sf
import tempfile
import os

# === Function to change gender/tune audio ===
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


# === Load input audio ===
INPUT_PATH = "tts.wav"  # Change this to your input file
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"File not found: {INPUT_PATH}")

sound = parselmouth.Sound(INPUT_PATH)
audio_array = sound.values.T.astype(np.float32)
sample_rate = int(sound.sampling_frequency)

# === User parameters (change these!) ===
formant_shift_ratio = 1.6
new_pitch_median = 250.0
pitch_range_factor = 1.3
duration_factor = 1.0

# === Apply transformation ===
transformed_audio = change_gender(
    audio_array,
    sample_rate,
    pitch_min=75,
    pitch_max=600,
    formant_shift_ratio=formant_shift_ratio,
    new_pitch_median=new_pitch_median,
    pitch_range_factor=pitch_range_factor,
    duration_factor=duration_factor
)

# === Save transformed audio ===
output_path = "tuned_output.wav"
sf.write(output_path, transformed_audio.flatten(), sample_rate)
print(f"Saved transformed audio as '{output_path}'")
