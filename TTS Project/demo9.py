import numpy as np
import soundfile as sf
import parselmouth
import os
import json
from datetime import datetime
import argparse
from dtaidistance import dtw
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


# Đường dẫn file tạm cố định
TEMP_FILE = "temp_audio.wav"

def change_gender(
    input_audio: np.ndarray, 
    sampling_rate: int, 
    pitch_min: float = 75, 
    pitch_max: float = 600, 
    formant_shift_ratio: float = 1.0, 
    new_pitch_median: float = 0, 
    pitch_range_factor: float = 1.0, 
    duration_factor: float = 1.0,
    temp_file_path: str = TEMP_FILE
) -> np.ndarray:
    """
    Sử dụng hàm Change Gender của Praat để thay đổi đặc trưng giọng nói.
    """
    sf.write(temp_file_path, input_audio, sampling_rate)
    sound = parselmouth.Sound(temp_file_path)
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
    output_audio = np.array(tuned_sound.values.T)
    return output_audio

def extract_pitch_contour(audio, sampling_rate, pitch_min=75, pitch_max=600):
    """
    Trích xuất pitch contour từ audio sử dụng Praat.
    """
    sf.write(TEMP_FILE, audio, sampling_rate)
    sound = parselmouth.Sound(TEMP_FILE)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=pitch_min, pitch_ceiling=pitch_max)
    time_points = np.array(pitch.xs())
    f0_contour = np.array([pitch.get_value_at_time(t) for t in time_points])
    f0_contour = np.nan_to_num(f0_contour)
    return time_points, f0_contour

def extract_formants(audio, sampling_rate, max_formant=5500, num_formants=5):
    """
    Trích xuất formants từ audio sử dụng Praat.
    """
    sf.write(TEMP_FILE, audio, sampling_rate)
    sound = parselmouth.Sound(TEMP_FILE)
    formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=num_formants, 
                                     maximum_formant=max_formant)
    time_points = np.arange(0, sound.duration, 0.01)
    formant_data = {'time': time_points, 'formants': []}
    for t in time_points:
        frame_formants = []
        for i in range(1, num_formants + 1):
            try:
                value = formants.get_value_at_time(formant_number=i, time=t)
                frame_formants.append(value if not np.isnan(value) else 0)
            except:
                frame_formants.append(0)
        formant_data['formants'].append(frame_formants)
    formant_data['formants'] = np.array(formant_data['formants'])
    return formant_data

def calculate_dtw_distance(contour1, contour2):
    """
    Tính khoảng cách DTW giữa hai contour.
    """
    contour1_filtered = contour1[contour1 > 0]
    contour2_filtered = contour2[contour2 > 0]
    if len(contour1_filtered) == 0 or len(contour2_filtered) == 0:
        return float('inf'), []
    contour1_norm = (contour1_filtered - np.mean(contour1_filtered)) / (np.std(contour1_filtered) + 1e-6)
    contour2_norm = (contour2_filtered - np.mean(contour2_filtered)) / (np.std(contour2_filtered) + 1e-6)
    distance, paths = dtw.warping_paths(contour1_norm, contour2_norm, use_c=True)
    best_path = dtw.best_path(paths)
    return distance, best_path

def apply_target_intonation(transformed_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max):
    """
    Áp dụng ngữ điệu (pitch contour) từ giọng mục tiêu lên giọng đã chuyển đổi.
    """
    time_points_transformed, transformed_contour = extract_pitch_contour(
        transformed_audio, source_rate, pitch_min, pitch_max
    )
    time_points_target, target_contour = extract_pitch_contour(
        target_audio, target_rate, pitch_min, pitch_max
    )
    min_length = min(len(transformed_contour), len(target_contour))
    time_points_transformed = time_points_transformed[:min_length]
    transformed_contour = transformed_contour[:min_length]
    time_points_target = time_points_target[:min_length]
    target_contour = target_contour[:min_length]
    sf.write(TEMP_FILE, transformed_audio, source_rate)
    sound = parselmouth.Sound(TEMP_FILE)
    manipulation = sound.to_manipulation()
    pitch_tier = parselmouth.PitchTier(0, sound.duration)
    for t, target_f0 in zip(time_points_target, target_contour):
        if target_f0 > 0:
            pitch_tier.add_point(t, target_f0)
    parselmouth.praat.call((manipulation, pitch_tier), "Replace pitch tier")
    tuned_sound = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    output_audio = np.array(tuned_sound.values.T)
    return output_audio

def apply_target_rhythm(transformed_audio, target_audio, source_rate, target_rate):
    """
    Áp dụng nhịp điệu (intensity contour) từ giọng mục tiêu.
    """
    sf.write(TEMP_FILE, transformed_audio, source_rate)
    transformed_sound = parselmouth.Sound(TEMP_FILE)
    sf.write("temp_target.wav", target_audio, target_rate)
    target_sound = parselmouth.Sound("temp_target.wav")
    transformed_intensity = transformed_sound.to_intensity(minimum_pitch=100)
    target_intensity = target_sound.to_intensity(minimum_pitch=100)
    time_points = np.array(transformed_intensity.xs())
    transformed_int = np.array([transformed_intensity.get_value(t) for t in time_points])
    target_int = np.array([target_intensity.get_value(t) for t in time_points])
    min_length = min(len(transformed_int), len(target_int))
    time_points = time_points[:min_length]
    transformed_int = transformed_int[:min_length]
    target_int = target_int[:min_length]
    valid_indices = (transformed_int > 0) & (target_int > 0)
    if np.sum(valid_indices) > 0:
        int_ratio = target_int[valid_indices] / transformed_int[valid_indices]
        mean_int_ratio = np.mean(int_ratio)
    else:
        mean_int_ratio = 1.0
    intensity_tier = parselmouth.IntensityTier(0, transformed_sound.duration)
    for t, target_i in zip(time_points, target_int):
        if target_i > 0:
            adjusted_i = target_i / mean_int_ratio
            intensity_tier.add_point(t, adjusted_i)
    tuned_sound = parselmouth.praat.call(
        (transformed_sound, intensity_tier), "Multiply", 100
    )
    output_audio = np.array(tuned_sound.values.T)
    os.remove("temp_target.wav")
    return output_audio

def adjust_formant_bandwidth(audio, source_rate, bandwidth_factor=1.2):
    """
    Điều chỉnh formant bandwidth để cải thiện âm sắc.
    """
    sf.write(TEMP_FILE, audio, source_rate)
    sound = parselmouth.Sound(TEMP_FILE)
    formants = sound.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500
    )
    formant_new = parselmouth.Formant(formants.xs(), formants.ys())
    for t in formants.xs():
        for i in range(1, 6):
            try:
                freq = formants.get_value_at_time(i, t)
                bw = formants.get_bandwidth_at_time(i, t)
                if not np.isnan(freq) and not np.isnan(bw):
                    formant_new.add_point(t, freq, bw * bandwidth_factor)
            except:
                pass
    tuned_sound = parselmouth.praat.call(
        (sound, formant_new), "Resynthesize", "overlap-add"
    )
    output_audio = np.array(tuned_sound.values.T)
    return output_audio

def add_breathiness(audio, source_rate, noise_amplitude=0.01):
    """
    Thêm nhiễu trắng nhẹ để mô phỏng breathiness.
    """
    noise = np.random.normal(0, noise_amplitude, len(audio))
    output_audio = audio + noise
    max_amplitude = np.max(np.abs(output_audio))
    if max_amplitude > 1.0:
        output_audio = output_audio / max_amplitude * 0.95
    return output_audio

def calculate_combined_score(
    source_audio, 
    transformed_audio, 
    target_audio, 
    source_rate, 
    target_rate,
    pitch_min,
    pitch_max,
    pitch_weight=0.4,
    formant_weight=0.3,
    intensity_weight=0.3
):
    """
    Tính điểm tương đồng kết hợp giữa pitch, formant và intensity.
    """
    _, transformed_contour = extract_pitch_contour(
        transformed_audio, source_rate, pitch_min, pitch_max
    )
    _, target_contour = extract_pitch_contour(
        target_audio, target_rate, pitch_min, pitch_max
    )
    pitch_distance, path = calculate_dtw_distance(transformed_contour, target_contour)

    transformed_formants = extract_formants(
        transformed_audio, source_rate, max_formant=5500, num_formants=3
    )
    target_formants = extract_formants(
        target_audio, target_rate, max_formant=5500, num_formants=3
    )
    formant_distances = []
    for i in range(3):
        tf = transformed_formants['formants'][:, i]
        tf = tf[tf > 0]
        tgf = target_formants['formants'][:, i]
        tgf = tgf[tgf > 0]
        if len(tf) > 0 and len(tgf) > 0:
            tf_norm = (tf - np.mean(tf)) / (np.std(tf) + 1e-6)
            tgf_norm = (tgf - np.mean(tgf)) / (np.std(tgf) + 1e-6)
            f_dist, _ = dtw.warping_paths(tf_norm, tgf_norm, use_c=True)
            formant_distances.append(f_dist)
    formant_distance = np.mean(formant_distances) if formant_distances else float('inf')

    sf.write(TEMP_FILE, transformed_audio, source_rate)
    transformed_sound = parselmouth.Sound(TEMP_FILE)
    sf.write("temp_target.wav", target_audio, target_rate)
    target_sound = parselmouth.Sound("temp_target.wav")
    transformed_intensity = transformed_sound.to_intensity(minimum_pitch=100)
    target_intensity = target_sound.to_intensity(minimum_pitch=100)
    time_points = np.array(transformed_intensity.xs())
    transformed_int = np.array([transformed_intensity.get_value(t) for t in time_points])
    target_int = np.array([target_intensity.get_value(t) for t in time_points])
    min_length = min(len(transformed_int), len(target_int))
    transformed_int = transformed_int[:min_length]
    target_int = target_int[:min_length]
    int_distance, _ = dtw.warping_paths(
        (transformed_int - np.mean(transformed_int)) / (np.std(transformed_int) + 1e-6),
        (target_int - np.mean(target_int)) / (np.std(target_int) + 1e-6),
        use_c=True
    )
    os.remove("temp_target.wav")

    combined_score = (
        pitch_weight * pitch_distance +
        formant_weight * formant_distance +
        intensity_weight * int_distance
    )
    return combined_score, path

def evaluate_parameter_set(params, source_audio, target_audio, source_rate, target_rate, 
                          pitch_min, pitch_max):
    """
    Đánh giá một bộ tham số bằng cách chuyển đổi và tính điểm tương đồng.
    """
    try:
        transformed_audio = change_gender(
            source_audio,
            source_rate,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            formant_shift_ratio=params['formant_shift_ratio'],
            new_pitch_median=params['new_pitch_median'],
            pitch_range_factor=params['pitch_range_factor'],
            duration_factor=params['duration_factor'],
            temp_file_path=TEMP_FILE
        )
        transformed_audio = apply_target_intonation(
            transformed_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max
        )
        transformed_audio = apply_target_rhythm(
            transformed_audio, target_audio, source_rate, target_rate
        )
        transformed_audio = adjust_formant_bandwidth(
            transformed_audio, source_rate, bandwidth_factor=1.2
        )
        transformed_audio = add_breathiness(transformed_audio, source_rate, noise_amplitude=0.01)
        score, path = calculate_combined_score(
            source_audio, 
            transformed_audio, 
            target_audio, 
            source_rate, 
            target_rate,
            pitch_min,
            pitch_max
        )
        return score, transformed_audio, path
    except Exception as e:
        print(f"Error in parameter evaluation: {e}")
        return float('inf'), None, None

def bayesian_optimization(
    source_audio, 
    target_audio, 
    source_rate,
    target_rate,
    pitch_min=75, 
    pitch_max=600,
    max_iter=30
):
    """
    Tối ưu hóa tham số sử dụng Bayesian Optimization.
    """
    def objective_function(formant_shift_ratio, new_pitch_median, pitch_range_factor, duration_factor):
        params = {
            'formant_shift_ratio': formant_shift_ratio,
            'new_pitch_median': new_pitch_median,
            'pitch_range_factor': pitch_range_factor,
            'duration_factor': duration_factor
        }
        score, _, _ = evaluate_parameter_set(
            params, source_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max
        )
        return -score  # BO tối ưu hóa để tối đa hóa, nên lấy -score để tối thiểu hóa

    pbounds = {
        'formant_shift_ratio': (0.5, 2.0),
        'new_pitch_median': (0, 300),
        'pitch_range_factor': (0.5, 2.0),
        'duration_factor': (0.8, 1.2)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=max_iter
    )

    best_params = optimizer.max['params']
    best_score = -optimizer.max['target']
    _, best_audio, best_path = evaluate_parameter_set(
        best_params, source_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max
    )
    _, best_contour = extract_pitch_contour(best_audio, source_rate, pitch_min, pitch_max)
    _, target_contour = extract_pitch_contour(target_audio, target_rate, pitch_min, pitch_max)
    return best_params, best_audio, best_score, best_contour, target_contour, best_path, []

def plot_results(source_contour, target_contour, best_contour, best_path, history, output_file=None):
    """
    Vẽ biểu đồ kết quả.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Pitch contours
    plt.subplot(2, 2, 1)
    plt.plot(source_contour, label='Source', color='blue', alpha=0.5)
    plt.plot(target_contour, label='Target', color='green')
    plt.plot(best_contour, label='Optimized', color='red', linestyle='--')
    plt.title('Pitch Contours')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    
    # Plot 2: DTW alignment
    if best_path:
        plt.subplot(2, 2, 2)
        best_filtered = best_contour[best_contour > 0]
        target_filtered = target_contour[target_contour > 0]
        if len(best_path) > 0 and len(best_filtered) > 0 and len(target_filtered) > 0:
            plt.plot(best_filtered, label='Optimized', color='red')
            plt.plot(target_filtered, label='Target', color='green')
            for i, j in best_path:
                if i < len(best_filtered) and j < len(target_filtered):
                    plt.plot([i, j], [best_filtered[i], target_filtered[j]], 
                             color='gray', alpha=0.3, linestyle='-')
        plt.title('DTW Alignment')
        plt.legend()
    
    # Plot 3: Optimization progress
    plt.subplot(2, 2, 3)
    scores = [entry['score'] for entry in history if entry['score'] != float('inf')]
    iterations = [entry['iteration'] for entry in history if entry['score'] != float('inf')]
    if scores:
        plt.plot(iterations, scores)
        plt.title('Score Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Score (lower is better)')
    
    # Plot 4: Parameter values for best results
    plt.subplot(2, 2, 4)
    param_names = ['formant_shift_ratio', 'new_pitch_median', 'pitch_range_factor', 'duration_factor']
    top_entries = sorted(history, key=lambda x: x['score'])[:20] if history else []
    for param in param_names:
        values = [entry['params'][param] for entry in top_entries]
        scores = [entry['score'] for entry in top_entries]
        if values:
            plt.scatter(values, scores, label=param, alpha=0.7)
    plt.title('Parameter Values vs Score (Top 20)')
    plt.xlabel('Parameter Value')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Optimize voice parameters for natural voice conversion")
    parser.add_argument("source_file", help="Path to source audio file")
    parser.add_argument("target_file", help="Path to target audio file")
    parser.add_argument("--output-file", "-o", help="Path to output audio file (optional)")
    parser.add_argument("--max-iter", type=int, default=30, help="Number of iterations for Bayesian Optimization")
    parser.add_argument("--pitch-min", type=float, default=75, help="Minimum pitch in Hz")
    parser.add_argument("--pitch-max", type=float, default=600, help="Maximum pitch in Hz")
    parser.add_argument("--plot", "-p", action="store_true", help="Generate and display result plots")
    parser.add_argument("--plot-file", help="Path to save plot (optional)")

    args = parser.parse_args()

    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(args.source_file)
        args.output_file = f"{base}_optimized_{timestamp}{ext}"

    source_audio, source_rate = sf.read(args.source_file)
    target_audio, target_rate = sf.read(args.target_file)

    if source_rate != target_rate:
        print(f"Warning: Different sampling rates. Source: {source_rate}Hz, Target: {target_rate}Hz")
        print("Using source sampling rate for processing.")

    print("Extracting pitch contours...")
    _, source_contour = extract_pitch_contour(source_audio, source_rate, args.pitch_min, args.pitch_max)

    print("Starting Bayesian optimization...")
    best_params, best_audio, best_score, best_contour, target_contour, best_path, history = (
        bayesian_optimization(
            source_audio,
            target_audio,
            source_rate,
            target_rate,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            max_iter=args.max_iter
        )
    )

    sf.write(args.output_file, best_audio, source_rate)
    print(f"Optimized audio saved to: {args.output_file}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    if args.plot or args.plot_file:
        plot_results(
            source_contour,
            target_contour,
            best_contour,
            best_path,
            history,
            output_file=args.plot_file
        )

    # Xóa file tạm nếu có
    if os.path.exists(TEMP_FILE):
        os.unlink(TEMP_FILE)

    return 0

if __name__ == "__main__":
    exit(main())