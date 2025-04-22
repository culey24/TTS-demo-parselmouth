import numpy as np
import soundfile as sf
import parselmouth
import tempfile
import os
from datetime import datetime
import argparse
from dtaidistance import dtw
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def change_gender(
    input_audio: np.ndarray, 
    sampling_rate: int, 
    pitch_min: float = 75, 
    pitch_max: float = 600, 
    formant_shift_ratio: float = 1.0, 
    new_pitch_median: float = 0, 
    pitch_range_factor: float = 1.0, 
    duration_factor: float = 1.0,
) -> np.ndarray:
    """
    Transforms audio using Praat's Change Gender function
    
    Args:
        input_audio: Audio data as numpy array
        sampling_rate: Sample rate of the audio
        pitch_min: Minimum pitch in Hz
        pitch_max: Maximum pitch in Hz
        formant_shift_ratio: Factor to shift formants (1.0 = no change)
        new_pitch_median: New median pitch in Hz (0 = no change)
        pitch_range_factor: Factor to change pitch range (1.0 = no change)
        duration_factor: Factor to change duration (1.0 = no change)
        
    Returns:
        Transformed audio as numpy array
    """
    assert pitch_min <= pitch_max, "pitch_min should be less than or equal to pitch_max"
    assert duration_factor <= 3.0, "duration_factor cannot be larger than 3.0"
    
    # Create temporary file for Praat
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # Write audio to temp file
        sf.write(tmp_file.name, input_audio, sampling_rate)
        
        # Load audio in Praat
        sound = parselmouth.Sound(tmp_file.name)
        
        # Apply Change Gender function
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
        
        # Convert back to numpy array
        output_audio = np.array(tuned_sound.values.T)
        
    finally:
        # Clean up temp file
        tmp_file.close()
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    return output_audio

def extract_pitch_contour(audio, sampling_rate, pitch_min=75, pitch_max=600):
    """
    Extract the fundamental frequency (F0) contour from audio using Praat
    
    Args:
        audio: Audio data as numpy array
        sampling_rate: Sample rate of the audio
        pitch_min: Minimum pitch in Hz
        pitch_max: Maximum pitch in Hz
        
    Returns:
        time_points: Array of time points
        f0_contour: Array of F0 values (0 for unvoiced frames)
    """
    # Create temporary file for Praat
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # Write audio to temp file
        sf.write(tmp_file.name, audio, sampling_rate)
        
        # Load audio in Praat
        sound = parselmouth.Sound(tmp_file.name)
        
        # Extract pitch
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=pitch_min, pitch_ceiling=pitch_max)
        
        # Get pitch values
        time_points = np.array(pitch.xs())
        f0_contour = np.array([pitch.get_value_at_time(t) for t in time_points])
        
        # Replace NaN values (unvoiced frames) with 0
        f0_contour = np.nan_to_num(f0_contour)
        
    finally:
        # Clean up temp file
        tmp_file.close()
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    return time_points, f0_contour

def calculate_dtw_distance(contour1, contour2):
    """
    Calculate the DTW distance between two pitch contours
    
    Args:
        contour1: First pitch contour
        contour2: Second pitch contour
        
    Returns:
        distance: DTW distance
        best_path: Optimal warping path
    """
    # Filter out zeros (unvoiced frames) for better comparison
    contour1_filtered = contour1[contour1 > 0]
    contour2_filtered = contour2[contour2 > 0]
    
    # If either contour is empty after filtering, return a high distance
    if len(contour1_filtered) == 0 or len(contour2_filtered) == 0:
        return float('inf'), []
    
    # Calculate DTW distance
    distance, paths = dtw.warping_paths(contour1_filtered, contour2_filtered, use_c=True)
    best_path = dtw.best_path(paths)
    
    return distance, best_path

def random_search_optimization(
    source_audio, 
    target_audio, 
    sampling_rate, 
    n_iterations=100, 
    pitch_min=75, 
    pitch_max=600
):
    """
    Perform random search to find optimal parameters
    
    Args:
        source_audio: Source audio data as numpy array
        target_audio: Target audio data as numpy array
        sampling_rate: Sample rate of the audio
        n_iterations: Number of random search iterations
        pitch_min: Minimum pitch in Hz
        pitch_max: Maximum pitch in Hz
        
    Returns:
        best_params: Dictionary of best parameters
        best_audio: Transformed audio with best parameters
        best_distance: Best DTW distance achieved
    """
    # Extract target pitch contour
    _, target_contour = extract_pitch_contour(target_audio, sampling_rate, pitch_min, pitch_max)
    
    # Initialize tracking variables
    best_distance = float('inf')
    best_params = {}
    best_audio = None
    best_contour = None
    best_path = None
    
    # Parameter ranges for random search
    param_ranges = {
        'formant_shift_ratio': (0.5, 2.0),
        'new_pitch_median': (0, 300),  # 0 means no change
        'pitch_range_factor': (0.5, 2.0),
        'duration_factor': (0.8, 1.2)  # Keep duration relatively close to original
    }
    
    # Create a history to track progress
    history = []
    
    # Run random search
    for i in tqdm(range(n_iterations)):
        # Generate random parameters
        params = {
            'formant_shift_ratio': random.uniform(*param_ranges['formant_shift_ratio']),
            'new_pitch_median': random.uniform(*param_ranges['new_pitch_median']),
            'pitch_range_factor': random.uniform(*param_ranges['pitch_range_factor']),
            'duration_factor': random.uniform(*param_ranges['duration_factor'])
        }
        
        # Transform source audio with current parameters
        try:
            transformed_audio = change_gender(
                source_audio,
                sampling_rate,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
                formant_shift_ratio=params['formant_shift_ratio'],
                new_pitch_median=params['new_pitch_median'],
                pitch_range_factor=params['pitch_range_factor'],
                duration_factor=params['duration_factor']
            )
            
            # Extract pitch contour from transformed audio
            _, transformed_contour = extract_pitch_contour(
                transformed_audio, 
                sampling_rate, 
                pitch_min, 
                pitch_max
            )
            
            # Calculate DTW distance
            distance, path = calculate_dtw_distance(transformed_contour, target_contour)
            
            # Record in history
            history.append({
                'iteration': i,
                'distance': distance,
                'params': params.copy()
            })
            
            # Update best if improved
            if distance < best_distance:
                best_distance = distance
                best_params = params.copy()
                best_audio = transformed_audio
                best_contour = transformed_contour
                best_path = path
                
                print(f"New best at iteration {i}: distance = {best_distance:.4f}")
                print(f"Parameters: {best_params}")
                
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
    
    return best_params, best_audio, best_distance, best_contour, target_contour, best_path, history

def plot_results(source_contour, target_contour, best_contour, best_path, history, output_file=None):
    """
    Plot the results of the optimization
    
    Args:
        source_contour: Original source pitch contour
        target_contour: Target pitch contour
        best_contour: Best transformed pitch contour
        best_path: DTW alignment path
        history: Optimization history
        output_file: Optional path to save plot
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
        # Filter contours for plotting alignment
        best_filtered = best_contour[best_contour > 0]
        target_filtered = target_contour[target_contour > 0]
        
        if len(best_path) > 0 and len(best_filtered) > 0 and len(target_filtered) > 0:
            plt.plot(best_filtered, label='Optimized', color='red')
            plt.plot(target_filtered, label='Target', color='green')
            
            # Draw alignment lines
            for i, j in best_path:
                if i < len(best_filtered) and j < len(target_filtered):
                    plt.plot([i, j], [best_filtered[i], target_filtered[j]], 
                             color='gray', alpha=0.3, linestyle='-')
        
        plt.title('DTW Alignment')
        plt.legend()
    
    # Plot 3: Optimization progress
    plt.subplot(2, 2, 3)
    distances = [entry['distance'] for entry in history if entry['distance'] != float('inf')]
    plt.plot(distances)
    plt.title('DTW Distance Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('DTW Distance')
    
    # Plot 4: Parameter values
    plt.subplot(2, 2, 4)
    param_names = ['formant_shift_ratio', 'new_pitch_median', 'pitch_range_factor', 'duration_factor']
    for param in param_names:
        values = [entry['params'][param] for entry in history if entry['distance'] != float('inf')]
        plt.plot(values, label=param)
    plt.title('Parameter Values Over Iterations')
    plt.xlabel('Iteration')
    plt.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Optimize voice parameters using DTW")
    parser.add_argument("source_file", help="Path to source audio file")
    parser.add_argument("target_file", help="Path to target audio file")
    parser.add_argument("--output-file", "-o", help="Path to output audio file (optional)")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="Number of random search iterations")
    parser.add_argument("--pitch-min", type=float, default=75,
                        help="Minimum pitch in Hz")
    parser.add_argument("--pitch-max", type=float, default=600,
                        help="Maximum pitch in Hz")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Generate and display result plots")
    parser.add_argument("--plot-file", help="Path to save plot (optional)")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(args.source_file)
        args.output_file = f"{base}_optimized_{timestamp}{ext}"
    
    # Read input audio files
    source_audio, source_rate = sf.read(args.source_file)
    target_audio, target_rate = sf.read(args.target_file)
    
    # Ensure same sampling rate
    if source_rate != target_rate:
        print(f"Warning: Different sampling rates. Source: {source_rate}Hz, Target: {target_rate}Hz")
        print("Using source sampling rate for processing.")
    
    # Extract original pitch contours for reference
    _, source_contour = extract_pitch_contour(
        source_audio, source_rate, args.pitch_min, args.pitch_max
    )
    
    # Run optimization
    print(f"Starting random search optimization with {args.iterations} iterations...")
    best_params, best_audio, best_distance, best_contour, target_contour, best_path, history = (
        random_search_optimization(
            source_audio,
            target_audio,
            source_rate,
            n_iterations=args.iterations,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max
        )
    )
    
    # Save optimized audio
    sf.write(args.output_file, best_audio, source_rate)
    print(f"Optimized audio saved to: {args.output_file}")
    print(f"Best DTW distance: {best_distance:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Plot results if requested
    if args.plot or args.plot_file:
        plot_results(
            source_contour,
            target_contour,
            best_contour,
            best_path,
            history,
            output_file=args.plot_file
        )
    
    return 0

if __name__ == "__main__":
    exit(main())