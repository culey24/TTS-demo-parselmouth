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
import concurrent.futures
import time

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

def extract_formants(audio, sampling_rate, max_formant=5500, num_formants=5):
    """
    Extract formant frequencies from audio using Praat
    
    Args:
        audio: Audio data as numpy array
        sampling_rate: Sample rate of the audio
        max_formant: Maximum formant frequency in Hz
        num_formants: Number of formants to extract
        
    Returns:
        formant_data: Dictionary with formant frequencies
    """
    # Create temporary file for Praat
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # Write audio to temp file
        sf.write(tmp_file.name, audio, sampling_rate)
        
        # Load audio in Praat
        sound = parselmouth.Sound(tmp_file.name)
        
        # Extract formants
        formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=num_formants, 
                                         maximum_formant=max_formant)
        
        # Get time points
        time_points = np.arange(0, sound.duration, 0.01)
        
        # Extract formant values at each time point
        formant_data = {
            'time': time_points,
            'formants': []
        }
        
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
        
    finally:
        # Clean up temp file
        tmp_file.close()
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
    
    return formant_data

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
    
    # Normalize contours to make comparison more robust
    contour1_norm = (contour1_filtered - np.mean(contour1_filtered)) / (np.std(contour1_filtered) + 1e-6)
    contour2_norm = (contour2_filtered - np.mean(contour2_filtered)) / (np.std(contour2_filtered) + 1e-6)
    
    # Calculate DTW distance
    distance, paths = dtw.warping_paths(contour1_norm, contour2_norm, use_c=True)
    best_path = dtw.best_path(paths)
    
    return distance, best_path

def calculate_combined_score(
    source_audio, 
    transformed_audio, 
    target_audio, 
    source_rate, 
    target_rate,
    pitch_min,
    pitch_max,
    pitch_weight=0.7,
    formant_weight=0.3
):
    """
    Calculate a combined score based on pitch and formant similarity
    
    Args:
        source_audio: Source audio data
        transformed_audio: Transformed audio data
        target_audio: Target audio data
        source_rate: Source sampling rate
        target_rate: Target sampling rate
        pitch_min: Minimum pitch in Hz
        pitch_max: Maximum pitch in Hz
        pitch_weight: Weight for pitch similarity
        formant_weight: Weight for formant similarity
    
    Returns:
        score: Combined similarity score (lower is better)
    """
    # Extract pitch contours
    _, transformed_contour = extract_pitch_contour(
        transformed_audio, source_rate, pitch_min, pitch_max
    )
    _, target_contour = extract_pitch_contour(
        target_audio, target_rate, pitch_min, pitch_max

    )
    
    # Calculate pitch DTW distance
    pitch_distance, path = calculate_dtw_distance(transformed_contour, target_contour)
    
    # Extract formants (first 3 formants are most important for voice quality)
    transformed_formants = extract_formants(
        transformed_audio, source_rate, max_formant=5500, num_formants=3
    )
    
    target_formants = extract_formants(
        target_audio, target_rate, max_formant=5500, num_formants=3
    )
    
    # Calculate formant distances
    formant_distances = []
    for i in range(3):  # Consider first 3 formants
        # Extract formant values, filtering zeros
        tf = transformed_formants['formants'][:, i]
        tf = tf[tf > 0]
        
        tgf = target_formants['formants'][:, i]
        tgf = tgf[tgf > 0]
        
        if len(tf) > 0 and len(tgf) > 0:
            # Normalize and calculate distance
            tf_norm = (tf - np.mean(tf)) / (np.std(tf) + 1e-6)
            tgf_norm = (tgf - np.mean(tgf)) / (np.std(tgf) + 1e-6)
            
            f_dist, _ = dtw.warping_paths(tf_norm, tgf_norm, use_c=True)
            formant_distances.append(f_dist)
    
    # Average formant distance
    formant_distance = np.mean(formant_distances) if formant_distances else float('inf')
    
    # Combine scores
    combined_score = pitch_weight * pitch_distance + formant_weight * formant_distance
    
    return combined_score, path

def evaluate_parameter_set(params, source_audio, target_audio, source_rate, target_rate, 
                          pitch_min, pitch_max):
    """
    Evaluate a single parameter set
    
    Args:
        params: Dictionary of parameters
        source_audio: Source audio data
        target_audio: Target audio data
        source_rate: Source sampling rate
        target_rate: Target sampling rate
        pitch_min: Minimum pitch in Hz
        pitch_max: Maximum pitch in Hz
        
    Returns:
        score: Similarity score
        transformed_audio: Transformed audio
        path: DTW alignment path
    """
    try:
        # Transform source audio with current parameters
        transformed_audio = change_gender(
            source_audio,
            source_rate,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            formant_shift_ratio=params['formant_shift_ratio'],
            new_pitch_median=params['new_pitch_median'],
            pitch_range_factor=params['pitch_range_factor'],
            duration_factor=params['duration_factor']
        )
        
        # Calculate combined score
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

def random_search_optimization(
    source_audio, 
    target_audio, 
    source_rate,
    target_rate,
    n_iterations=100, 
    n_best=5,
    pitch_min=75, 
    pitch_max=600,
    parallel=True,
    max_workers=4
):
    """
    Perform random search to find optimal parameters with local refinement
    
    Args:
        source_audio: Source audio data as numpy array
        target_audio: Target audio data as numpy array
        source_rate: Source sampling rate
        target_rate: Target sampling rate
        n_iterations: Number of random search iterations
        n_best: Number of best candidates for refinement
        pitch_min: Minimum pitch in Hz
        pitch_max: Maximum pitch in Hz
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        
    Returns:
        best_params: Dictionary of best parameters
        best_audio: Transformed audio with best parameters
        best_score: Best score achieved
    """
    # Extract target pitch contour for reference
    _, target_contour = extract_pitch_contour(target_audio, target_rate, pitch_min, pitch_max)
    _, source_contour = extract_pitch_contour(source_audio, source_rate, pitch_min, pitch_max)
    
    # Initialize tracking variables
    best_score = float('inf')
    best_params = {}
    best_audio = None
    best_path = None
    
    # Intelligent parameter ranges based on source and target pitch analysis
    source_voiced = source_contour[source_contour > 0]
    target_voiced = target_contour[target_contour > 0]
    
    # Calculate median pitches
    if len(source_voiced) > 0 and len(target_voiced) > 0:
        source_median = np.median(source_voiced)
        target_median = np.median(target_voiced)
        
        # Calculate pitch ratio for initial guidance
        pitch_ratio = target_median / source_median if source_median > 0 else 1.0
        
        # Set up parameter ranges based on analysis
        param_ranges = {
            'formant_shift_ratio': (0.5, 2.0),  # Full range for formant exploration
            'new_pitch_median': (0, target_median * 1.5),  # Center around target median
            'pitch_range_factor': (0.5, 2.0),
            'duration_factor': (0.8, 1.2)  # Keep duration relatively close to original
        }
        
        # Smart initial guess based on pitch analysis
        initial_params = {
            'formant_shift_ratio': pitch_ratio if 0.5 <= pitch_ratio <= 2.0 else 1.0,
            'new_pitch_median': target_median,
            'pitch_range_factor': 1.0,
            'duration_factor': 1.0
        }
    else:
        # Default ranges if analysis fails
        param_ranges = {
            'formant_shift_ratio': (0.5, 2.0),
            'new_pitch_median': (0, 300),
            'pitch_range_factor': (0.5, 2.0),
            'duration_factor': (0.8, 1.2)
        }
        
        initial_params = {
            'formant_shift_ratio': 1.0,
            'new_pitch_median': 0,
            'pitch_range_factor': 1.0,
            'duration_factor': 1.0
        }
    
    # Create a history to track progress
    history = []
    
    # Evaluate initial parameters
    print("Evaluating initial parameters...")
    score, audio, path = evaluate_parameter_set(
        initial_params, source_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max
    )
    
    if score < best_score:
        best_score = score
        best_params = initial_params.copy()
        best_audio = audio
        best_path = path
        
    history.append({
        'iteration': 0,
        'score': score,
        'params': initial_params.copy()
    })
    
    print(f"Initial parameters score: {score:.4f}")
    print(f"Parameters: {initial_params}")
    
    # Generate random parameter sets
    parameter_sets = []
    for i in range(n_iterations):
        params = {
            'formant_shift_ratio': random.uniform(*param_ranges['formant_shift_ratio']),
            'new_pitch_median': random.uniform(*param_ranges['new_pitch_median']),
            'pitch_range_factor': random.uniform(*param_ranges['pitch_range_factor']),
            'duration_factor': random.uniform(*param_ranges['duration_factor'])
        }
        parameter_sets.append((i, params))
    
    # Run random search
    print(f"\nStarting random search with {n_iterations} iterations...")
    results = []
    
    if parallel:
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {
                executor.submit(
                    evaluate_parameter_set, 
                    params, 
                    source_audio, 
                    target_audio, 
                    source_rate, 
                    target_rate, 
                    pitch_min, 
                    pitch_max
                ): (i, params) for i, params in parameter_sets
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_params), total=len(parameter_sets)):
                i, params = future_to_params[future]
                try:
                    score, audio, path = future.result()
                    results.append((i, params, score, audio, path))
                except Exception as e:
                    print(f"Error in iteration {i}: {e}")
    else:
        # Sequential execution
        for i, params in tqdm(parameter_sets):
            score, audio, path = evaluate_parameter_set(
                params, source_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max
            )
            results.append((i, params, score, audio, path))
    
    # Process results
    valid_results = [(i, params, score, audio, path) for i, params, score, audio, path in results if score != float('inf')]
    
    # Sort by score
    valid_results.sort(key=lambda x: x[2])
    
    # Record all results in history
    for i, params, score, _, _ in valid_results:
        history.append({
            'iteration': i + 1,  # +1 because 0 is the initial guess
            'score': score,
            'params': params.copy()
        })
    
    # Update best if improved
    if valid_results and valid_results[0][2] < best_score:
        i, best_params, best_score, best_audio, best_path = valid_results[0]
        print(f"\nBest from random search (iteration {i}): score = {best_score:.4f}")
        print(f"Parameters: {best_params}")
    
    # Local refinement phase
    print("\nStarting local refinement phase...")
    
    # Take top n_best candidates for refinement
    top_candidates = valid_results[:n_best]
    
    # Refine each candidate
    refined_results = []
    
    for candidate_idx, (i, params, score, audio, path) in enumerate(top_candidates):
        print(f"\nRefining candidate {candidate_idx + 1}/{n_best} (original score: {score:.4f})...")
        
        # Create small variations around this candidate
        refinement_iterations = 20
        refinement_sets = []
        
        for j in range(refinement_iterations):
            # Generate variations proportional to parameter value but with decreasing magnitude
            scale_factor = 0.1 * (1 - j/refinement_iterations)  # Gradually decrease the search space
            
            refined_params = {
                'formant_shift_ratio': max(0.1, params['formant_shift_ratio'] * (1 + scale_factor * (random.random() * 2 - 1))),
                'new_pitch_median': max(0, params['new_pitch_median'] * (1 + scale_factor * (random.random() * 2 - 1))),
                'pitch_range_factor': max(0.1, params['pitch_range_factor'] * (1 + scale_factor * (random.random() * 2 - 1))),
                'duration_factor': max(0.1, min(3.0, params['duration_factor'] * (1 + scale_factor * (random.random() * 2 - 1))))
            }
            
            refinement_sets.append((j, refined_params))
        
        # Evaluate refinements
        refinement_results = []
        
        if parallel:
            # Parallel execution for refinements
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {
                    executor.submit(
                        evaluate_parameter_set, 
                        params, 
                        source_audio, 
                        target_audio, 
                        source_rate, 
                        target_rate, 
                        pitch_min, 
                        pitch_max
                    ): (j, params) for j, params in refinement_sets
                }
                
                for future in concurrent.futures.as_completed(future_to_params):
                    j, params = future_to_params[future]
                    try:
                        ref_score, ref_audio, ref_path = future.result()
                        refinement_results.append((j, params, ref_score, ref_audio, ref_path))
                    except Exception as e:
                        print(f"Error in refinement {j}: {e}")
        else:
            # Sequential execution for refinements
            for j, params in tqdm(refinement_sets, desc=f"Refining candidate {candidate_idx + 1}"):
                ref_score, ref_audio, ref_path = evaluate_parameter_set(
                    params, source_audio, target_audio, source_rate, target_rate, pitch_min, pitch_max
                )
                refinement_results.append((j, params, ref_score, ref_audio, ref_path))
        
        # Add to overall results
        valid_refinements = [(j, params, score, audio, path) for j, params, score, audio, path in refinement_results 
                           if score != float('inf')]
        
        if valid_refinements:
            # Sort by score
            valid_refinements.sort(key=lambda x: x[2])
            refined_results.append(valid_refinements[0])
            
            # Record in history
            for j, params, score, _, _ in valid_refinements:
                history.append({
                    'iteration': n_iterations + i * refinement_iterations + j + 1,
                    'score': score,
                    'params': params.copy()
                })
    
    # Find best refinement
    if refined_results:
        refined_results.sort(key=lambda x: x[2])
        j, refined_params, refined_score, refined_audio, refined_path = refined_results[0]
        
        if refined_score < best_score:
            best_score = refined_score
            best_params = refined_params
            best_audio = refined_audio
            best_path = refined_path
            
            print(f"\nBest from refinement: score = {best_score:.4f}")
            print(f"Parameters: {best_params}")
    
    # Extract best contour for plotting
    _, best_contour = extract_pitch_contour(best_audio, source_rate, pitch_min, pitch_max)
    
    return best_params, best_audio, best_score, best_contour, target_contour, best_path, history

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
    scores = [entry['score'] for entry in history if entry['score'] != float('inf')]
    iterations = [entry['iteration'] for entry in history if entry['score'] != float('inf')]
    plt.plot(iterations, scores)
    plt.title('Score Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Score (lower is better)')
    
    # Plot 4: Parameter values for best results
    plt.subplot(2, 2, 4)
    param_names = ['formant_shift_ratio', 'new_pitch_median', 'pitch_range_factor', 'duration_factor']
    
    # Get top 20 results
    top_entries = sorted(history, key=lambda x: x['score'])[:20]
    
    for param in param_names:
        values = [entry['params'][param] for entry in top_entries]
        scores = [entry['score'] for entry in top_entries]
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
    parser = argparse.ArgumentParser(description="Optimize voice parameters using DTW")
    parser.add_argument("source_file", help="Path to source audio file")
    parser.add_argument("target_file", help="Path to target audio file")
    parser.add_argument("--output-file", "-o", help="Path to output audio file (optional)")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="Number of random search iterations")
    parser.add_argument("--best-candidates", "-b", type=int, default=5,
                        help="Number of best candidates to refine")
    parser.add_argument("--pitch-min", type=float, default=75,
                        help="Minimum pitch in Hz")
    parser.add_argument("--pitch-max", type=float, default=600,
                        help="Maximum pitch in Hz")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Generate and display result plots")
    parser.add_argument("--plot-file", help="Path to save plot (optional)")
    parser.add_argument("--sequential", "-s", action="store_true",
                        help="Run in sequential mode (no parallelization)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of worker processes for parallel execution")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(args.source_file)
        args.output_file = f"{base}_optimized_{timestamp}{ext}"
    
    # Record start time
    start_time = time.time()
    
    # Read input audio files
    print(f"Reading source file: {args.source_file}")
    source_audio, source_rate = sf.read(args.source_file)
    
    print(f"Reading target file: {args.target_file}")
    target_audio, target_rate = sf.read(args.target_file)
    
    # Ensure same sampling rate
    if source_rate != target_rate:
        print(f"Warning: Different sampling rates. Source: {source_rate}Hz, Target: {target_rate}Hz")
        print("Using source sampling rate for processing.")
    
    # Extract original pitch contours for reference
    print("Extracting pitch contours...")
    _, source_contour = extract_pitch_contour(
        source_audio, source_rate, args.pitch_min, args.pitch_max
    )
    
    # Run optimization
    print(f"Starting optimization with {args.iterations} iterations and {args.best_candidates} refinement candidates...")
    best_params, best_audio, best_score, best_contour, target_contour, best_path, history = (
        random_search_optimization(
            source_audio,
            target_audio,
            source_rate,
            target_rate,
            n_iterations=args.iterations,
            n_best=args.best_candidates,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            parallel=not args.sequential,
            max_workers=args.workers
        )
    )
    
    # Save optimized audio
    sf.write(args.output_file, best_audio, source_rate)
    print(f"Optimized audio saved to: {args.output_file}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds")
    
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