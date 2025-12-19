import numpy as np
import librosa
import sys

def compare_audio_files(file1, file2):
    """
    Compare two stereo audio files and return the overall difference percentage.
    
    Args:
        file1: Path to first audio file
        file2: Path to second audio file
    
    Returns:
        Dictionary containing difference metrics
    """
    try:
        # Load audio files (stereo=True preserves both channels)
        audio1, sr1 = librosa.load(file1, sr=None, mono=False)
        audio2, sr2 = librosa.load(file2, sr=None, mono=False)
        
        # Check if sample rates match
        if sr1 != sr2:
            print(f"Warning: Sample rates differ ({sr1} Hz vs {sr2} Hz)")
            print("Resampling second file to match first...")
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1
        
        # Handle mono files
        if audio1.ndim == 1:
            audio1 = np.array([audio1, audio1])
        if audio2.ndim == 1:
            audio2 = np.array([audio2, audio2])
        
        # Match lengths by padding or truncating
        min_len = min(audio1.shape[1], audio2.shape[1])
        max_len = max(audio1.shape[1], audio2.shape[1])
        
        if audio1.shape[1] != audio2.shape[1]:
            print(f"Warning: Audio lengths differ ({audio1.shape[1]} vs {audio2.shape[1]} samples)")
            # Pad shorter audio with zeros
            if audio1.shape[1] < audio2.shape[1]:
                audio1 = np.pad(audio1, ((0, 0), (0, audio2.shape[1] - audio1.shape[1])))
            else:
                audio2 = np.pad(audio2, ((0, 0), (0, audio1.shape[1] - audio2.shape[1])))
        
        # Calculate differences for each channel
        diff_left = audio1[0] - audio2[0]
        diff_right = audio1[1] - audio2[1]
        
        # Calculate RMS (Root Mean Square) values
        rms1_left = np.sqrt(np.mean(audio1[0]**2))
        rms1_right = np.sqrt(np.mean(audio1[1]**2))
        rms2_left = np.sqrt(np.mean(audio2[0]**2))
        rms2_right = np.sqrt(np.mean(audio2[1]**2))
        rms_diff_left = np.sqrt(np.mean(diff_left**2))
        rms_diff_right = np.sqrt(np.mean(diff_right**2))
        
        # Calculate overall RMS
        rms1_overall = np.sqrt(np.mean(audio1**2))
        rms_diff_overall = np.sqrt(np.mean((audio1 - audio2)**2))
        
        # Calculate difference percentages
        # Percentage relative to the first file's RMS energy
        if rms1_overall > 0:
            diff_percentage = (rms_diff_overall / rms1_overall) * 100
        else:
            diff_percentage = 0 if rms_diff_overall == 0 else 100
        
        # Per-channel percentages
        diff_pct_left = (rms_diff_left / rms1_left * 100) if rms1_left > 0 else 0
        diff_pct_right = (rms_diff_right / rms1_right * 100) if rms1_right > 0 else 0
        
        # Calculate correlation coefficient (similarity measure)
        corr_left = np.corrcoef(audio1[0], audio2[0])[0, 1]
        corr_right = np.corrcoef(audio1[1], audio2[1])[0, 1]
        
        return {
            'overall_difference_pct': diff_percentage,
            'left_channel_difference_pct': diff_pct_left,
            'right_channel_difference_pct': diff_pct_right,
            'left_channel_correlation': corr_left,
            'right_channel_correlation': corr_right,
            'sample_rate': sr1,
            'duration_seconds': audio1.shape[1] / sr1
        }
        
    except Exception as e:
        print(f"Error comparing audio files: {e}")
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python audio_compare.py <file1> <file2>")
        print("Example: python audio_compare.py song1.wav song2.mp3")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    print(f"Comparing audio files...")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print("-" * 50)
    
    results = compare_audio_files(file1, file2)
    
    if results:
        print(f"\nSample Rate: {results['sample_rate']} Hz")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print("\n" + "=" * 50)
        print(f"OVERALL DIFFERENCE: {results['overall_difference_pct']:.2f}%")
        print("=" * 50)
        print(f"\nLeft Channel Difference:  {results['left_channel_difference_pct']:.2f}%")
        print(f"Right Channel Difference: {results['right_channel_difference_pct']:.2f}%")
        print(f"\nLeft Channel Correlation:  {results['left_channel_correlation']:.4f}")
        print(f"Right Channel Correlation: {results['right_channel_correlation']:.4f}")
        print("\nNote: Higher correlation (closer to 1.0) means more similar")
        print("      0% difference means identical, 100% means completely different")

if __name__ == "__main__":
    main()