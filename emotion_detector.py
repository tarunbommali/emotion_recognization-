# 🔧 STEP 0: Install Required Libraries
# pip install numpy pandas matplotlib scikit-learn scipy

# 📦 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.signal import welch, butter, sosfiltfilt
import re # Make sure re is imported
import warnings # To control warnings if needed, though direct fix is better

# --- Configuration for Model Training ---
PROCESSED_FEATURES_CSV_PATH = "eeg_features_emotions.csv" # Used for training
EXPECTED_FEATURE_COLUMNS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
EEG_BANDS_DEFINITIONS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45) # Adjusted gamma upper limit for 128Hz sampling rate
}

# --- Configuration for Raw EEG File Processing ---
# 🚩 SET THESE PATHS AND PARAMETERS FOR YOUR RAW EEG FILE
RAW_EEG_CSV_PATH = "M1_EPOCX_266276_2025.01.30T12.06.14+05.30.md.csv" # Replace with your raw EEG file path
EEG_CHANNEL_TO_PROCESS = 'EEG.AF3' # Example channel, choose one from your file
EPOCH_DURATION_SECONDS = 2  # Duration of each epoch for analysis
FILTER_LOWCUT_HZ = 0.5
FILTER_HIGHCUT_HZ = 45 # Keep below Nyquist frequency (sampling_rate / 2)
# Default number of metadata lines to skip if header auto-detection fails
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1


print("*********************************************************************")
print("*** EEG Emotion Recognition - Raw File Processing Demo ***")
print("*********************************************************************\n")

def train_emotion_model(csv_path, feature_column_names):
    """
    Loads data from PROCESSED_FEATURES_CSV_PATH, trains a RandomForest model.
    """
    print("\n--- STAGE 1: TRAINING EMOTION RECOGNITION MODEL ---")
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded '{csv_path}' for training.")
    except FileNotFoundError:
        print(f"❌ ERROR: Training feature file '{csv_path}' not found.")
        print("Please ensure this CSV file with pre-extracted features and emotions exists.")
        return None

    if 'emotion' not in data.columns:
        print(f"❌ ERROR: CSV must contain 'emotion' column for labels.")
        return None
    for feature_col in feature_column_names:
        if feature_col not in data.columns:
            print(f"❌ ERROR: Expected feature column '{feature_col}' not found in '{csv_path}'.")
            return None

    X = data[feature_column_names]
    y = data['emotion']
    print(f"Training with {len(X)} samples and {len(feature_column_names)} features.")

    if len(X) < 10 or y.nunique() < 2:
        print("❌ ERROR: Not enough data or unique classes in the training CSV to train a meaningful model.")
        print(f"   Samples: {len(X)}, Unique Classes: {y.nunique()}")
        return None
    
    X_train, y_train = X,y # Training on all data for simplicity

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X_train, y_train)
        print("✅ Emotion recognition model trained successfully!")
    except Exception as e:
        print(f"❌ ERROR during model training: {e}")
        return None
    return model

def load_and_parse_raw_eeg_csv(filepath, eeg_channel_name, default_num_metadata_lines_before_header=1):
    """
    Loads raw EEG data from the specific Emotiv-like CSV format.
    Attempts to auto-detect the header row and sampling rate.
    Handles potential non-numeric data in the selected EEG channel.
    """
    print(f"\n--- Loading raw EEG data from: {filepath} ---")
    sampling_rate = None
    # channel_data = None # Removed as it's assigned later

    try:
        # 1. Attempt to detect sampling rate from initial lines
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= default_num_metadata_lines_before_header + 15: # Search a bit more
                    break
                if "sampling rate:eeg_" in line.lower():
                    try:
                        rate_str_part = line.lower().split("sampling rate:eeg_")[1]
                        num_str = ""
                        for char_s in rate_str_part:
                            if char_s.isdigit() or char_s == '.':
                                num_str += char_s
                            else:
                                break 
                        if num_str:
                            sampling_rate = int(float(num_str.strip()))
                            print(f"💡 Detected EEG Sampling Rate: {sampling_rate} Hz")
                        break 
                    except Exception as e:
                        print(f"⚠️ Could not parse sampling rate from line: '{line.strip()}'. Error: {e}")
                elif "samplingrate" in line.lower() and "eeg" in line.lower():
                    match = re.search(r'eeg(?:_|\s*)(\d+)', line.lower())
                    if match:
                        sampling_rate = int(match.group(1))
                        print(f"💡 Detected EEG Sampling Rate (generic): {sampling_rate} Hz")
                        break

        # 2. Attempt to find the actual header row index
        header_line_idx_in_file = -1  # 0-indexed
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line_text in enumerate(f):
                if i > default_num_metadata_lines_before_header + 20: # Increased search limit for header
                    break
                cells_in_line = [cell.strip().lower() for cell in line_text.split(',')]
                
                is_target_channel_present = any(eeg_channel_name.lower() == cell for cell in cells_in_line)

                if (is_target_channel_present or \
                    "timestamp" in cells_in_line or \
                    "eeg.counter" in cells_in_line) and len(cells_in_line) > 5 :
                    header_like_cells = sum([bool(re.fullmatch(r'^[a-zA-Z0-9._-]+$', cell)) for cell in cells_in_line[:15]]) 
                    if header_like_cells > 3: 
                        header_line_idx_in_file = i
                        break
        
        if header_line_idx_in_file == -1:
            print(f"⚠️ Header containing '{eeg_channel_name}' or 'Timestamp' not auto-detected. Using default skip of {default_num_metadata_lines_before_header} line(s).")
            header_line_idx_in_file = default_num_metadata_lines_before_header
        
        print(f"Reading CSV data, assuming header is on file line {header_line_idx_in_file + 1} (1-indexed), thus skipping {header_line_idx_in_file} lines.")
        
        # 3. Load CSV data using the determined header row
        data_df = pd.read_csv(filepath, skiprows=header_line_idx_in_file, low_memory=False, encoding='utf-8')

        data_df.columns = [col.strip() for col in data_df.columns]
        eeg_channel_name_to_find = eeg_channel_name.strip()

        if eeg_channel_name_to_find not in data_df.columns:
            found_channel_alt = None
            for col_in_df in data_df.columns:
                if col_in_df.lower() == eeg_channel_name_to_find.lower():
                    found_channel_alt = col_in_df
                    print(f"ℹ️ Found channel as '{found_channel_alt}' (case-insensitive match for '{eeg_channel_name_to_find}').")
                    eeg_channel_name_to_find = found_channel_alt
                    break
            if not found_channel_alt:
                print(f"❌ ERROR: EEG Channel '{eeg_channel_name_to_find}' not found in the CSV columns.")
                print(f"   Available columns (after stripping): {data_df.columns.tolist()}")
                return None, sampling_rate

        # 4. Extract and clean channel data
        channel_data_series = data_df[eeg_channel_name_to_find]
        
        channel_data_numeric = pd.to_numeric(channel_data_series, errors='coerce')
        
        nan_count = channel_data_numeric.isnull().sum()
        if nan_count > 0:
            print(f"⚠️ Found {nan_count} non-numeric entries in channel '{eeg_channel_name_to_find}', converted to NaN.")
            print(f"   Replacing NaNs with 0.0 for further processing.")
            channel_data_numeric = channel_data_numeric.fillna(0.0)
            if channel_data_numeric.isnull().sum() > 0:
                 print(f"❌ CRITICAL: Still NaNs after fillna. Problem with data in channel '{eeg_channel_name_to_find}'.")
                 return None, sampling_rate

        channel_data_np = channel_data_numeric.to_numpy() # Renamed to avoid conflict
        print(f"✅ Successfully loaded and cleaned data for channel '{eeg_channel_name_to_find}'. Samples: {len(channel_data_np)}")
        return channel_data_np, sampling_rate

    except FileNotFoundError:
        print(f"❌ ERROR: Raw EEG file '{filepath}' not found.")
        return None, None
    except Exception as e:
        print(f"❌ ERROR loading/parsing raw EEG file: {e}")
        import traceback
        traceback.print_exc() 
        return None, None
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def calculate_band_powers_from_psd(psd, freqs, bands):
    """Calculates mean power in specified frequency bands from PSD."""
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        if np.sum(idx_band) > 0: 
            band_power = np.mean(psd[idx_band])
        else:
            band_power = 0 
        band_powers[band_name] = band_power
    return band_powers

def process_eeg_channel_for_features(eeg_data, fs, epoch_duration, lowcut, highcut, bands_def, expected_features_order):
    """
    Processes raw EEG channel data to extract band power features for each epoch.
    """
    if eeg_data is None or fs is None:
        print("❌ Cannot process EEG channel: Data or sampling rate is None.")
        return np.array([]) 

    print(f"\n--- Processing EEG Channel Data ---")
    if len(eeg_data) < fs * epoch_duration: 
        print(f"⚠️ Data length ({len(eeg_data)} samples) is less than one epoch ({int(fs*epoch_duration)} samples). Cannot extract features.")
        return np.array([])

    print(f"Applying band-pass filter ({lowcut}-{highcut} Hz)...")
    filtered_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs)

    epoch_length_samples = int(epoch_duration * fs)
    num_epochs = len(filtered_data) // epoch_length_samples
    
    if num_epochs == 0:
        print(f"⚠️ Data too short after filtering for any full epochs. Filtered length: {len(filtered_data)}, Epoch samples: {epoch_length_samples}")
        return np.array([])
        
    print(f"Segmenting data into {num_epochs} epochs of {epoch_duration}s each.")

    all_epoch_features = []

    for i in range(num_epochs):
        epoch_data = filtered_data[i * epoch_length_samples : (i + 1) * epoch_length_samples]
        
        nperseg_val = min(epoch_length_samples, 256 if fs >=128 else int(fs)) 
        if nperseg_val == 0 : 
            print(f"Error: nperseg_val is 0 for epoch {i}. Skipping.")
            continue

        try:
            freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg_val, scaling='density')
        except ValueError as e:
            print(f"Error during Welch calculation for epoch {i+1}: {e}. Skipping epoch.")
            print(f"  Epoch data length: {len(epoch_data)}, nperseg: {nperseg_val}, fs: {fs}")
            continue 

        band_powers = calculate_band_powers_from_psd(psd, freqs, bands_def)
        
        current_epoch_feature_vector = [band_powers.get(feature_name, 0) for feature_name in expected_features_order]
        all_epoch_features.append(current_epoch_feature_vector)
        
        if i < 3 or i == num_epochs -1 : 
             print(f"  Epoch {i+1}/{num_epochs} - Extracted Features (rounded): {[f'{x:.2f}' for x in current_epoch_feature_vector]}")

    if not all_epoch_features: 
        print("❌ No features were extracted after processing all epochs.")
        return np.array([])

    print(f"✅ Feature extraction complete. Extracted features for {len(all_epoch_features)} epochs.")
    return np.array(all_epoch_features)


# --- Main Execution ---
if __name__ == "__main__":
    # STAGE 1: Train the emotion recognition model
    trained_model = train_emotion_model(PROCESSED_FEATURES_CSV_PATH, EXPECTED_FEATURE_COLUMNS)

    if trained_model:
        print("\n--- STAGE 2: PREDICTING EMOTIONS FROM RAW EEG FILE ---")
        raw_eeg_signal, sampling_rate = load_and_parse_raw_eeg_csv(
            RAW_EEG_CSV_PATH, 
            EEG_CHANNEL_TO_PROCESS,
            default_num_metadata_lines_before_header=DEFAULT_METADATA_LINES_BEFORE_HEADER
        )

        if raw_eeg_signal is not None and sampling_rate is not None:
            if sampling_rate < (2 * FILTER_HIGHCUT_HZ) : 
                print(f"⚠️ WARNING: Filter highcut ({FILTER_HIGHCUT_HZ} Hz) is too high for sampling rate ({sampling_rate} Hz). Adjusting highcut.")
                FILTER_HIGHCUT_HZ = (sampling_rate / 2.0) - 1 
                print(f"Adjusted filter highcut to {FILTER_HIGHCUT_HZ:.2f} Hz.")

            extracted_features_from_raw = process_eeg_channel_for_features(
                raw_eeg_signal,
                fs=sampling_rate,
                epoch_duration=EPOCH_DURATION_SECONDS,
                lowcut=FILTER_LOWCUT_HZ,
                highcut=FILTER_HIGHCUT_HZ,
                bands_def=EEG_BANDS_DEFINITIONS,
                expected_features_order=EXPECTED_FEATURE_COLUMNS
            )

            if extracted_features_from_raw.size > 0 and extracted_features_from_raw.shape[0] > 0 :
                print(f"\n--- Making predictions for {extracted_features_from_raw.shape[0]} epochs ---")
                
                # Convert NumPy array to DataFrame with feature names to avoid UserWarning
                predict_df = pd.DataFrame(extracted_features_from_raw, columns=EXPECTED_FEATURE_COLUMNS)
                predictions = trained_model.predict(predict_df) # Use DataFrame for prediction
                
                # If you want probabilities:
                # probabilities = trained_model.predict_proba(predict_df)

                for i, prediction in enumerate(predictions):
                    if i < 10 or i > len(predictions) - 6 :
                        print(f"Epoch {i+1}: Predicted Emotion = {prediction.upper()}")
                    elif i == 10:
                        print("      ...") 

                print("\n✅ Prediction from raw EEG file processing complete.")
            else:
                print("❌ No features extracted from raw EEG file, or features array is empty. Cannot make predictions.")
        else:
            print("❌ Failed to load or parse raw EEG data. Cannot proceed with prediction.")
    else:
        print("\n❌ Emotion model training failed. Cannot proceed with raw EEG file prediction.")

    print("\n*********************************************************************")
    print("*** End of EEG Emotion Recognition - Raw File Processing Demo ***")
    print("*********************************************************************")
