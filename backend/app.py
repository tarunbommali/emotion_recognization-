import io
import re
import traceback
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.ensemble import RandomForestClassifier

# --- Configuration Constants ---
PROCESSED_FEATURES_CSV_PATH = "eeg_features_emotions.csv"
MODEL_PATH = "emotion_model.joblib"
EXPECTED_FEATURE_COLUMNS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
EEG_BANDS_DEFINITIONS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
# Default settings for EEG processing (can be overridden or made configurable)
EEG_CHANNEL_TO_PROCESS = 'EEG.AF3' # Make sure this channel exists in your CSVs
EPOCH_DURATION_SECONDS = 2
FILTER_LOWCUT_HZ = 0.5
FILTER_HIGHCUT_HZ = 45 # Will be adjusted if sampling rate is too low
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1 # Fallback if header detection fails

# --- EEG Processing Functions (Adapted from your original script) ---

def load_and_parse_raw_eeg_data_from_upload(file_content_string, eeg_channel_name):
    """
    Loads raw EEG data from an uploaded CSV file string.
    Attempts to auto-detect the header row and sampling rate.
    """
    print(f"Attempting to parse uploaded data for channel: {eeg_channel_name}")
    sampling_rate = None
    header_line_idx = -1
    lines = file_content_string.splitlines()

    # 1. Attempt to detect sampling rate and header from initial lines
    # Increased search window for metadata
    max_lines_to_scan_metadata = min(len(lines), DEFAULT_METADATA_LINES_BEFORE_HEADER + 30)

    for i, line_text in enumerate(lines[:max_lines_to_scan_metadata]):
        # Sampling rate detection (example patterns)
        if sampling_rate is None:
            if "sampling rate:eeg_" in line_text.lower():
                try:
                    rate_str_part = line_text.lower().split("sampling rate:eeg_")[1]
                    num_str = "".join(filter(lambda char_s: char_s.isdigit() or char_s == '.', rate_str_part.split()[0]))
                    if num_str:
                        sampling_rate = int(float(num_str))
                        print(f"💡 Detected EEG Sampling Rate (Pattern 1): {sampling_rate} Hz")
                except Exception as e:
                    print(f"⚠️ Could not parse sampling rate from line (Pattern 1): '{line_text.strip()}'. Error: {e}")
            elif "samplingrate" in line_text.lower() and "eeg" in line_text.lower():
                match = re.search(r'eeg(?:_|\s*)(\d+)', line_text.lower())
                if match:
                    sampling_rate = int(match.group(1))
                    print(f"💡 Detected EEG Sampling Rate (Pattern 2): {sampling_rate} Hz")

        # Header detection
        if header_line_idx == -1:
            cells_in_line = [cell.strip().lower() for cell in line_text.split(',')]
            is_target_channel_present = any(eeg_channel_name.lower() == cell for cell in cells_in_line)
            common_headers = ["timestamp", "eeg.counter", eeg_channel_name.lower()]
            
            # Check if enough common headers or the target channel is present, and many cells look like headers
            if (any(ch in cells_in_line for ch in common_headers) or is_target_channel_present) and len(cells_in_line) > 3:
                # Heuristic: count cells that look like typical column headers (alphanumeric, dots, underscores)
                header_like_cells = sum([bool(re.fullmatch(r'^[a-zA-Z0-9._-]+$', cell)) for cell in cells_in_line[:15]])
                if header_like_cells > min(3, len(cells_in_line) -1 ): # Require at least a few header-like cells
                    header_line_idx = i
                    print(f"💡 Tentative header found at line index {i} (0-indexed). Content: {line_text[:100]}...")


    if sampling_rate is None:
        print("⚠️ Sampling rate not detected. Using default of 128 Hz. Accuracy may be affected.")
        sampling_rate = 128 # Fallback sampling rate

    if header_line_idx == -1:
        print(f"⚠️ Header containing '{eeg_channel_name}' or common terms not auto-detected. "
              f"Using fallback skip of {DEFAULT_METADATA_LINES_BEFORE_HEADER} line(s).")
        header_line_idx = DEFAULT_METADATA_LINES_BEFORE_HEADER
        if header_line_idx >= len(lines):
             raise ValueError("Not enough lines in file for fallback header skip.")


    print(f"Reading CSV data, skipping {header_line_idx} lines for header.")
    csv_data_io = io.StringIO("\n".join(lines[header_line_idx:]))
    
    try:
        data_df = pd.read_csv(csv_data_io, low_memory=False)
    except Exception as e:
        raise ValueError(f"Pandas CSV parsing error after skipping {header_line_idx} lines: {e}")

    data_df.columns = [str(col).strip() for col in data_df.columns] # Ensure columns are strings and stripped

    # Case-insensitive channel search
    actual_channel_name_in_df = None
    if eeg_channel_name in data_df.columns:
        actual_channel_name_in_df = eeg_channel_name
    else:
        for col_in_df in data_df.columns:
            if col_in_df.lower() == eeg_channel_name.lower():
                actual_channel_name_in_df = col_in_df
                print(f"ℹ️ Found channel as '{actual_channel_name_in_df}' (case-insensitive match for '{eeg_channel_name}').")
                break
    
    if not actual_channel_name_in_df:
        available_cols_str = ", ".join(data_df.columns.tolist()[:10]) # Show first 10
        raise ValueError(f"EEG Channel '{eeg_channel_name}' not found in the CSV columns after header processing. "
                         f"Available columns start with: {available_cols_str}...")

    channel_data_series = data_df[actual_channel_name_in_df]
    channel_data_numeric = pd.to_numeric(channel_data_series, errors='coerce')
    
    nan_count = channel_data_numeric.isnull().sum()
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} non-numeric entries in channel '{actual_channel_name_in_df}', converted to NaN and filled with 0.0.")
        channel_data_numeric = channel_data_numeric.fillna(0.0)
    
    print(f"✅ Successfully parsed data for channel '{actual_channel_name_in_df}'. Samples: {len(channel_data_numeric)}. Sampling Rate: {sampling_rate} Hz.")
    return channel_data_numeric.to_numpy(), sampling_rate


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low >= 1.0 or high >= 1.0 or low <=0 or high <=0 : # Basic check for valid normalized frequencies
        raise ValueError(f"Invalid normalized frequencies for filter: low={low}, high={high}. Nyquist={nyq}, fs={fs}, lowcut={lowcut}, highcut={highcut}")
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def calculate_band_powers_from_psd(psd, freqs, bands):
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        if np.sum(idx_band) > 0:
            band_power = np.mean(psd[idx_band])
        else:
            band_power = 0.0 # Ensure float
        band_powers[band_name] = band_power
    return band_powers

def process_eeg_channel_for_features(eeg_data, fs, epoch_duration, lowcut, highcut, bands_def, expected_features_order):
    if eeg_data is None or fs is None:
        print("❌ Cannot process EEG channel: Data or sampling rate is None.")
        return np.array([])

    print(f"Applying band-pass filter ({lowcut}-{highcut} Hz) to {len(eeg_data)} samples at {fs} Hz...")
    try:
        # Adjust highcut if it's too close to Nyquist frequency
        if highcut >= fs / 2.0:
            original_highcut = highcut
            highcut = (fs / 2.0) - 1 # Adjust to be safely below Nyquist
            print(f"⚠️ Filter highcut ({original_highcut} Hz) was at or above Nyquist frequency for sampling rate ({fs} Hz). Adjusted to {highcut:.2f} Hz.")
            if lowcut >= highcut: # Check if lowcut is now problematic
                 raise ValueError(f"Lowcut ({lowcut} Hz) is now greater than or equal to adjusted highcut ({highcut} Hz). Cannot proceed.")

        filtered_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs)
    except ValueError as e:
        print(f"❌ Error during filtering: {e}")
        return np.array([])


    epoch_length_samples = int(epoch_duration * fs)
    if epoch_length_samples == 0:
        print(f"❌ Epoch length in samples is zero (duration: {epoch_duration}s, fs: {fs}Hz). Cannot segment.")
        return np.array([])
        
    num_epochs = len(filtered_data) // epoch_length_samples
    
    if num_epochs == 0:
        print(f"⚠️ Data too short ({len(filtered_data)} samples) for any full epochs of {epoch_length_samples} samples. No features extracted.")
        return np.array([])
        
    print(f"Segmenting data into {num_epochs} epochs of {epoch_duration}s each.")
    all_epoch_features = []

    for i in range(num_epochs):
        epoch_data = filtered_data[i * epoch_length_samples : (i + 1) * epoch_length_samples]
        
        # nperseg_val for Welch: use full epoch if shorter than 256, else 256 (common default)
        # Ensure nperseg_val does not exceed epoch_length_samples.
        nperseg_val = min(epoch_length_samples, 256 if fs >= 128 else int(fs)) # Heuristic for nperseg
        if nperseg_val == 0: # Should not happen if epoch_length_samples > 0
            print(f"Error: nperseg_val is 0 for epoch {i}. Skipping.")
            continue
        if nperseg_val > len(epoch_data): # Safety check for Welch
            nperseg_val = len(epoch_data)


        try:
            freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg_val, scaling='density')
        except ValueError as e:
            print(f"Error during Welch calculation for epoch {i+1}: {e}. Length: {len(epoch_data)}, nperseg: {nperseg_val}. Skipping epoch.")
            continue

        band_powers = calculate_band_powers_from_psd(psd, freqs, bands_def)
        current_epoch_feature_vector = [band_powers.get(feature_name, 0.0) for feature_name in expected_features_order]
        all_epoch_features.append(current_epoch_feature_vector)
        
        if i < 2 or i == num_epochs -1 : # Log first few and last
            print(f"   Epoch {i+1}/{num_epochs} - Features (rounded): {[f'{x:.3f}' for x in current_epoch_feature_vector]}")

    if not all_epoch_features:
        print("❌ No features were extracted after processing all epochs.")
        return np.array([])

    print(f"✅ Feature extraction complete. Extracted features for {len(all_epoch_features)} epochs.")
    return np.array(all_epoch_features)


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allows requests from frontend (different port)
trained_model = None

def get_or_train_model():
    global trained_model
    if trained_model is not None:
        return trained_model
    
    try:
        trained_model = joblib.load(MODEL_PATH)
        print(f"✅ Pre-trained model loaded from {MODEL_PATH}")
        return trained_model
    except FileNotFoundError:
        print(f"⚠️ Model file '{MODEL_PATH}' not found. Attempting to train a new model...")
        try:
            data = pd.read_csv(PROCESSED_FEATURES_CSV_PATH)
            if 'emotion' not in data.columns or not all(f in data.columns for f in EXPECTED_FEATURE_COLUMNS):
                raise ValueError("CSV must contain 'emotion' column and all expected feature columns for training.")
            
            X = data[EXPECTED_FEATURE_COLUMNS]
            y = data['emotion']
            
            if len(X) < 10 or y.nunique() < 2: # Basic check for sufficient data
                raise ValueError(f"Not enough data or unique classes to train. Samples: {len(X)}, Classes: {y.nunique()}")

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            print(f"✅ New model trained and saved to {MODEL_PATH}")
            trained_model = model
            return trained_model
        except FileNotFoundError:
            print(f"❌ ERROR: Training feature file '{PROCESSED_FEATURES_CSV_PATH}' not found. Cannot train model.")
            return None
        except Exception as e:
            print(f"❌ ERROR during model training: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        traceback.print_exc()
        return None

# Load or train model on startup
get_or_train_model()

@app.route('/predict', methods=['POST'])
def predict_emotions_endpoint():
    model = get_or_train_model()
    if model is None:
        return jsonify({"error": "Model not available. Please check server logs."}), 500

    if 'eeg_file' not in request.files:
        return jsonify({"error": "No EEG file provided in the request."}), 400

    file = request.files['eeg_file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload."}), 400

    try:
        print(f"Received file: {file.filename}")
        file_content_string = file.read().decode('utf-8') # Assuming UTF-8 encoded CSV

        # Process the raw EEG data from the uploaded file content
        raw_eeg_signal, sampling_rate = load_and_parse_raw_eeg_data_from_upload(
            file_content_string,
            EEG_CHANNEL_TO_PROCESS
        )

        if raw_eeg_signal is None or sampling_rate is None or len(raw_eeg_signal) == 0:
            return jsonify({"error": "Failed to parse valid EEG signal data from the uploaded file."}), 400
        
        current_filter_highcut = FILTER_HIGHCUT_HZ # Use configured highcut

        extracted_features_from_raw = process_eeg_channel_for_features(
            raw_eeg_signal,
            fs=sampling_rate,
            epoch_duration=EPOCH_DURATION_SECONDS,
            lowcut=FILTER_LOWCUT_HZ,
            highcut=current_filter_highcut,
            bands_def=EEG_BANDS_DEFINITIONS,
            expected_features_order=EXPECTED_FEATURE_COLUMNS
        )

        if extracted_features_from_raw.size == 0:
            return jsonify({"error": "No features could be extracted from the EEG data. The file might be too short or data unsuitable."}), 400

        predict_df = pd.DataFrame(extracted_features_from_raw, columns=EXPECTED_FEATURE_COLUMNS)
        predictions = model.predict(predict_df)
        predictions_list = [str(p) for p in predictions.tolist()] # Ensure serializable list of strings

        print(f"✅ Predictions generated: {predictions_list[:5]}...") # Log first few
        return jsonify({"predictions": predictions_list})

    except ValueError as ve: # Catch specific parsing or processing errors
        print(f"ValueError during prediction: {ve}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred. " + str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server for EEG Emotion Prediction...")
    # Make sure model is loaded/trained before starting
    if get_or_train_model() is None:
        print("🔴 CRITICAL: Model could not be loaded or trained. Predictions will fail.")
    app.run(debug=True, host='0.0.0.0', port=5000) # Runs on http://localhost:5000