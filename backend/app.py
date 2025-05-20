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
EEG_CHANNEL_TO_PROCESS = 'EEG.AF3' # Make sure this channel exists in your CSVs
EPOCH_DURATION_SECONDS = 2
FILTER_LOWCUT_HZ = 0.5
FILTER_HIGHCUT_HZ = 45
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1

# --- EEG Data Loading and Parsing ---
def load_and_parse_raw_eeg_data_from_upload(file_content_string, eeg_channel_name):
    """
    Loads raw EEG data from an uploaded CSV file string.
    Attempts to auto-detect the header row and sampling rate.
    """
    print(f"Attempting to parse uploaded data for channel: {eeg_channel_name}")
    sampling_rate = None
    header_line_idx = -1
    lines = file_content_string.splitlines()
    max_lines_to_scan_metadata = min(len(lines), DEFAULT_METADATA_LINES_BEFORE_HEADER + 30)

    for i, line_text in enumerate(lines[:max_lines_to_scan_metadata]):
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
        if header_line_idx == -1:
            cells_in_line = [cell.strip().lower() for cell in line_text.split(',')]
            is_target_channel_present = any(eeg_channel_name.lower() == cell for cell in cells_in_line)
            common_headers = ["timestamp", "eeg.counter", eeg_channel_name.lower()]
            if (any(ch in cells_in_line for ch in common_headers) or is_target_channel_present) and len(cells_in_line) > 3:
                header_like_cells = sum([bool(re.fullmatch(r'^[a-zA-Z0-9._-]+$', cell)) for cell in cells_in_line[:15]])
                if header_like_cells > min(3, len(cells_in_line) -1 ):
                    header_line_idx = i
                    print(f"💡 Tentative header found at line index {i}. Content: {line_text[:100]}...")
    if sampling_rate is None:
        print("⚠️ Sampling rate not detected. Using default of 128 Hz. Accuracy may be affected.")
        sampling_rate = 128
    if header_line_idx == -1:
        print(f"⚠️ Header not auto-detected. Using fallback skip of {DEFAULT_METADATA_LINES_BEFORE_HEADER} line(s).")
        header_line_idx = DEFAULT_METADATA_LINES_BEFORE_HEADER
    if header_line_idx >= len(lines):
         raise ValueError("Not enough lines in file for fallback header skip.")
    print(f"Reading CSV data, skipping {header_line_idx} lines for header.")
    csv_data_io = io.StringIO("\n".join(lines[header_line_idx:]))
    try:
        data_df = pd.read_csv(csv_data_io, low_memory=False)
    except Exception as e:
        raise ValueError(f"Pandas CSV parsing error after skipping {header_line_idx} lines: {e}")
    data_df.columns = [str(col).strip() for col in data_df.columns]
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
        available_cols_str = ", ".join(data_df.columns.tolist()[:10])
        raise ValueError(f"EEG Channel '{eeg_channel_name}' not found. Available columns start with: {available_cols_str}...")
    channel_data_series = data_df[actual_channel_name_in_df]
    channel_data_numeric = pd.to_numeric(channel_data_series, errors='coerce')
    nan_count = channel_data_numeric.isnull().sum()
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} non-numeric entries in channel '{actual_channel_name_in_df}', filled with 0.0.")
        channel_data_numeric = channel_data_numeric.fillna(0.0)
    print(f"✅ Parsed data for '{actual_channel_name_in_df}'. Samples: {len(channel_data_numeric)}. Rate: {sampling_rate} Hz.")
    return channel_data_numeric.to_numpy(), sampling_rate

# --- EEG Signal Processing Functions ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < 1 and 0 < high < 1 and low < high) :
        raise ValueError(f"Invalid normalized frequencies for filter: low={low}, high={high}. Must be 0 < low < high < 1.")
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
            band_power = 0.0
        band_powers[band_name] = band_power
    return band_powers

def process_eeg_channel_for_features(eeg_data, fs, epoch_duration, lowcut, highcut, bands_def, expected_features_order, channel_name):
    if eeg_data is None or fs is None:
        print("❌ Cannot process EEG: Data or sampling rate is None.")
        return np.array([]), []
    print(f"\n--- Processing EEG Channel Data for {channel_name} ---")
    if len(eeg_data) < fs * epoch_duration:
        print(f"⚠️ Data length ({len(eeg_data)}) < one epoch. No features/plots.")
        return np.array([]), []
    print(f"Applying band-pass filter ({lowcut}-{highcut} Hz)...")
    try:
        if highcut >= fs / 2.0:
            original_highcut = highcut
            highcut = (fs / 2.0) - 1.0
            print(f"⚠️ Filter highcut ({original_highcut}Hz) adjusted to {highcut:.2f}Hz for sampling rate ({fs}Hz).")
            if lowcut >= highcut:
                raise ValueError(f"Lowcut ({lowcut}Hz) is >= adjusted highcut ({highcut}Hz).")
        filtered_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs)
    except ValueError as e:
        print(f"❌ Error during filtering: {e}")
        return np.array([]), []
    epoch_length_samples = int(epoch_duration * fs)
    if epoch_length_samples == 0:
        print(f"❌ Epoch length in samples is zero. Cannot segment.")
        return np.array([]), []
    num_epochs = len(filtered_data) // epoch_length_samples
    if num_epochs == 0:
        print(f"⚠️ Data too short for full epochs. No features/plots.")
        return np.array([]), []
    print(f"Segmenting data into {num_epochs} epochs of {epoch_duration}s each.")
    all_epoch_features = []
    epoch_plot_data_list = []
    for i in range(num_epochs):
        epoch_data = filtered_data[i * epoch_length_samples : (i + 1) * epoch_length_samples]
        nperseg_val = min(epoch_length_samples, 256 if fs >= 128 else int(fs))
        nperseg_val = min(nperseg_val, len(epoch_data)) # Ensure nperseg not > data length
        if nperseg_val == 0:
            print(f"Epoch {i+1} has no data or nperseg is zero. Skipping.")
            continue
        try:
            freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg_val, scaling='density')
            if i == 0 or (num_epochs > 5 and i == num_epochs // 2) or i == num_epochs - 1:
                 epoch_plot_data_list.append({
                    "epoch_number": i + 1, "channel_name": channel_name,
                    "freqs": freqs.tolist(), "psd": psd.tolist(), "bands": bands_def
                })
        except ValueError as e:
            print(f"Error during Welch for epoch {i+1}: {e}. Skipping.")
            continue
        band_powers = calculate_band_powers_from_psd(psd, freqs, bands_def)
        current_epoch_feature_vector = [band_powers.get(feature_name, 0.0) for feature_name in expected_features_order]
        all_epoch_features.append(current_epoch_feature_vector)
        if i < 2 or i == num_epochs -1:
            print(f"   Epoch {i+1}/{num_epochs} - Features: {[f'{x:.3f}' for x in current_epoch_feature_vector]}")
    if not all_epoch_features:
        print("❌ No features extracted.")
        return np.array([]), []
    print(f"✅ Features extracted for {len(all_epoch_features)} epochs.")
    return np.array(all_epoch_features), epoch_plot_data_list

# --- Model Loading and Training ---
def get_or_train_model():
    global trained_model # Use the global variable defined below
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
                raise ValueError("CSV must contain 'emotion' and all expected feature columns for training.")
            X = data[EXPECTED_FEATURE_COLUMNS]
            y = data['emotion']
            if len(X) < 10 or y.nunique() < 2:
                raise ValueError(f"Not enough data/classes for training. Samples: {len(X)}, Classes: {y.nunique()}")
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            print(f"✅ New model trained and saved to {MODEL_PATH}")
            trained_model = model # Assign to the global variable
            return trained_model
        except FileNotFoundError:
            print(f"❌ ERROR: Training CSV '{PROCESSED_FEATURES_CSV_PATH}' not found. Cannot train.")
            return None
        except Exception as e:
            print(f"❌ ERROR during model training: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"❌ ERROR loading existing model: {e}")
        traceback.print_exc()
        return None

# --- Flask App Setup ---

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

trained_model = None # Initialize global model variable

# Load or train the model when the application starts
# This populates the global 'trained_model' variable
get_or_train_model()

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_emotions_endpoint():
    # The 'trained_model' global variable should be populated by the get_or_train_model() call at startup
    # You could also call get_or_train_model() here again, it will just return the loaded model
    if trained_model is None: # Check if model loading/training failed at startup
        return jsonify({"error": "Model not available. Server startup might have failed to load/train it."}), 500

    if 'eeg_file' not in request.files:
        return jsonify({"error": "No EEG file provided."}), 400
    file = request.files['eeg_file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        print(f"Received file: {file.filename}")
        file_content_string = file.read().decode('utf-8')
        raw_eeg_signal, sampling_rate = load_and_parse_raw_eeg_data_from_upload(
            file_content_string, EEG_CHANNEL_TO_PROCESS
        )
        if raw_eeg_signal is None or sampling_rate is None or len(raw_eeg_signal) == 0:
            return jsonify({"error": "Failed to parse valid EEG signal from file."}), 400
        
        current_filter_highcut = FILTER_HIGHCUT_HZ
        extracted_features, plot_data = process_eeg_channel_for_features(
            raw_eeg_signal, fs=sampling_rate, epoch_duration=EPOCH_DURATION_SECONDS,
            lowcut=FILTER_LOWCUT_HZ, highcut=current_filter_highcut, bands_def=EEG_BANDS_DEFINITIONS,
            expected_features_order=EXPECTED_FEATURE_COLUMNS, channel_name=EEG_CHANNEL_TO_PROCESS
        )
        if extracted_features.size == 0:
            return jsonify({"error": "No features extracted. Cannot make predictions or generate plots."}), 400

        predict_df = pd.DataFrame(extracted_features, columns=EXPECTED_FEATURE_COLUMNS)
        predictions = trained_model.predict(predict_df) # Use the globally loaded model
        predictions_list = [str(p) for p in predictions.tolist()]

        print(f"✅ Predictions: {predictions_list[:3]}... Plot data for {len(plot_data)} epochs.")
        return jsonify({"predictions": predictions_list, "plot_data": plot_data})
    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask server for EEG Emotion Prediction with Plotting Data...")
    # The model is already loaded or attempted to be trained by the get_or_train_model() call above.
    if trained_model is None:
         print("🔴 CRITICAL: Model could not be loaded or trained. Predictions will fail.")
    app.run(debug=True, host='0.0.0.0', port=5000)