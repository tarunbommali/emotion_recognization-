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
PROCESSED_FEATURES_CSV_PATH = "eeg_features_emotions.csv"  # Make sure this CSV has the new multi-channel feature format
MODEL_PATH = "emotion_model_multichannel.joblib" # New model path for multi-channel model

# Define the EEG channels you want to process from the CSV
# Example: EEG_CHANNELS_TO_PROCESS = ['EEG.AF3', 'EEG.F7', 'EEG.AF4', 'EEG.F8']
# Ensure these channels exist in your uploaded CSVs and your training data CSV
EEG_CHANNELS_TO_PROCESS = ['EEG.AF3', 'EEG.F7'] # << --- UPDATE THIS WITH YOUR DESIRED CHANNELS


EEG_BANDS_DEFINITIONS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
# Dynamically create the expected feature columns for the model based on channels and bands
MODEL_FEATURE_COLUMNS = [
    f"{channel.replace('.', '')}{band}" # Ensure channel names are valid for DataFrame columns (e.g. EEGAF3delta)
    for channel in EEG_CHANNELS_TO_PROCESS
    for band in EEG_BANDS_DEFINITIONS.keys()
]

EPOCH_DURATION_SECONDS = 2
FILTER_LOWCUT_HZ = 0.5
FILTER_HIGHCUT_HZ = 45 # Max highcut for the filter
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1

# --- EEG Data Loading and Parsing ---
def load_and_parse_raw_eeg_data_from_upload(file_content_string, eeg_channel_names_list):
    print(f"Attempting to parse uploaded data for channels: {', '.join(eeg_channel_names_list)}")
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
            is_any_target_channel_present = any(ch_name.lower() in cells_in_line for ch_name in eeg_channel_names_list)
            common_headers = ["timestamp", "eeg.counter"] # Common general headers
            specific_channel_headers = [ch_name.lower() for ch_name in eeg_channel_names_list]

            # Check if it looks like a header row
            if (any(ch_hdr.lower() in cells_in_line for ch_hdr in common_headers) or \
                any(specific_ch.lower() in cells_in_line for specific_ch in specific_channel_headers)) and \
                len(cells_in_line) > max(1, len(eeg_channel_names_list)/2): # Heuristic: more than 1 col or half the num of channels
                
                header_like_cells_count = sum(1 for cell in cells_in_line if re.fullmatch(r'^[a-zA-Z0-9._\s-]+$', cell.strip()))
                if header_like_cells_count / len(cells_in_line) > 0.7: # If >70% cells look like headers
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
        raise ValueError(f"Pandas CSV parsing error after skipping {header_line_idx} lines: {e}. Check CSV format and header detection.")

    data_df.columns = [str(col).strip() for col in data_df.columns]
    
    all_channels_data = {}
    found_all_required_channels = True

    for eeg_channel_name in eeg_channel_names_list:
        actual_channel_name_in_df = None
        if eeg_channel_name in data_df.columns:
            actual_channel_name_in_df = eeg_channel_name
        else: # Try case-insensitive match
            for col_in_df in data_df.columns:
                if col_in_df.lower() == eeg_channel_name.lower():
                    actual_channel_name_in_df = col_in_df
                    print(f"ℹ️ Found channel as '{actual_channel_name_in_df}' (case-insensitive match for '{eeg_channel_name}').")
                    break
        
        if not actual_channel_name_in_df:
            print(f"❌ EEG Channel '{eeg_channel_name}' not found in CSV columns.")
            found_all_required_channels = False
            continue 

        channel_data_series = data_df[actual_channel_name_in_df]
        channel_data_numeric = pd.to_numeric(channel_data_series, errors='coerce')
        nan_count = channel_data_numeric.isnull().sum()
        if nan_count > 0:
            print(f"⚠️ Found {nan_count} non-numeric entries in channel '{actual_channel_name_in_df}', filled with 0.0.")
            channel_data_numeric = channel_data_numeric.fillna(0.0)
        
        all_channels_data[eeg_channel_name] = channel_data_numeric.to_numpy()
        print(f"✅ Parsed data for '{actual_channel_name_in_df}'. Samples: {len(all_channels_data[eeg_channel_name])}.")

    if not found_all_required_channels or len(all_channels_data) != len(eeg_channel_names_list):
        missing_channels = [ch for ch in eeg_channel_names_list if ch not in all_channels_data]
        available_cols_str = ", ".join(data_df.columns.tolist()[:15])
        raise ValueError(f"Could not find all required EEG channels. Missing: {missing_channels}. Available columns start with: {available_cols_str}... Required: {eeg_channel_names_list}")

    print(f"✅ Parsed data for all {len(all_channels_data)} requested channels. Rate: {sampling_rate} Hz.")
    return all_channels_data, sampling_rate

# --- EEG Signal Processing Functions ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure high is slightly less than 1.0 if it's at Nyquist, as some sosfiltfilt versions might warn
    if high >= 1.0:
        high = 1.0 - 1e-6 
    if low <= 0: # Lowcut must be > 0
        low = 1e-6 
    if low >= high:
        raise ValueError(f"Filter lowcut ({low*nyq}Hz) must be less than highcut ({high*nyq}Hz). Adjusted low={low}, high={high}")

    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def calculate_band_powers_from_psd(psd, freqs, bands):
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs < high_freq) # Use < for high_freq to avoid overlap
        if np.sum(idx_band) > 0:
            band_power = np.mean(psd[idx_band]) 
        else:
            band_power = 0.0 
        band_powers[band_name] = band_power
    return band_powers

def extract_features_from_eeg_data(
    all_channels_eeg_data, # Dict: {channel_name: np.array}
    fs,
    epoch_duration,
    global_lowcut, # Renamed for clarity
    global_highcut, # Renamed for clarity
    bands_def,
    target_channels_order, 
    band_names_order 
    ):
    
    if not all_channels_eeg_data or fs is None:
        print("❌ Cannot process EEG: Data or sampling rate is None.")
        return np.array([]), []

    print(f"\n--- Processing EEG Data for channels: {', '.join(target_channels_order)} ---")
    
    min_length = min(len(data) for data in all_channels_eeg_data.values())
    if min_length < fs * epoch_duration:
        print(f"⚠️ Data length ({min_length}) for at least one channel is less than one epoch ({fs * epoch_duration} samples). No features can be extracted.")
        return np.array([]), []

    # Adjust highcut for filtering based on Nyquist frequency
    current_filter_highcut = global_highcut
    if current_filter_highcut >= fs / 2.0:
        original_highcut_val = current_filter_highcut
        current_filter_highcut = (fs / 2.0) - 1e-3 
        print(f"⚠️ Global filter highcut ({original_highcut_val}Hz) adjusted to {current_filter_highcut:.3f}Hz for sampling rate ({fs}Hz).")
        if global_lowcut >= current_filter_highcut:
            raise ValueError(f"Global lowcut ({global_lowcut}Hz) is >= adjusted highcut ({current_filter_highcut}Hz). Cannot create filter.")

    filtered_channels_data = {}
    for channel_name, eeg_data in all_channels_eeg_data.items():
        if channel_name not in target_channels_order: 
            continue
        print(f"Applying band-pass filter ({global_lowcut}-{current_filter_highcut} Hz) for {channel_name}...")
        try:
            filtered_channels_data[channel_name] = butter_bandpass_filter(eeg_data[:min_length], global_lowcut, current_filter_highcut, fs)
        except ValueError as e:
            print(f"❌ Error during filtering for channel {channel_name}: {e}")
            return np.array([]), [] 

    epoch_length_samples = int(epoch_duration * fs)
    if epoch_length_samples == 0: # Should not happen if min_length check passed
        print(f"❌ Epoch length in samples is zero. Cannot segment.")
        return np.array([]), []
        
    num_epochs = min_length // epoch_length_samples
    if num_epochs == 0:
        print(f"⚠️ Data too short for any full epochs after potential truncation. No features.")
        return np.array([]), []
    
    print(f"Segmenting data into {num_epochs} epochs of {epoch_duration}s each.")
    
    all_epochs_combined_features = []
    all_epochs_plot_data_list = []

    for i in range(num_epochs):
        current_epoch_feature_vector_parts = []
        
        for channel_name in target_channels_order: 
            if channel_name not in filtered_channels_data:
                print(f"⚠️ Channel {channel_name} missing in filtered data for epoch {i+1}. Appending zeros.")
                current_epoch_feature_vector_parts.extend([0.0] * len(band_names_order))
                continue

            epoch_channel_data = filtered_channels_data[channel_name][i * epoch_length_samples : (i + 1) * epoch_length_samples]
            
            nperseg_val = min(epoch_length_samples, 256 if fs >= 128 else int(fs*2)) # Welch segment length
            nperseg_val = min(nperseg_val, len(epoch_channel_data)) 
            if nperseg_val <= 0 or len(epoch_channel_data) < nperseg_val : # Ensure enough data for Welch
                print(f"Epoch {i+1}, Channel {channel_name} has insufficient data ({len(epoch_channel_data)} samples) for nperseg={nperseg_val}. Appending zeros.")
                current_epoch_feature_vector_parts.extend([0.0] * len(band_names_order))
                continue
            
            try:
                freqs, psd = welch(epoch_channel_data, fs=fs, nperseg=nperseg_val, scaling='density')
                # Collect plot data for first, middle, last epoch for this channel
                if i == 0 or (num_epochs > 2 and i == num_epochs // 2) or i == num_epochs - 1:
                    all_epochs_plot_data_list.append({
                        "epoch_number": i + 1, 
                        "channel_name": channel_name,
                        "freqs": freqs.tolist(), 
                        "psd": psd.tolist(), 
                        "bands": bands_def # Send band definitions for client-side annotation if needed
                    })
            except ValueError as e:
                print(f"Error during Welch for epoch {i+1}, Channel {channel_name}: {e}. Appending zeros.")
                current_epoch_feature_vector_parts.extend([0.0] * len(band_names_order)) 
                continue

            band_powers = calculate_band_powers_from_psd(psd, freqs, bands_def)
            channel_features = [band_powers.get(band_name, 0.0) for band_name in band_names_order]
            current_epoch_feature_vector_parts.extend(channel_features)

        if len(current_epoch_feature_vector_parts) == len(MODEL_FEATURE_COLUMNS):
            all_epochs_combined_features.append(current_epoch_feature_vector_parts)
            if i < 1 or i == num_epochs -1: # Print for first and last epoch
                print(f"   Epoch {i+1}/{num_epochs} - Combined Features (first 3 of {len(current_epoch_feature_vector_parts)}): {[f'{x:.3e}' for x in current_epoch_feature_vector_parts[:3]]}...")
        else:
            print(f"   Epoch {i+1}/{num_epochs} - Feature vector length mismatch. Expected {len(MODEL_FEATURE_COLUMNS)}, got {len(current_epoch_feature_vector_parts)}. Skipping epoch.")

    if not all_epochs_combined_features:
        print("❌ No combined features extracted across channels.")
        return np.array([]), all_epochs_plot_data_list
        
    print(f"✅ Combined features extracted for {len(all_epochs_combined_features)} epochs.")
    return np.array(all_epochs_combined_features), all_epochs_plot_data_list

# --- Model Loading and Training ---
def get_or_train_model():
    global trained_model 
    if trained_model is not None:
        return trained_model
    try:
        trained_model = joblib.load(MODEL_PATH)
        print(f"✅ Pre-trained multi-channel model loaded from {MODEL_PATH}")
        if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != len(MODEL_FEATURE_COLUMNS):
            print(f"⚠️ CRITICAL WARNING: Loaded model expects {trained_model.n_features_in_} features, but current config (EEG_CHANNELS_TO_PROCESS) expects {len(MODEL_FEATURE_COLUMNS)} features.")
            print(f"Model expected features based on: (check training script). Current config features: {MODEL_FEATURE_COLUMNS}")
            # Decide if this should be a fatal error for the app
        return trained_model
    except FileNotFoundError:
        print(f"⚠️ Model file '{MODEL_PATH}' not found. Attempting to train a new model...")
        try:
            print(f"Attempting to read training data from: {PROCESSED_FEATURES_CSV_PATH}")
            data = pd.read_csv(PROCESSED_FEATURES_CSV_PATH)
            
            expected_cols_in_csv = MODEL_FEATURE_COLUMNS + ['emotion']
            missing_cols = [f_col for f_col in expected_cols_in_csv if f_col not in data.columns]
            if missing_cols:
                raise ValueError(f"Training CSV '{PROCESSED_FEATURES_CSV_PATH}' is missing required columns. Missing: {missing_cols}. Expected columns based on current EEG_CHANNELS_TO_PROCESS: {MODEL_FEATURE_COLUMNS} and 'emotion'.")
            
            X = data[MODEL_FEATURE_COLUMNS]
            y = data['emotion'].astype(str) # Ensure emotion labels are strings
            
            unique_emotions = y.unique()
            print(f"Training with emotions: {unique_emotions.tolist()} from column 'emotion'.")
            if len(X) < 10 or len(unique_emotions) < 2:
                raise ValueError(f"Not enough data or classes for training. Samples: {len(X)}, Unique Emotion Classes: {len(unique_emotions)}. Emotions found: {unique_emotions.tolist()}")

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            print(f"✅ New multi-channel model trained and saved to {MODEL_PATH}")
            trained_model = model 
            return trained_model
        except FileNotFoundError:
            print(f"❌ ERROR: Training CSV '{PROCESSED_FEATURES_CSV_PATH}' not found. Cannot train new model.")
            print(f"Please ensure it exists and has columns like: '{MODEL_FEATURE_COLUMNS[0]}', ..., 'emotion'.")
            return None
        except Exception as e:
            print(f"❌ ERROR during new model training: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"❌ ERROR loading existing model from '{MODEL_PATH}': {e}")
        traceback.print_exc()
        return None

# --- Flask App Setup ---
app = Flask(__name__) # Corrected: _name_ to __name__
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 
CORS(app, resources={r"/predict": {"origins": "*"}}) # Allow all origins for simplicity in dev

trained_model = None 
get_or_train_model() # Load or train model at startup

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_emotions_endpoint():
    global trained_model # Ensure we're using the global model instance
    if trained_model is None:
        print("Model not loaded at startup, attempting to load/train again...")
        get_or_train_model() 
        if trained_model is None:
             return jsonify({"error": "Model is not available. Server failed to load or train the model."}), 500

    if 'eeg_file' not in request.files:
        return jsonify({"error": "No EEG file ('eeg_file') provided in the request."}), 400
    file = request.files['eeg_file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        print(f"Received file: {file.filename}")
        file_content_string = file.read().decode('utf-8', errors='ignore')
        
        raw_eeg_signals_dict, sampling_rate = load_and_parse_raw_eeg_data_from_upload(
            file_content_string, EEG_CHANNELS_TO_PROCESS
        )
        
        extracted_features, plot_data_all_channels = extract_features_from_eeg_data(
            raw_eeg_signals_dict,
            fs=sampling_rate,
            epoch_duration=EPOCH_DURATION_SECONDS,
            global_lowcut=FILTER_LOWCUT_HZ,
            global_highcut=FILTER_HIGHCUT_HZ, 
            bands_def=EEG_BANDS_DEFINITIONS,
            target_channels_order=EEG_CHANNELS_TO_PROCESS,
            band_names_order=list(EEG_BANDS_DEFINITIONS.keys())
        )
        
        if extracted_features.size == 0:
            error_message = "No features were extracted from the EEG data. Cannot make predictions."
            if not plot_data_all_channels: # if plot data also failed
                error_message += " Plot data generation also failed."
            else: # if some plot data exists (e.g. from partial processing)
                 error_message += " Some plot data might be available for inspection."
            return jsonify({"error": error_message, "plot_data": plot_data_all_channels or []}), 400

        predict_df = pd.DataFrame(extracted_features, columns=MODEL_FEATURE_COLUMNS)
        
        # Sanity check for model compatibility if possible (already done at load time ideally)
        if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != predict_df.shape[1]:
            return jsonify({
                "error": f"Model feature mismatch. Model expects {trained_model.n_features_in_} features, but got {predict_df.shape[1]} from processed data. Check EEG_CHANNELS_TO_PROCESS and model training."
            }), 500

        predictions = trained_model.predict(predict_df) 
        predictions_list = [str(p) for p in predictions.tolist()] 

        print(f"✅ Predictions generated: {predictions_list[:5]}... (Total: {len(predictions_list)})")
        print(f"Plot data generated for {len(plot_data_all_channels)} channel-epoch segments.")
        
        return jsonify({
            "predictions": predictions_list, 
            "plot_data": plot_data_all_channels,
            "processed_channels": EEG_CHANNELS_TO_PROCESS,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "emotions_legend": trained_model.classes_.tolist() if hasattr(trained_model, 'classes_') else "N/A (model has no classes_ attribute)"
            })

    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__': # Corrected: _name_ to __name__ and _main_ to '__main__'
    print("--------------------------------------------------------------------")
    print("Starting Flask server for Multi-Channel EEG Emotion Prediction")
    print(f"Configured to process channels: {EEG_CHANNELS_TO_PROCESS}")
    print(f"Model will expect {len(MODEL_FEATURE_COLUMNS)} features based on these channels.")
    print(f"Feature names example: {MODEL_FEATURE_COLUMNS[:2] if MODEL_FEATURE_COLUMNS else 'N/A'}...")
    print(f"Attempting to use/train model stored at: {MODEL_PATH}")
    print(f"Training data CSV expected at: {PROCESSED_FEATURES_CSV_PATH}")
    print("--------------------------------------------------------------------")
    
    if trained_model is None:
        print("🔴 CRITICAL: Model could not be loaded or trained at startup. Predictions will fail until a model is available or the issue is resolved.")
    else:
        if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != len(MODEL_FEATURE_COLUMNS):
             print(f"🔴 CRITICAL WARNING AT STARTUP: Loaded model expects {trained_model.n_features_in_} features, but current config is for {len(MODEL_FEATURE_COLUMNS)} features.")
             print("Ensure your saved model and current EEG_CHANNELS_TO_PROCESS setting are compatible, or retrain the model with current settings.")
        if hasattr(trained_model, 'classes_'):
            print(f"✅ Model loaded/trained successfully. It can predict these emotions: {trained_model.classes_.tolist()}")
        else:
            print("ℹ️ Model loaded/trained, but 'classes_' attribute not found (might be an issue or model type).")


    app.run(debug=True, host='0.0.0.0', port=5000)