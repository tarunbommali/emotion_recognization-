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
import time # For creating time vector

# --- Configuration Constants ---
PROCESSED_FEATURES_CSV_PATH = "eeg_features_emotions.csv"
MODEL_PATH = "emotion_model_multichannel.joblib"
EEG_CHANNELS_TO_PROCESS = ['EEG.AF3', 'EEG.F7'] # Using 2 channels as currently configured

EEG_BANDS_DEFINITIONS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45) # Gamma is used for features, but not requested for specific plots
}

# Bands requested for specific plots by the user
REQUESTED_PLOT_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}


MODEL_FEATURE_COLUMNS = [
    f"{channel.replace('.', '')}{band}"
    for channel in EEG_CHANNELS_TO_PROCESS
    for band in EEG_BANDS_DEFINITIONS.keys() # Features from all defined bands
]

EPOCH_DURATION_SECONDS = 2
FILTER_LOWCUT_HZ = 0.5 # Overall filter for feature extraction
FILTER_HIGHCUT_HZ = 45 # Overall filter for feature extraction
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1

# --- EEG Data Loading and Parsing (remains the same) ---
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
            common_headers = ["timestamp", "eeg.counter"]
            specific_channel_headers = [ch_name.lower() for ch_name in eeg_channel_names_list]
            if (any(ch_hdr.lower() in cells_in_line for ch_hdr in common_headers) or \
                any(specific_ch.lower() in cells_in_line for specific_ch in specific_channel_headers)) and \
                len(cells_in_line) > max(1, len(eeg_channel_names_list)/2):
                header_like_cells_count = sum(1 for cell in cells_in_line if re.fullmatch(r'^[a-zA-Z0-9._\s-]+$', cell.strip()))
                if header_like_cells_count / len(cells_in_line) > 0.7:
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

    csv_data_io = io.StringIO("\n".join(lines[header_line_idx:]))
    try:
        data_df = pd.read_csv(csv_data_io, low_memory=False)
    except Exception as e:
        raise ValueError(f"Pandas CSV parsing error: {e}. Check CSV format and header detection.")
    data_df.columns = [str(col).strip() for col in data_df.columns]
    all_channels_data = {}
    found_all_required_channels = True
    for eeg_channel_name in eeg_channel_names_list:
        actual_channel_name_in_df = None
        if eeg_channel_name in data_df.columns:
            actual_channel_name_in_df = eeg_channel_name
        else:
            for col_in_df in data_df.columns:
                if col_in_df.lower() == eeg_channel_name.lower():
                    actual_channel_name_in_df = col_in_df
                    print(f"ℹ️ Found channel as '{actual_channel_name_in_df}' (case-insensitive for '{eeg_channel_name}').")
                    break
        if not actual_channel_name_in_df:
            print(f"❌ EEG Channel '{eeg_channel_name}' not found.")
            found_all_required_channels = False
            continue 
        channel_data_series = data_df[actual_channel_name_in_df]
        channel_data_numeric = pd.to_numeric(channel_data_series, errors='coerce').fillna(0.0)
        if channel_data_numeric.isnull().sum() > 0: # Should be handled by fillna(0.0)
             print(f"⚠️ Non-numeric entries in '{actual_channel_name_in_df}' were filled with 0.0.")
        all_channels_data[eeg_channel_name] = channel_data_numeric.to_numpy()
        print(f"✅ Parsed data for '{actual_channel_name_in_df}'. Samples: {len(all_channels_data[eeg_channel_name])}.")
    if not found_all_required_channels or len(all_channels_data) != len(eeg_channel_names_list):
        missing = [ch for ch in eeg_channel_names_list if ch not in all_channels_data]
        raise ValueError(f"Missing channels: {missing}. Available: {', '.join(data_df.columns[:15])}...")
    print(f"✅ Parsed data for all requested channels. Rate: {sampling_rate} Hz.")
    return all_channels_data, sampling_rate


# --- EEG Signal Processing Functions (butter_bandpass_filter is the same) ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0: high = 1.0 - 1e-6
    if low <= 0: low = 1e-6
    if low >= high:
        # Attempt to fix invalid range if highcut was drastically reduced due to low fs
        if highcut < lowcut: lowcut = highcut / 2
        low = lowcut / nyq
        if low >= high: # If still invalid, raise error
            raise ValueError(f"Filter lowcut ({lowcut}Hz) must be less than highcut ({highcut}Hz). Adjusted low={low}, high={high}")
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

def calculate_band_powers_from_psd(psd, freqs, bands): # Remains the same
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs < high_freq)
        band_powers[band_name] = np.mean(psd[idx_band]) if np.sum(idx_band) > 0 else 0.0
    return band_powers

def extract_features_and_plot_data( # Renamed for clarity and expanded functionality
    all_channels_raw_data, # Dict: {channel_name: np.array from CSV}
    fs,
    epoch_duration,
    overall_filter_lowcut, # For feature extraction PSD
    overall_filter_highcut, # For feature extraction PSD
    all_bands_definitions, # For feature extraction (e.g., including gamma)
    requested_plot_bands_defs, # For specific band-filtered time series plots (delta, theta, alpha, beta)
    target_channels_order,
    band_names_order_for_features # For ordering features for the model
    ):

    if not all_channels_raw_data or fs is None:
        return np.array([]), [], [] # features, psd_plot_data, band_amplitude_plot_data

    print(f"\n--- Extracting Features & Preparing Plot Data for channels: {', '.join(target_channels_order)} ---")
    
    min_length = min(len(data) for data in all_channels_raw_data.values())
    epoch_length_samples = int(epoch_duration * fs)
    num_epochs = min_length // epoch_length_samples

    if num_epochs == 0:
        print(f"⚠️ Data too short for any full epochs. No data processed.")
        return np.array([]), [], []

    print(f"Segmenting data into {num_epochs} epochs of {epoch_duration}s each ({epoch_length_samples} samples/epoch).")

    # Prepare overall filtered data for feature extraction PSD
    # This is a one-time filter over the whole signal for each channel before epoching for PSD
    overall_filtered_signal_for_psd = {}
    current_psd_filter_highcut = overall_filter_highcut
    if current_psd_filter_highcut >= fs / 2.0:
        current_psd_filter_highcut = (fs / 2.0) - 1e-3
        print(f"⚠️ Overall PSD filter highcut adjusted to {current_psd_filter_highcut:.3f}Hz for SR ({fs}Hz).")

    for ch_name in target_channels_order:
        try:
            overall_filtered_signal_for_psd[ch_name] = butter_bandpass_filter(
                all_channels_raw_data[ch_name][:min_length], # Use truncated signal
                overall_filter_lowcut,
                current_psd_filter_highcut,
                fs
            )
        except ValueError as e:
            print(f"❌ Error during overall signal filtering for PSD for channel {ch_name}: {e}")
            return np.array([]), [], []


    all_epochs_features = []
    psd_plot_data_for_frontend = [] # For 3D plot and general PSD
    band_amplitude_data_for_frontend = [] # For new Amplitude vs Time plots

    time_vector_full_signal = np.arange(0, min_length / fs, 1 / fs)[:min_length]

    # 1. Generate Band-Filtered Amplitude Data for the *first* epoch for requested bands
    # We'll do this for the first epoch as an example, you might want all epochs or a selected one.
    # If you want for all epochs, this loop structure would need to be different or data stored per epoch.
    # For simplicity, let's take the first epoch's band-filtered data.
    
    first_epoch_data_for_band_filtering = {}
    for ch_name in target_channels_order:
         first_epoch_data_for_band_filtering[ch_name] = all_channels_raw_data[ch_name][:epoch_length_samples]
    
    time_vector_epoch = time_vector_full_signal[:epoch_length_samples]

    for band_name, (low, high) in requested_plot_bands_defs.items():
        band_specific_traces = []
        for ch_name in target_channels_order:
            if ch_name in first_epoch_data_for_band_filtering:
                try:
                    # Filter the *raw data of the first epoch* for this specific band
                    band_filtered_amplitude = butter_bandpass_filter(
                        first_epoch_data_for_band_filtering[ch_name], low, high, fs, order=4 # Use a slightly lower order for less ringing if needed
                    )
                    band_specific_traces.append({
                        "channel_name": ch_name,
                        "amplitude": band_filtered_amplitude.tolist()
                    })
                except ValueError as e:
                    print(f"❌ Error filtering {ch_name} for {band_name} band amplitude: {e}")
                    band_specific_traces.append({"channel_name": ch_name, "amplitude": [0]*epoch_length_samples }) # Error case
            
        band_amplitude_data_for_frontend.append({
            "band_name": band_name,
            "time_vector": time_vector_epoch.tolist(),
            "traces": band_specific_traces # List of {channel_name, amplitude_data}
        })


    # 2. Process Features and Full PSD for selected epochs (for 3D plot & model prediction)
    for i in range(num_epochs):
        current_epoch_feature_vector_parts = []
        
        for ch_name in target_channels_order:
            # Use the overall filtered signal for PSD feature extraction
            epoch_channel_data_for_psd = overall_filtered_signal_for_psd[ch_name][i * epoch_length_samples : (i + 1) * epoch_length_samples]
            
            nperseg_val = min(epoch_length_samples, 256 if fs >= 128 else int(fs*2))
            nperseg_val = min(nperseg_val, len(epoch_channel_data_for_psd))

            if nperseg_val <= 0 or len(epoch_channel_data_for_psd) < nperseg_val:
                current_epoch_feature_vector_parts.extend([0.0] * len(band_names_order_for_features))
                continue
            
            try:
                freqs, psd = welch(epoch_channel_data_for_psd, fs=fs, nperseg=nperseg_val, scaling='density')
                # Collect plot data for 3D plot (first, middle, last epoch)
                if i == 0 or (num_epochs > 2 and i == num_epochs // 2) or i == num_epochs - 1:
                    psd_plot_data_for_frontend.append({
                        "epoch_number": i + 1, 
                        "channel_name": ch_name,
                        "freqs": freqs.tolist(), 
                        "psd": psd.tolist(), 
                        "bands": all_bands_definitions # Full band defs for annotations
                    })
            except ValueError as e:
                print(f"Error during Welch for epoch {i+1}, Channel {ch_name}: {e}. Appending zeros for features.")
                current_epoch_feature_vector_parts.extend([0.0] * len(band_names_order_for_features))
                # Add empty PSD data for this problematic case if needed by frontend, or skip
                if i == 0 or (num_epochs > 2 and i == num_epochs // 2) or i == num_epochs - 1:
                     psd_plot_data_for_frontend.append({
                        "epoch_number": i + 1, "channel_name": ch_name, "freqs": [], "psd": [], "bands": all_bands_definitions
                    })
                continue

            band_powers = calculate_band_powers_from_psd(psd, freqs, all_bands_definitions) # Use all_bands_definitions for features
            channel_features = [band_powers.get(b_name, 0.0) for b_name in band_names_order_for_features]
            current_epoch_feature_vector_parts.extend(channel_features)

        if len(current_epoch_feature_vector_parts) == len(MODEL_FEATURE_COLUMNS):
            all_epochs_features.append(current_epoch_feature_vector_parts)
        else:
            print(f"Epoch {i+1} - Feature vector length mismatch. Skipping.")

    if not all_epochs_features:
        print("❌ No features extracted.")
        # Still return plot data if any was generated
        return np.array([]), psd_plot_data_for_frontend, band_amplitude_data_for_frontend 
        
    print(f"✅ Features extracted for {len(all_epochs_features)} epochs.")
    return np.array(all_epochs_features), psd_plot_data_for_frontend, band_amplitude_data_for_frontend

# --- Model Loading and Training (remains the same) ---
def get_or_train_model():
    global trained_model
    if trained_model is not None: return trained_model
    try:
        trained_model = joblib.load(MODEL_PATH)
        print(f"✅ Pre-trained model loaded from {MODEL_PATH}")
        if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != len(MODEL_FEATURE_COLUMNS):
            print(f"⚠️ CRITICAL: Loaded model expects {trained_model.n_features_in_} features, config expects {len(MODEL_FEATURE_COLUMNS)}.")
            print(f"Current config features: {MODEL_FEATURE_COLUMNS}")
        return trained_model
    except FileNotFoundError:
        print(f"⚠️ Model file '{MODEL_PATH}' not found. Training new model...")
        try:
            data = pd.read_csv(PROCESSED_FEATURES_CSV_PATH)
            expected_cols = MODEL_FEATURE_COLUMNS + ['emotion']
            missing = [col for col in expected_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Training CSV missing: {missing}. Expected: {MODEL_FEATURE_COLUMNS} and 'emotion'.")
            X = data[MODEL_FEATURE_COLUMNS]
            y = data['emotion'].astype(str)
            if len(X) < 10 or len(y.unique()) < 2: # Basic check
                raise ValueError(f"Insufficient data/classes for training. Samples: {len(X)}, Classes: {len(y.unique())}")
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            print(f"✅ New model trained and saved to {MODEL_PATH}")
            trained_model = model
            return trained_model
        except FileNotFoundError:
            print(f"❌ ERROR: Training CSV '{PROCESSED_FEATURES_CSV_PATH}' not found.")
            return None
        except Exception as e:
            print(f"❌ ERROR during new model training: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"❌ ERROR loading model '{MODEL_PATH}': {e}")
        traceback.print_exc()
        return None

# --- Flask App Setup (remains the same) ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 
CORS(app, resources={r"/predict": {"origins": "*"}})
trained_model = None
get_or_train_model()

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_emotions_endpoint():
    global trained_model
    if trained_model is None: # Retry loading/training if it failed at startup
        print("Model not ready, attempting to load/train again for this request...")
        get_or_train_model()
        if trained_model is None:
            return jsonify({"error": "Model is not available. Server failed to load or train the model."}), 500

    if 'eeg_file' not in request.files:
        return jsonify({"error": "No EEG file ('eeg_file') provided."}), 400
    file = request.files['eeg_file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        print(f"Received file: {file.filename}")
        file_content_string = file.read().decode('utf-8', errors='ignore')
        
        # Parse raw data from the uploaded CSV
        raw_eeg_data_per_channel, sampling_rate = load_and_parse_raw_eeg_data_from_upload(
            file_content_string, EEG_CHANNELS_TO_PROCESS
        )
        
        # Extract features for prediction and detailed plot data
        extracted_features, psd_plot_data, band_amplitude_plot_data = extract_features_and_plot_data(
            raw_eeg_data_per_channel, # Pass the raw data dict
            fs=sampling_rate,
            epoch_duration=EPOCH_DURATION_SECONDS,
            overall_filter_lowcut=FILTER_LOWCUT_HZ, # Use overall filters for feature PSD
            overall_filter_highcut=FILTER_HIGHCUT_HZ,
            all_bands_definitions=EEG_BANDS_DEFINITIONS, # For feature extraction
            requested_plot_bands_defs=REQUESTED_PLOT_BANDS, # For amplitude plots
            target_channels_order=EEG_CHANNELS_TO_PROCESS,
            band_names_order_for_features=list(EEG_BANDS_DEFINITIONS.keys())
        )
        
        predictions_list = []
        if extracted_features.size > 0:
            predict_df = pd.DataFrame(extracted_features, columns=MODEL_FEATURE_COLUMNS)
            if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != predict_df.shape[1]:
                return jsonify({"error": f"Model feature mismatch. Model: {trained_model.n_features_in_}, Data: {predict_df.shape[1]}." }), 500
            predictions = trained_model.predict(predict_df)
            predictions_list = [str(p) for p in predictions.tolist()]
            print(f"✅ Predictions: {predictions_list[:5] if predictions_list else 'None'}...")
        else:
            print("⚠️ No features extracted, so no predictions will be made.")
            # Still return plot data if any was generated
            # predictions_list will be empty

        print(f"PSD plot data items: {len(psd_plot_data)}. Band amplitude plot items: {len(band_amplitude_plot_data)}.")
        
        return jsonify({
            "predictions": predictions_list,
            "psd_plot_data": psd_plot_data, # For 3D plot and band-specific PSD plots
            "band_amplitude_data": band_amplitude_plot_data, # New data for amplitude vs time
            "processed_channels": EEG_CHANNELS_TO_PROCESS,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "emotions_legend": trained_model.classes_.tolist() if hasattr(trained_model, 'classes_') else "N/A"
        })

    except ValueError as ve:
        print(f"ValueError in prediction: {ve}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"Unexpected error in prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error: {str(e)}"}), 500

# --- Main Execution (remains the same) ---
if __name__ == '__main__':
    print("--------------------------------------------------------------------")
    print("Starting Flask server for Multi-Channel EEG Emotion Prediction")
    print(f"Processing channels: {EEG_CHANNELS_TO_PROCESS}")
    print(f"Model expects {len(MODEL_FEATURE_COLUMNS)} features. E.g., {MODEL_FEATURE_COLUMNS[:2] if MODEL_FEATURE_COLUMNS else 'N/A'}...")
    print(f"Model path: {MODEL_PATH}, Training CSV: {PROCESSED_FEATURES_CSV_PATH}")
    print("--------------------------------------------------------------------")
    
    if trained_model is None:
        print("🔴 CRITICAL: Model not loaded/trained at startup.")
    else:
        if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != len(MODEL_FEATURE_COLUMNS):
             print(f"🔴 CRITICAL STARTUP WARNING: Model expects {trained_model.n_features_in_} features, config is for {len(MODEL_FEATURE_COLUMNS)}.")
        if hasattr(trained_model, 'classes_'):
            print(f"✅ Model ready. Predictable emotions: {trained_model.classes_.tolist()}")
        else:
            print("ℹ️ Model ready, but 'classes_' attribute not found.")
    app.run(debug=True, host='0.0.0.0', port=5000)