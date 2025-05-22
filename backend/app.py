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
EEG_CHANNELS_TO_PROCESS = ['EEG.AF3', 'EEG.F7'] # Currently 2 channels

EEG_BANDS_DEFINITIONS = { # Used for feature extraction AND plot band definitions
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45) # Gamma used for features, can be plotted if added to REQUESTED_PLOT_BANDS
}

# Bands specifically requested for the new plots
REQUESTED_PLOT_BANDS = {
    'delta': EEG_BANDS_DEFINITIONS['delta'],
    'theta': EEG_BANDS_DEFINITIONS['theta'],
    'alpha': EEG_BANDS_DEFINITIONS['alpha'],
    'beta': EEG_BANDS_DEFINITIONS['beta'],
}

MODEL_FEATURE_COLUMNS = [
    f"{channel.replace('.', '')}{band}"
    for channel in EEG_CHANNELS_TO_PROCESS
    for band in EEG_BANDS_DEFINITIONS.keys() # Features from all defined bands
]

EPOCH_DURATION_SECONDS = 2
FILTER_LOWCUT_HZ = 0.5 # Overall filter for feature extraction PSD
FILTER_HIGHCUT_HZ = 45 # Overall filter for feature extraction PSD
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1

# --- EEG Data Loading and Parsing (remains the same as your provided version) ---
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
        if channel_data_numeric.isnull().sum() > 0: 
             print(f"⚠️ Non-numeric entries in '{actual_channel_name_in_df}' were filled with 0.0.")
        all_channels_data[eeg_channel_name] = channel_data_numeric.to_numpy()
        print(f"✅ Parsed data for '{actual_channel_name_in_df}'. Samples: {len(all_channels_data[eeg_channel_name])}.")
    if not found_all_required_channels or len(all_channels_data) != len(eeg_channel_names_list):
        missing = [ch for ch in eeg_channel_names_list if ch not in all_channels_data]
        raise ValueError(f"Missing channels: {missing}. Available: {', '.join(data_df.columns[:15])}...")
    print(f"✅ Parsed data for all requested channels. Rate: {sampling_rate} Hz.")
    return all_channels_data, sampling_rate


# --- EEG Signal Processing Functions ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0: high = 1.0 - 1e-6
    if low <= 0: low = 1e-6
    if low >= high: # Attempt to fix common issue if highcut was drastically reduced due to low fs
        original_lowcut = low * nyq
        original_highcut = high * nyq
        print(f"Warning: Filter lowcut ({original_lowcut}Hz) was >= highcut ({original_highcut}Hz). Attempting to adjust.")
        lowcut = highcut / 2 # Example adjustment
        low = lowcut / nyq
        if low >= high: # If still invalid, raise error
            raise ValueError(f"Corrected filter lowcut ({lowcut}Hz) is still >= highcut ({highcut}Hz). Cannot create filter.")
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

def calculate_band_powers_from_psd(psd, freqs, bands_def): # Renamed arg for clarity
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands_def.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs < high_freq)
        band_powers[band_name] = np.mean(psd[idx_band]) if np.sum(idx_band) > 0 else 0.0
    return band_powers

# This function now also generates band-filtered time series data
def extract_features_from_eeg_data( # Kept original name as per user's app.py
    all_channels_raw_data, # Dict: {channel_name: np.array from CSV}
    fs,
    epoch_duration,
    global_lowcut, # Renamed for clarity: overall filter for feature extraction PSD
    global_highcut, # Renamed for clarity: overall filter for feature extraction PSD
    bands_def_for_features, # For feature extraction (e.g., including gamma)
    plot_bands_definitions, # For specific band-filtered time series plots (delta, theta, alpha, beta)
    target_channels_order,
    band_names_order_for_features # For ordering features for the model
    ):

    if not all_channels_raw_data or fs is None:
        return np.array([]), [], [] # features, psd_plot_data, band_amplitude_plot_data

    print(f"\n--- Extracting Features & Plot Data for: {', '.join(target_channels_order)} ---")
    
    min_length = min(len(data) for data in all_channels_raw_data.values())
    epoch_length_samples = int(epoch_duration * fs)
    num_epochs = min_length // epoch_length_samples

    if num_epochs == 0:
        print(f"⚠️ Data too short for any full epochs. No data processed.")
        return np.array([]), [], []

    # Create time vector for the first epoch (used for amplitude plots)
    time_vector_epoch = np.arange(0, epoch_length_samples / fs, 1 / fs)[:epoch_length_samples]

    # 1. Generate Band-Filtered Amplitude Data for the *first* epoch
    band_amplitude_data_for_frontend = []
    first_epoch_raw_data_map = {
        ch_name: all_channels_raw_data[ch_name][:epoch_length_samples] for ch_name in target_channels_order
    }

    for band_name_key, (low, high) in plot_bands_definitions.items():
        band_specific_traces = []
        for ch_name in target_channels_order:
            if ch_name in first_epoch_raw_data_map:
                try:
                    band_filtered_amplitude = butter_bandpass_filter(
                        first_epoch_raw_data_map[ch_name], low, high, fs, order=4
                    )
                    band_specific_traces.append({
                        "channel_name": ch_name,
                        "amplitude": band_filtered_amplitude.tolist()
                    })
                except ValueError as e:
                    print(f"❌ Error filtering {ch_name} for {band_name_key} band amplitude: {e}")
                    band_specific_traces.append({"channel_name": ch_name, "amplitude": [0]*epoch_length_samples })
            
        band_amplitude_data_for_frontend.append({
            "band_name": band_name_key, # e.g. 'alpha'
            "freq_range_hz": [low, high], # e.g. [8, 13]
            "time_vector": time_vector_epoch.tolist(),
            "traces": band_specific_traces
        })
        print(f"Generated amplitude data for {band_name_key} band.")

    # 2. Process Features and Full PSD for selected epochs (for 3D plot & model prediction)
    # Apply overall filter to the necessary length of the signal ONCE for PSD calculation
    overall_filtered_signal_for_psd = {}
    current_psd_filter_highcut = global_highcut
    if current_psd_filter_highcut >= fs / 2.0:
        current_psd_filter_highcut = (fs / 2.0) - 1e-3
    
    for ch_name in target_channels_order:
        try:
            # Filter only the portion of the signal that will be used for epochs
            signal_to_filter = all_channels_raw_data[ch_name][:num_epochs * epoch_length_samples]
            overall_filtered_signal_for_psd[ch_name] = butter_bandpass_filter(
                signal_to_filter, global_lowcut, current_psd_filter_highcut, fs
            )
        except ValueError as e:
            print(f"❌ Error during overall signal filtering for PSD for channel {ch_name}: {e}")
            # Fallback or error handling
            overall_filtered_signal_for_psd[ch_name] = np.zeros(num_epochs * epoch_length_samples)


    all_epochs_features = []
    psd_plot_data_for_frontend = [] 

    for i in range(num_epochs):
        current_epoch_feature_vector_parts = []
        for ch_name in target_channels_order:
            epoch_channel_data_for_psd = overall_filtered_signal_for_psd[ch_name][i * epoch_length_samples : (i + 1) * epoch_length_samples]
            
            nperseg_val = min(epoch_length_samples, 256 if fs >= 128 else int(fs*1.5)) # Adjusted nperseg
            nperseg_val = max(1, min(nperseg_val, len(epoch_channel_data_for_psd))) # Ensure nperseg is valid
            
            if len(epoch_channel_data_for_psd) < nperseg_val : # Check again if epoch data is too short for nperseg
                print(f"Epoch {i+1}, Chan {ch_name}: data length {len(epoch_channel_data_for_psd)} < nperseg {nperseg_val}. Zeroing features.")
                freqs, psd = np.array([]), np.array([]) # Empty arrays for safety
                band_powers = {b_name: 0.0 for b_name in band_names_order_for_features}
            else:
                try:
                    freqs, psd = welch(epoch_channel_data_for_psd, fs=fs, nperseg=nperseg_val, scaling='density')
                except ValueError as e: # Catch specific Welch errors (e.g., if data is all zeros)
                    print(f"Error during Welch for epoch {i+1}, Channel {ch_name}: {e}. Zeroing features.")
                    freqs, psd = np.array([]), np.array([]) # Use empty arrays
                    band_powers = {b_name: 0.0 for b_name in band_names_order_for_features}
            
            # Collect plot data for 3D plot (first, middle, last epoch)
            # This is ALSO the data used for "Power vs Frequency per band" plots on the frontend
            # The frontend will filter by band for those specific charts.
            if i == 0 or (num_epochs > 2 and i == num_epochs // 2) or i == num_epochs - 1:
                psd_plot_data_for_frontend.append({
                    "epoch_number": i + 1, 
                    "channel_name": ch_name,
                    "freqs": freqs.tolist() if freqs.size > 0 else [], 
                    "psd": psd.tolist() if psd.size > 0 else [], 
                    "bands": bands_def_for_features 
                })
            
            if freqs.size > 0 and psd.size > 0:
                 band_powers = calculate_band_powers_from_psd(psd, freqs, bands_def_for_features)
            else: # If Welch failed or freqs/psd are empty
                 band_powers = {b_name: 0.0 for b_name in band_names_order_for_features}

            channel_features = [band_powers.get(b_name, 0.0) for b_name in band_names_order_for_features]
            current_epoch_feature_vector_parts.extend(channel_features)

        if len(current_epoch_feature_vector_parts) == len(MODEL_FEATURE_COLUMNS):
            all_epochs_features.append(current_epoch_feature_vector_parts)
        # No else needed; if mismatch, it's skipped, features list won't grow for this epoch.

    if not all_epochs_features: print("❌ No features extracted for any epoch.")
    else: print(f"✅ Features extracted for {len(all_epochs_features)} epochs.")
        
    return np.array(all_epochs_features), psd_plot_data_for_frontend, band_amplitude_data_for_frontend


# --- Model Loading and Training (remains the same as your provided version) ---
def get_or_train_model():
    global trained_model
    if trained_model is not None: return trained_model
    try:
        trained_model = joblib.load(MODEL_PATH)
        print(f"✅ Pre-trained model loaded from {MODEL_PATH}")
        if hasattr(trained_model, 'n_features_in_') and trained_model.n_features_in_ != len(MODEL_FEATURE_COLUMNS):
            print(f"⚠️ CRITICAL: Loaded model expects {trained_model.n_features_in_} features, config expects {len(MODEL_FEATURE_COLUMNS)}.")
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
            if len(X) < 5 or len(y.unique()) < 2: # Reduced minimum samples for small debug datasets
                print(f"Warning: Training with very few samples ({len(X)}) or classes ({len(y.unique())}).")
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
    if trained_model is None:
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
        
        raw_eeg_data_per_channel, sampling_rate = load_and_parse_raw_eeg_data_from_upload(
            file_content_string, EEG_CHANNELS_TO_PROCESS
        )
        
        extracted_features, psd_plot_data_for_fe, band_amplitude_data_for_fe = extract_features_from_eeg_data(
            raw_eeg_data_per_channel,
            fs=sampling_rate,
            epoch_duration=EPOCH_DURATION_SECONDS,
            global_lowcut=FILTER_LOWCUT_HZ, 
            global_highcut=FILTER_HIGHCUT_HZ,
            bands_def_for_features=EEG_BANDS_DEFINITIONS, # For feature extraction
            plot_bands_definitions=REQUESTED_PLOT_BANDS, # For specific amplitude plots
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
        
        return jsonify({
            "predictions": predictions_list,
            "psd_plot_data": psd_plot_data_for_fe, 
            "band_amplitude_data": band_amplitude_data_for_fe,
            "processed_channels": EEG_CHANNELS_TO_PROCESS,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "emotions_legend": trained_model.classes_.tolist() if hasattr(trained_model, 'classes_') else "N/A",
            "band_definitions_for_plots": REQUESTED_PLOT_BANDS # Send this to help frontend title charts
        })

    except ValueError as ve:
        print(f"ValueError in prediction: {ve}")
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"Unexpected error in prediction: {e}")
        return jsonify({"error": f"An unexpected server error: {str(e)}"}), 500

# --- Main Execution (remains the same) ---
if __name__ == '__main__':
    print("--- Server Starting ---") # Simplified startup messages
    print(f"Channels: {EEG_CHANNELS_TO_PROCESS}")
    print(f"Model Features: {len(MODEL_FEATURE_COLUMNS)} (e.g., {MODEL_FEATURE_COLUMNS[:1]})")
    if trained_model is None: print("🔴 CRITICAL: Model not loaded/trained at startup.")
    else: print(f"✅ Model ready. Emotions: {trained_model.classes_.tolist() if hasattr(trained_model, 'classes_') else 'N/A'}")
    app.run(debug=True, host='0.0.0.0', port=5000)