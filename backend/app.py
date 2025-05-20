import io
import re
import traceback
import joblib # Ensure this is imported
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.ensemble import RandomForestClassifier # Ensure this is imported

# --- Configuration Constants (keep as is) ---
PROCESSED_FEATURES_CSV_PATH = "eeg_features_emotions.csv"
MODEL_PATH = "emotion_model.joblib"
EXPECTED_FEATURE_COLUMNS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
EEG_BANDS_DEFINITIONS = { # This will be sent to the frontend
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
EEG_CHANNEL_TO_PROCESS = 'EEG.AF3'
EPOCH_DURATION_SECONDS = 2
FILTER_LOWCUT_HZ = 0.5
FILTER_HIGHCUT_HZ = 45
DEFAULT_METADATA_LINES_BEFORE_HEADER = 1

# --- EEG Processing Functions (load_and_parse_raw_eeg_data_from_upload, butter_bandpass_filter, calculate_band_powers_from_psd - keep as they were in the previous complete app.py) ---
# For brevity, I'm not re-pasting these full functions here, assume they are present and correct from the previous version.
# Ensure your `load_and_parse_raw_eeg_data_from_upload` is robust.

# --- Modified process_eeg_channel_for_features ---
def process_eeg_channel_for_features(eeg_data, fs, epoch_duration, lowcut, highcut, bands_def, expected_features_order, channel_name):
    if eeg_data is None or fs is None:
        print("❌ Cannot process EEG channel: Data or sampling rate is None.")
        return np.array([]), [] # Return empty features and empty plot data

    print(f"\n--- Processing EEG Channel Data for {channel_name} ---")
    # (Initial checks and filtering logic as before)
    if len(eeg_data) < fs * epoch_duration:
        print(f"⚠️ Data length ({len(eeg_data)} samples) is less than one epoch. No features/plots.")
        return np.array([]), []

    print(f"Applying band-pass filter ({lowcut}-{highcut} Hz)...")
    try:
        if highcut >= fs / 2.0:
            original_highcut = highcut
            highcut = (fs / 2.0) - 1.0 # Ensure it's strictly less
            print(f"⚠️ Filter highcut ({original_highcut} Hz) adjusted to {highcut:.2f} Hz for sampling rate ({fs} Hz).")
            if lowcut >= highcut:
                print(f"❌ Lowcut ({lowcut}Hz) is >= adjusted highcut ({highcut}Hz). Cannot filter.")
                return np.array([]), []
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
        print(f"⚠️ Data too short for any full epochs. No features/plots.")
        return np.array([]), []
        
    print(f"Segmenting data into {num_epochs} epochs of {epoch_duration}s each.")

    all_epoch_features = []
    epoch_plot_data_list = [] # To store data for plots

    for i in range(num_epochs):
        epoch_data = filtered_data[i * epoch_length_samples : (i + 1) * epoch_length_samples]
        nperseg_val = min(epoch_length_samples, 256 if fs >= 128 else int(fs))
        if nperseg_val == 0 or nperseg_val > len(epoch_data):
            nperseg_val = len(epoch_data) # Adjust if too large or zero
        if nperseg_val == 0: # Still zero means epoch_data is empty, skip
            print(f"Epoch {i+1} has no data after nperseg adjustment. Skipping.")
            continue

        try:
            freqs, psd = welch(epoch_data, fs=fs, nperseg=nperseg_val, scaling='density')
            
            # Collect plot data for the first epoch and one near the middle (e.g. if num_epochs > 5)
            # You can adjust this logic to select which epochs to send plot data for
            if i == 0 or (num_epochs > 5 and i == num_epochs // 2) or i == num_epochs -1 : # First, middle-ish, and last
                 epoch_plot_data_list.append({
                    "epoch_number": i + 1,
                    "channel_name": channel_name,
                    "freqs": freqs.tolist(), # Convert numpy arrays to lists for JSON
                    "psd": psd.tolist(),
                    "bands": bands_def # Send band definitions for highlighting
                })
                
        except ValueError as e:
            print(f"Error during Welch calculation for epoch {i+1}: {e}. Skipping epoch.")
            continue 

        band_powers = calculate_band_powers_from_psd(psd, freqs, bands_def)
        current_epoch_feature_vector = [band_powers.get(feature_name, 0.0) for feature_name in expected_features_order]
        all_epoch_features.append(current_epoch_feature_vector)
        
        if i < 2 or i == num_epochs -1: 
            print(f"   Epoch {i+1}/{num_epochs} - Features: {[f'{x:.3f}' for x in current_epoch_feature_vector]}")

    if not all_epoch_features: 
        print("❌ No features were extracted after processing all epochs.")
        return np.array([]), []

    print(f"✅ Feature extraction complete. Extracted features for {len(all_epoch_features)} epochs.")
    return np.array(all_epoch_features), epoch_plot_data_list # Return plot data as well

# --- Flask App Setup (keep get_or_train_model as is) ---
app = Flask(__name__)
CORS(app)
trained_model = None # Keep global model variable

# (Ensure get_or_train_model, load_and_parse_raw_eeg_data_from_upload, butter_bandpass_filter, calculate_band_powers_from_psd are defined here)
# For brevity, let's assume the full `app.py` from the previous "perfect connection" step, but we'll modify the endpoint.
# PASTE YOUR FULL `load_and_parse_raw_eeg_data_from_upload`, `butter_bandpass_filter`, `calculate_band_powers_from_psd`, and `get_or_train_model` functions here

# --- Modified /predict endpoint ---
@app.route('/predict', methods=['POST'])
def predict_emotions_endpoint():
    model = get_or_train_model() # Your existing model loading function
    if model is None:
        return jsonify({"error": "Model not available. Please check server logs."}), 500

    if 'eeg_file' not in request.files:
        return jsonify({"error": "No EEG file provided in the request."}), 400
    # (File checks as before) ...
    file = request.files['eeg_file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload."}), 400

    try:
        print(f"Received file: {file.filename}")
        file_content_string = file.read().decode('utf-8')

        raw_eeg_signal, sampling_rate = load_and_parse_raw_eeg_data_from_upload(
            file_content_string,
            EEG_CHANNEL_TO_PROCESS # This is still from your global config
        )

        if raw_eeg_signal is None or sampling_rate is None or len(raw_eeg_signal) == 0:
            return jsonify({"error": "Failed to parse valid EEG signal data from the uploaded file."}), 400
        
        current_filter_highcut = FILTER_HIGHCUT_HZ

        # Call the modified processing function
        extracted_features_from_raw, plot_data_for_epochs = process_eeg_channel_for_features(
            raw_eeg_signal,
            fs=sampling_rate,
            epoch_duration=EPOCH_DURATION_SECONDS,
            lowcut=FILTER_LOWCUT_HZ,
            highcut=current_filter_highcut,
            bands_def=EEG_BANDS_DEFINITIONS, # Pass the definitions
            expected_features_order=EXPECTED_FEATURE_COLUMNS,
            channel_name=EEG_CHANNEL_TO_PROCESS # Pass channel name for plots
        )

        if extracted_features_from_raw.size == 0:
            return jsonify({"error": "No features extracted. Cannot make predictions or generate plots."}), 400

        predict_df = pd.DataFrame(extracted_features_from_raw, columns=EXPECTED_FEATURE_COLUMNS)
        predictions = model.predict(predict_df)
        predictions_list = [str(p) for p in predictions.tolist()]

        print(f"✅ Predictions: {predictions_list[:3]}... Plot data for {len(plot_data_for_epochs)} epochs.")
        return jsonify({
            "predictions": predictions_list,
            "plot_data": plot_data_for_epochs # Add plot data to the response
        })

    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An unexpected server error occurred. " + str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server for EEG Emotion Prediction with Plotting Data...")
    # Ensure the full `get_or_train_model` and other helper functions are defined above this line
    # For brevity, I am not re-listing all helper functions. Refer to your complete `app.py`.
    if get_or_train_model() is None: # Example of how to call it
         print("🔴 CRITICAL: Model could not be loaded or trained.")
    app.run(debug=True, host='0.0.0.0', port=5000)