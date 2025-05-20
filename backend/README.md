# EEG Emotion Predictor - Backend 🐍

This backend server provides the API for the EEG Emotion Predictor application. It receives raw EEG data (CSV files), processes the signal, extracts features, and uses a machine learning model to predict emotions.

## ✨ Features

* Flask-based API endpoint for predictions.
* EEG signal processing (filtering, epoching).
* Feature extraction (band power calculation).
* Emotion prediction using a pre-trained Random Forest model.
* Automatic model training if a pre-trained model is not found (using `eeg_features_emotions.csv`).

## 🛠️ Tech Stack

* **Python** (3.9+)
* **Flask** (for the web server and API)
* **Flask-CORS** (for Cross-Origin Resource Sharing)
* **Pandas** & **NumPy** (for data manipulation)
* **SciPy** (for signal processing)
* **Scikit-learn** (for the machine learning model)
* **Joblib** (for saving/loading the trained model)

## 📋 Prerequisites

* **Python** (version 3.9 or later recommended)
* **pip** (Python package installer)
* Access to a terminal or command prompt.

## 🚀 Getting Started

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <your-repository-url>
    cd <your-repository-url>/backend
    ```

2.  **Create and activate a virtual environment**:
    * On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On Windows (PowerShell):
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * On Windows (Command Prompt):
        ```bash
        python -m venv venv
        venv\Scripts\activate.bat
        ```

3.  **Install dependencies**:
    Ensure you have a `requirements.txt` file in this `backend` directory with the following content:
    ```txt
    Flask
    Flask-CORS
    numpy
    pandas
    scikit-learn
    scipy
    joblib
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Files and Model**:
    * **Training Data**: Make sure the `eeg_features_emotions.csv` file is present in this `backend` directory. This file is used to train the emotion prediction model if a pre-trained model (`emotion_model.joblib`) is not found.
    * **Pre-trained Model**: If you have a pre-trained `emotion_model.joblib`, place it in this `backend` directory. This will skip the training step on startup. If not present, the application will attempt to train and save a new model based on the CSV file.

5.  **Run the Flask application**:
    ```bash
    python app.py
    ```
    The server will start, typically on `http://localhost:5000`. Check your terminal for output, including model loading/training status and the server address.

## 📡 API Endpoints

### `/predict`

* **Method**: `POST`
* **Description**: Accepts an EEG CSV file, processes it, and returns emotion predictions for each epoch.
* **Request**: `multipart/form-data` with a single field:
    * `eeg_file`: The EEG data file (CSV format).
* **Response**:
    * **Success (200 OK)**:
        ```json
        {
          "predictions": ["HAPPY", "NEUTRAL", "SAD", ...]
        }
        ```
    * **Error (4xx/5xx)**:
        ```json
        {
          "error": "Error message describing the issue"
        }
        ```

## ⚙️ Configuration

The main application logic and configurations are in `app.py`:
* `PROCESSED_FEATURES_CSV_PATH`: Path to the CSV for training.
* `MODEL_PATH`: Path for saving/loading the trained model.
* `EEG_CHANNEL_TO_PROCESS`: Default EEG channel to analyze.
* Signal processing parameters (epoch duration, filter cutoffs).

## 📁 Project Structure

* `app.py`: The main Flask application file containing all logic and API endpoints.
* `eeg_features_emotions.csv`: Data used for training the model if no pre-trained model exists.
* `emotion_model.joblib`: The saved pre-trained machine learning model (generated if not present).
* `requirements.txt`: List of Python dependencies.
* `venv/`: Virtual environment directory (should be in `.gitignore`).

---