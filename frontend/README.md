# EEG Emotion Predictor - Frontend 🧠

This is the frontend for the EEG Emotion Predictor application. It provides a user interface to upload EEG data in CSV format, sends it to the backend for analysis, and displays the predicted emotions.

Okay, here's a section you can add to your **`backend/README.md`** file (or create as a separate `DATA_FORMAT.md` and link to it) to explain the expected CSV file format and its data contents.

-----

## 📄 Expected CSV Data Format for EEG Analysis

For the backend to correctly process your EEG data and for the analysis to be meaningful, the uploaded CSV file should adhere to the following structure and content guidelines:

### 1\. File Type

  * **Format:** Standard Comma-Separated Values (`.csv`) file.
  * **Encoding:** UTF-8 encoding is expected.

### 2\. File Structure

The CSV file should ideally be structured as follows:

  * **A. Metadata Section (Optional but Highly Recommended):**

      * These are informational lines at the very beginning of the file, appearing *before* the actual column headers.
      * Lines starting with `#` are often treated as comments but can also contain parsable information.
      * **Crucially, this section should contain information about the EEG recording's `Sampling Rate`.** The script attempts to automatically detect this.
          * **Examples of detectable sampling rate lines:**
              * `Sampling Rate:EEG_128Hz`
              * `Device samplingrate - EEG128`
              * Any line containing "sampling rate" (case-insensitive) and "eeg" followed by a number (e.g., `eeg_256`, `EEG 512`).
          * If the sampling rate cannot be automatically detected from these lines, the system will default to **128 Hz**. An incorrect sampling rate will lead to inaccurate frequency analysis and plots.
      * Other metadata might include device information, recording date, channel lists, etc.

  * **B. Header Row:**

      * A single line that defines the names for each column of data.
      * The script attempts to automatically locate this header row after processing the metadata. It looks for common EEG-related terms like "Timestamp", "EEG.Counter", or the specific channel name defined for processing.
      * If auto-detection fails, the system defaults to assuming the header is on the line immediately after `DEFAULT_METADATA_LINES_BEFORE_HEADER` (which is configured to `1` in `app.py`, meaning the 2nd line if no metadata is present or detection fails).

  * **C. Data Rows:**

      * Subsequent lines containing the actual recorded data, with values separated by commas, corresponding to the columns defined in the Header Row.
      * These should primarily be numerical values for EEG channels.

### 3\. Required Data Content

For successful processing, your CSV file must contain:

  * **Target EEG Channel Data:**

      * A column with the **exact name** specified by the `EEG_CHANNEL_TO_PROCESS` variable in the `app.py` configuration.
          * The default value is **`EEG.AF3`**.
          * The script attempts a case-insensitive match for this channel name.
      * The data within this column must be **numerical EEG signal values** (e.g., microvolts). Non-numeric entries will be converted to `0.0`, which could impact analysis if prevalent.

  * **Sufficient Data Length:**

      * The selected EEG channel must contain enough data points to create at least one complete "epoch" (a short time segment for analysis).
      * The epoch duration is defined by `EPOCH_DURATION_SECONDS` in `app.py` (default is 2 seconds).
      * For example, with a 128 Hz sampling rate and 2-second epochs, at least 256 valid data samples are needed for the chosen channel after any initial filtering.

### 4\. Example CSV Structure (Conceptual)

```csv
# Device: ExampleEEG V2
# Subject ID: P001
# Recording Date: 2025-05-21
# Sampling Rate:EEG_256Hz  <-- IMPORTANT: Line for sampling rate detection
# Channels: Timestamp,EEG.Counter,EEG.AF3,EEG.F7,EEG.Marker
Timestamp,EEG.Counter,EEG.AF3,EEG.F7,EEG.F3,EEG.FC5,EEG.T7,Marker  <-- Header Row
1634822400.000,1,4200.51,4250.12,4100.00,4150.75,4300.20,0         <-- Data Row 1
1634822400.004,2,4201.70,4251.30,4101.15,4151.90,4301.35,0         <-- Data Row 2
1634822400.008,3,4202.90,4252.50,4102.30,4153.05,4302.50,0         <-- Data Row 3
...                                                                 <-- etc.
```

### 5\. Importance of Correct Sampling Rate

The accuracy of the frequency spectrum analysis (including the plots showing Delta, Theta, Alpha, Beta, Gamma bands) and the derived features heavily depends on the **correct sampling rate (`fs`)**.

  * If the script fails to detect the sampling rate from your CSV's metadata, it defaults to 128 Hz.
  * If your file's actual sampling rate is different and not correctly detected, the frequency calculations will be skewed. This was observed previously where the output `freqs` array for plots showed an incorrect range (e.g., 0-6000+ Hz).
  * **Please ensure your CSV file contains clear sampling rate information that the backend script can parse, or verify/adjust the parsing logic in `app.py` (`load_and_parse_raw_eeg_data_from_upload` function) if necessary.**

By ensuring your CSV files meet these requirements, you will get more accurate and meaningful results from the EEG Emotion Predictor.

-----

## ✨ Features

* User-friendly interface for file uploads.
* Real-time feedback during file processing and prediction.
* Clear display of emotion predictions per epoch.
* Responsive design styled with Tailwind CSS.

## 🛠️ Tech Stack

* **React** (using Vite)
* **Tailwind CSS** for styling
* **Axios** for API communication

## 📋 Prerequisites

* **Node.js** (version 18.x or later recommended)
* **npm** (version 9.x or later) or **yarn**

## 🚀 Getting Started

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <your-repository-url>
    cd <your-repository-url>/frontend
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```
    or if you use yarn:
    ```bash
    yarn install
    ```

3.  **Ensure the Backend is Running**:
    This frontend application requires the backend server to be running to process EEG data and provide predictions. By default, it expects the backend to be available at `http://localhost:5000`. Please refer to the backend README for setup instructions.

4.  **Run the development server**:
    ```bash
    npm run dev
    ```
    or if you use yarn:
    ```bash
    yarn dev
    ```
    This will typically start the application on `http://localhost:5173` (Vite's default) or `http://localhost:3000`. Check your terminal output for the exact URL.

## ⚙️ Configuration

* The API endpoint for the backend is configured in `src/components/EEGProcessor.jsx`. By default, it's set to `http://localhost:5000/predict`. If your backend runs on a different URL or port, you'll need to update it there.

## 📁 Project Structure

* `src/components/EEGProcessor.jsx`: The main component for handling file uploads and displaying results.
* `src/App.jsx`: The root application component.
* `src/index.css`: Main CSS file, including Tailwind CSS directives.
* `tailwind.config.js`: Tailwind CSS configuration.
* `vite.config.js`: Vite configuration.

---