# EEG Emotion Predictor - Frontend 🧠

This is the frontend for the EEG Emotion Predictor application. It provides a user interface to upload EEG data in CSV format, sends it to the backend for analysis, and displays the predicted emotions.

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