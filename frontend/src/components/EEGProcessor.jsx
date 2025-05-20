// C:/Users/Tarun/.vscode/workspace/AIML/frontend/src/components/EEGProcessor.jsx

import React, { useState } from 'react';
import axios from 'axios';

// Simple SVG Icon for upload (optional, replace with your preferred icon library if you have one)
const UploadIcon = () => (
    <svg className="w-8 h-8 mb-3 text-gray-500 dark:text-gray-400 group-hover:text-blue-500 transition-colors" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
    </svg>
);


function EEGProcessor() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [fileName, setFileName] = useState("No file chosen");

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            if (file.type === "text/csv" || file.name.endsWith(".csv")) {
                setSelectedFile(file);
                setFileName(file.name);
                setPredictions([]);
                setError('');
            } else {
                setSelectedFile(null);
                setFileName("No file chosen");
                setError("Invalid file type. Please upload a .csv file.");
            }
        } else {
            setSelectedFile(null);
            setFileName("No file chosen");
        }
    };

// Inside EEGProcessor.jsx

const handleSubmit = async (event) => {
    event.preventDefault(); // Prevent default form submission behavior

    // First, check if a file is selected
    if (!selectedFile) {
        setError('Please select an EEG file first.');
        return; // Exit if no file is selected
    }

    // If a file is selected, proceed to prepare data and make the API call

    const formData = new FormData();
    formData.append('eeg_file', selectedFile); // Add the selected file to FormData

    // Set loading states and clear previous errors/predictions
    setIsLoading(true);
    setError('');
    setPredictions([]);

    try {
        // Make the POST request to your Flask backend
        const response = await axios.post('http://localhost:5000/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data', // Important for file uploads
            },
            timeout: 60000, // 60 seconds timeout
        });

        // Handle the response from the backend
        if (response.data && response.data.predictions) {
            setPredictions(response.data.predictions);
            if (response.data.predictions.length === 0) {
                setError("Processing complete, but no predictions were returned by the model.");
            }
        } else if (response.data && response.data.error) {
            // If the backend returns a specific error message
            setError(`Server error: ${response.data.error}`);
        } else {
            // If the response format is unexpected
            setError("Received an unexpected response format from the server.");
        }
    } catch (err) {
        // Handle errors during the API call
        console.error("Error uploading or processing file:", err);
        if (err.code === 'ECONNABORTED') {
            setError('The request timed out. Please try again or check the server.');
        } else if (err.response && err.response.data && err.response.data.error) {
            // Error response from the server (e.g., 4xx, 5xx with a JSON error message)
            setError(`Error: ${err.response.data.error}`);
        } else if (err.request) {
            // The request was made but no response was received (e.g., server is down)
            setError('Could not connect to the server. Please ensure it is running and accessible.');
        } else {
            // Other errors (e.g., setup issues in the request)
            setError('An unexpected error occurred. Please check the console and server logs.');
        }
    } finally {
        // Always set loading to false after the attempt, regardless of success or failure
        setIsLoading(false);
    }
};
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex flex-col items-center justify-center p-4 selection:bg-blue-500 selection:text-white">
            <div className="w-full max-w-2xl bg-white dark:bg-slate-800 shadow-2xl rounded-xl p-6 md:p-10 transform transition-all duration-500">
                <header className="mb-8 text-center">
                    <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-sky-500 to-cyan-400 pb-2">
                        EEG Emotion Predictor <span role="img" aria-label="brain emoji">🧠</span>
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400 text-lg">
                        Upload your EEG data in CSV format to get emotion predictions.
                    </p>
                </header>

                <form onSubmit={handleSubmit} className="space-y-8">
                    <div className="group">
                        <label
                            htmlFor="file-upload"
                            className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-300 ease-in-out
                                ${isLoading ? 'border-gray-400 bg-gray-100 dark:border-gray-600 dark:bg-slate-700' : 'border-gray-300 dark:border-slate-600 bg-gray-50 dark:bg-slate-700 hover:border-blue-500 dark:hover:border-blue-400 hover:bg-blue-50 dark:hover:bg-slate-600'}`}
                        >
                            <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
                                <UploadIcon />
                                <p className="mb-2 text-lg font-semibold text-gray-700 dark:text-gray-300 group-hover:text-blue-600 transition-colors">
                                    {isLoading ? "Uploading..." : (fileName !== "No file chosen" ? "File Selected" : "Click to upload or drag & drop")}
                                </p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">
                                    {fileName !== "No file chosen" ? fileName : "CSV files only (MAX. 5MB)"}
                                </p>
                            </div>
                            <input
                                id="file-upload"
                                type="file"
                                accept=".csv,text/csv"
                                onChange={handleFileChange}
                                className="hidden"
                                disabled={isLoading}
                            />
                        </label>
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading || !selectedFile}
                        className="w-full text-white font-bold py-4 px-6 rounded-lg shadow-lg transition-all duration-300 ease-in-out transform hover:scale-105
                                   focus:outline-none focus:ring-4 focus:ring-opacity-50
                                   bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br
                                   disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none disabled:from-gray-400 disabled:to-gray-500 dark:disabled:from-slate-600 dark:disabled:to-slate-700
                                   focus:ring-blue-300 dark:focus:ring-blue-800"
                    >
                        {isLoading ? (
                            <div className="flex items-center justify-center">
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Processing...
                            </div>
                        ) : 'Predict Emotions'}
                    </button>
                </form>

                {error && (
                    <div className="mt-8 p-4 bg-red-100 dark:bg-red-900/30 border-l-4 border-red-500 dark:border-red-400 text-red-700 dark:text-red-300 rounded-md shadow-md" role="alert">
                        <div className="flex">
                            <div className="py-1">
                                <svg className="fill-current h-6 w-6 text-red-500 dark:text-red-400 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                                    <path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zM10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm-1-5a1 1 0 0 1 1-1h.01a1 1 0 1 1 0 2H10a1 1 0 0 1-1-1zm0-3a1 1 0 0 1 1-1h.01a1 1 0 1 1 0 2H10a1 1 0 0 1-1-1z"/>
                                </svg>
                            </div>
                            <div>
                                <p className="font-bold">Error</p>
                                <p className="text-sm">{error}</p>
                            </div>
                        </div>
                    </div>
                )}

                {predictions.length > 0 && (
                    <div className="mt-10 p-6 bg-gray-50 dark:bg-slate-700/50 rounded-xl shadow-inner">
                        <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6 text-center">
                            Analysis Results
                        </h3>
                        <ul className="space-y-3">
                            {predictions.map((prediction, index) => (
                                <li
                                    key={index}
                                    className="p-4 bg-white dark:bg-slate-700 border border-gray-200 dark:border-slate-600 rounded-lg shadow-sm flex justify-between items-center
                                               hover:shadow-md transition-shadow duration-200"
                                >
                                    <span className="text-gray-700 dark:text-gray-300 font-medium">Epoch {index + 1}:</span>
                                    <span className={`px-3 py-1.5 text-sm font-semibold rounded-full
                                        ${getEmotionTagColor(String(prediction).toUpperCase())}`}>
                                        {String(prediction).toUpperCase()}
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
            <footer className="text-center mt-10 pb-5">
                <p className="text-sm text-slate-500 dark:text-slate-400">
                    EEG Emotion Analysis | Powered by AI
                </p>
            </footer>
        </div>
    );
}

// Helper function to get dynamic colors for emotion tags (optional, expand as needed)
const getEmotionTagColor = (emotion) => {
    switch (emotion) {
        case 'HAPPY':
            return 'bg-green-100 text-green-800 dark:bg-green-800/50 dark:text-green-300';
        case 'SAD':
            return 'bg-blue-100 text-blue-800 dark:bg-blue-800/50 dark:text-blue-300';
        case 'ANGRY':
            return 'bg-red-100 text-red-800 dark:bg-red-800/50 dark:text-red-300';
        case 'NEUTRAL':
            return 'bg-gray-100 text-gray-800 dark:bg-gray-600/50 dark:text-gray-300';
        case 'SURPRISED':
            return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-700/50 dark:text-yellow-300';
        case 'FEAR':
             return 'bg-purple-100 text-purple-800 dark:bg-purple-800/50 dark:text-purple-300';
        default:
            return 'bg-slate-100 text-slate-800 dark:bg-slate-600/50 dark:text-slate-300';
    }
};

export default EEGProcessor;