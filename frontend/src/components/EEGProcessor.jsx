// src/components/EEGProcessor.jsx
import React, { useState } from 'react';
import axios from 'axios';
import FrequencySpectrumChart from './FrequencySpectrumChart'; // Import the chart component

// Helper Icon for upload (can be replaced or improved)
const UploadIcon = () => (
    <svg className="w-8 h-8 mb-3 text-gray-500 dark:text-gray-400 group-hover:text-blue-500 transition-colors" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
    </svg>
);

// Helper function to get dynamic colors for emotion tags
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

function EEGProcessor() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [predictions, setPredictions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [fileName, setFileName] = useState("No file chosen");
    const [epochPlotData, setEpochPlotData] = useState([]); // State for plot data from backend

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            if (file.type === "text/csv" || file.name.endsWith(".csv")) {
                setSelectedFile(file);
                setFileName(file.name);
                setPredictions([]); // Clear previous predictions
                setError(''); // Clear previous errors
                setEpochPlotData([]); // Clear previous plot data
            } else {
                setSelectedFile(null);
                setFileName("No file chosen");
                setError("Invalid file type. Please upload a .csv file.");
                setPredictions([]);
                setEpochPlotData([]);
            }
        } else {
            setSelectedFile(null);
            setFileName("No file chosen");
            // Optionally clear other states if needed when file is deselected
            // setPredictions([]);
            // setError('');
            // setEpochPlotData([]);
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!selectedFile) {
            setError('Please select an EEG file first.');
            return;
        }

        const formData = new FormData();
        formData.append('eeg_file', selectedFile);

        setIsLoading(true);
        setError('');
        setPredictions([]);
        setEpochPlotData([]); // Clear plot data on new submission

        try {
            const response = await axios.post('http://localhost:5000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 120000, // Increased timeout (2 minutes)
            });

            // Handle response data carefully
            if (response.data) {
                let hasContent = false;
                if (response.data.predictions && Array.isArray(response.data.predictions)) {
                    setPredictions(response.data.predictions);
                    if (response.data.predictions.length > 0) hasContent = true;
                }
                if (response.data.plot_data && Array.isArray(response.data.plot_data)) {
                    setEpochPlotData(response.data.plot_data);
                     if (response.data.plot_data.length > 0) hasContent = true;
                }

                if (response.data.error) {
                     setError(`Server error: ${response.data.error}`);
                } else if (!hasContent) {
                     // If neither predictions nor plot_data (with content) nor error is present
                    setError("Processing complete, but no specific results or plots were returned.");
                }
            } else {
                setError("Received an empty or unexpected response format from the server.");
            }
        } catch (err) {
            console.error("Error uploading or processing file:", err);
            if (err.code === 'ECONNABORTED') {
                setError('The request timed out. The server might be busy or the file too large for the current timeout. Please try again.');
            } else if (err.response && err.response.data && err.response.data.error) {
                setError(`Error: ${err.response.data.error}`);
            } else if (err.response && err.response.status) { // Handle other HTTP error statuses
                 setError(`Server returned an error: ${err.response.status} ${err.response.statusText}`);
            }
            else if (err.request) {
                setError('Could not connect to the server. Please ensure it is running and accessible.');
            } else {
                setError('An unexpected error occurred. Please check the console and server logs.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex flex-col items-center justify-center p-4 selection:bg-blue-500 selection:text-white">
            <div className="w-full max-w-3xl xl:max-w-4xl bg-white dark:bg-slate-800 shadow-2xl rounded-xl p-6 md:p-10 transform transition-all duration-500 my-10">
                <header className="mb-8 text-center">
                    <h1 className="text-3xl sm:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-sky-500 to-cyan-400 pb-2">
                        EEG Emotion Predictor <span role="img" aria-label="brain emoji">🧠</span>
                    </h1>
                    <p className="text-gray-600 dark:text-gray-400 text-md sm:text-lg">
                        Upload your EEG data in CSV format for emotion predictions and spectrum analysis.
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
                                    {fileName !== "No file chosen" ? fileName : "CSV files only"}
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
                        className="w-full text-white font-bold py-3.5 px-6 rounded-lg shadow-lg transition-all duration-300 ease-in-out transform hover:scale-105
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
                        ) : 'Analyze EEG Data'}
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

                {/* Results Section */}
                {(predictions.length > 0 || epochPlotData.length > 0) && !isLoading && !error && (
                    <div className="mt-10 space-y-10">
                        {/* Display Predictions */}
                        {predictions.length > 0 && (
                            <section className="p-6 bg-gray-50 dark:bg-slate-700/50 rounded-xl shadow-inner">
                                <h3 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6 text-center">
                                    Emotion Predictions
                                </h3>
                                <ul className="space-y-3 max-h-96 overflow-y-auto pr-2">
                                    {predictions.map((prediction, index) => (
                                        <li
                                            key={index}
                                            className="p-3 sm:p-4 bg-white dark:bg-slate-700 border border-gray-200 dark:border-slate-600 rounded-lg shadow-sm flex justify-between items-center
                                                       hover:shadow-md transition-shadow duration-200"
                                        >
                                            <span className="text-gray-700 dark:text-gray-300 font-medium">Epoch {index + 1}:</span>
                                            <span className={`px-3 py-1.5 text-xs sm:text-sm font-semibold rounded-full
                                                ${getEmotionTagColor(String(prediction).toUpperCase())}`}>
                                                {String(prediction).toUpperCase()}
                                            </span>
                                        </li>
                                    ))}
                                </ul>
                            </section>
                        )}

                        {/* Display Plots */}
                        {epochPlotData.length > 0 && (
                            <section>
                                <h3 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6 text-center">
                                    Frequency Spectrum Analysis
                                </h3>
                                <div className="space-y-8">
                                {epochPlotData.map((plot, index) => (
                                    <FrequencySpectrumChart key={index} plotData={plot} />
                                ))}
                                </div>
                            </section>
                        )}
                    </div>
                )}
            </div> {/* End of main content card */}

            <footer className="text-center mt-8 mb-5">
                <p className="text-sm text-slate-500 dark:text-slate-400">
                    EEG Emotion Analysis | Advanced AI Insights
                </p>
            </footer>
        </div> // End of screen container
    );
}

export default EEGProcessor;