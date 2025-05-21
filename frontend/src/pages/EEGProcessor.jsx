// src/components/EEGProcessor.jsx
import React, { useState } from "react";
import axios from "axios";
import FrequencySpectrumChart from "../components/FrequencySpectrumChart"; // Ensure this path is correct

// Helper Icon for upload
const UploadIcon = () => (
  <svg
    className="w-10 h-10 mb-4 text-blue-500 group-hover:text-blue-600 transition-colors"
    aria-hidden="true"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 20 16"
  >
    <path
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5" // Slightly thinner stroke
      d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
    />
  </svg>
);

// Helper function to get dynamic colors for emotion tags
const getEmotionTagColor = (emotion) => {
  switch (emotion) {
    case "HAPPY":
      return "bg-green-100 text-green-700 border border-green-200";
    case "SAD":
      return "bg-blue-100 text-blue-700 border border-blue-200";
    case "ANGRY":
      return "bg-red-100 text-red-700 border border-red-200";
    case "NEUTRAL":
      return "bg-slate-100 text-slate-700 border border-slate-200";
    case "SURPRISED":
      return "bg-yellow-100 text-yellow-700 border border-yellow-200";
    case "FEAR":
      return "bg-purple-100 text-purple-700 border border-purple-200";
    default:
      return "bg-gray-100 text-gray-700 border border-gray-200";
  }
};

function EEGProcessor() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("No file chosen");
  const [epochPlotData, setEpochPlotData] = useState([]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        setSelectedFile(file);
        setFileName(file.name);
        setPredictions([]);
        setError("");
        setEpochPlotData([]);
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
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select an EEG file first.");
      return;
    }

    const formData = new FormData();
    formData.append("eeg_file", selectedFile);

    setIsLoading(true);
    setError("");
    setPredictions([]);
    setEpochPlotData([]);

    try {
      const response = await axios.post(
        "http://localhost:5000/predict", // Or your specific IP: 'http://192.168.159.29:5000/predict'
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 120000, 
        }
      );

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
        } else if (!hasContent && !error) {
          setError("Processing complete, but no specific results or plots were returned.");
        }
      } else {
        setError("Received an empty or unexpected response format from the server.");
      }
    } catch (err) {
      console.error("Error uploading or processing file:", err);
      if (err.code === "ECONNABORTED") {
        setError("The request timed out. Server might be busy or file too large. Please try again.");
      } else if (err.response && err.response.data && err.response.data.error) {
        setError(`Error: ${err.response.data.error}`);
      } else if (err.response && err.response.status) {
        setError(`Server returned an error: ${err.response.status} ${err.response.statusText}`);
      } else if (err.request) {
        setError("Could not connect to the server. Please ensure it is running and accessible.");
      } else {
        setError("An unexpected error occurred. Please check the console and server logs.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col items-center justify-start p-4 sm:p-6 selection:bg-blue-200 selection:text-blue-900">
      <div className="w-full max-w-4xl xl:max-w-6xl bg-white shadow-xl rounded-xl p-6 md:p-10 my-8 sm:my-12">
        <header className="mb-10 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-slate-800 pb-2">
            EEG Emotion Predictor{" "}
            <span role="img" aria-label="brain emoji" className="inline-block">
              🧠
            </span>
          </h1>
          <p className="text-slate-600 text-lg sm:text-xl">
            Upload CSV data for emotion predictions and spectrum analysis.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="space-y-8">
          <div className="group">
            <label
              htmlFor="file-upload"
              className={`flex flex-col items-center justify-center w-full h-60 sm:h-72 border-2 border-dashed rounded-lg cursor-pointer 
                                transition-all duration-300 ease-in-out
                                ${
                                  isLoading
                                    ? "border-slate-300 bg-slate-50"
                                    : "border-slate-300 hover:border-blue-500 bg-slate-50 hover:bg-blue-50"
                                }`}
            >
              <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center">
                <UploadIcon />
                <p className="mb-2 text-lg font-medium text-slate-700 group-hover:text-blue-600 transition-colors">
                  {isLoading
                    ? "Uploading..."
                    : fileName !== "No file chosen"
                    ? "File Selected"
                    : "Click to upload or drag & drop"}
                </p>
                <p className="text-sm text-slate-500">
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
            className="w-full text-white font-semibold py-3.5 px-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-300 ease-in-out 
                                   focus:outline-none focus:ring-4 focus:ring-opacity-50 focus:ring-blue-300
                                   bg-blue-600 hover:bg-blue-700 
                                   disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-400 disabled:hover:bg-slate-400"
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </div>
            ) : (
              "Analyze EEG Data"
            )}
          </button>
        </form>

        {error && (
          <div
            className="mt-8 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-md shadow-sm"
            role="alert"
          >
            <div className="flex">
              <div className="py-1">
                <svg className="fill-current h-6 w-6 text-red-500 mr-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zM10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm-1-5a1 1 0 0 1 1-1h.01a1 1 0 1 1 0 2H10a1 1 0 0 1-1-1zm0-3a1 1 0 0 1 1-1h.01a1 1 0 1 1 0 2H10a1 1 0 0 1-1-1z" />
                </svg>
              </div>
              <div>
                <p className="font-semibold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {(predictions.length > 0 || epochPlotData.length > 0) && !isLoading && !error && (
            <div className="mt-12 space-y-12">
              {predictions.length > 0 && (
                <section className="p-6 bg-slate-50 rounded-xl shadow">
                  <h3 className="text-2xl font-semibold text-slate-700 mb-6 text-center">
                    Emotion Predictions
                  </h3>
                  <ul className="space-y-3 max-h-[30rem] overflow-y-auto pr-2 custom-scrollbar">
                    {predictions.map((prediction, index) => (
                      <li
                        key={`pred-${index}`} 
                        className="p-4 bg-white border border-slate-200 rounded-lg shadow-sm flex justify-between items-center hover:shadow-md transition-shadow duration-200"
                      >
                        <span className="text-slate-600 font-medium">
                          Epoch {index + 1}:
                        </span>
                        <span
                          className={`px-3 py-1.5 text-sm font-semibold rounded-full
                                                 ${getEmotionTagColor(String(prediction).toUpperCase())}`}
                        >
                          {String(prediction).toUpperCase()}
                        </span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {epochPlotData.length > 0 && (
                <section>
                  <h3 className="text-2xl font-semibold text-slate-700 mb-8 text-center">
                    Frequency Spectrum Analysis
                  </h3>
                  <div className="space-y-10">
                    {epochPlotData.map((plot, index) => (
                      <FrequencySpectrumChart
                        key={`epoch-chart-${plot.channel_name}-${plot.epoch_number}-${index}`}
                        plotData={plot}
                      />
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
      </div>
 
    </div>
  );
}

export default EEGProcessor;