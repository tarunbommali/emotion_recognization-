// frontend/src/components/EEGProcessor.jsx
import React, { useState, useMemo } from "react";
import axios from "axios";
import EEG3DVisualization from "./EEG3DVisualization";
import PowerFrequencyBandChart from "./PowerFrequencyBandChart"; // New
import BandAmplitudeChart from "./BandAmplitudeChart";

const UploadIcon = () => ( <svg className="w-10 h-10 mb-4 text-blue-500 group-hover:text-blue-600 transition-colors" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16" > <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" /> </svg> );
const getEmotionTagColor = (emotion) => { switch (String(emotion).toUpperCase()) { case "HAPPY": return "bg-green-100 text-green-700 border border-green-200"; case "SAD": return "bg-blue-100 text-blue-700 border border-blue-200"; case "ANGRY": case "ANGER": return "bg-red-100 text-red-700 border border-red-200"; case "NEUTRAL": return "bg-slate-100 text-slate-700 border border-slate-200"; case "SURPRISED": case "SURPRISE": return "bg-yellow-100 text-yellow-700 border border-yellow-200"; case "FEAR": return "bg-purple-100 text-purple-700 border border-purple-200"; case "RELAXED": case "CALM": return "bg-sky-100 text-sky-700 border border-sky-200"; default: return "bg-gray-100 text-gray-700 border border-gray-200"; } };

function EEGProcessor() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("No file chosen");
  
  const [psdPlotData, setPsdPlotData] = useState([]); 
  const [bandAmplitudePlotData, setBandAmplitudePlotData] = useState([]);
  const [responseInfo, setResponseInfo] = useState(null);
  const [plotBandDefinitions, setPlotBandDefinitions] = useState({});


  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        setSelectedFile(file); setFileName(file.name); setPredictions([]); setError("");
        setPsdPlotData([]); setBandAmplitudePlotData([]); setResponseInfo(null); setPlotBandDefinitions({});
      } else {
        setSelectedFile(null); setFileName("No file chosen"); setError("Invalid file type. Please upload a .csv file.");
        setPredictions([]); setPsdPlotData([]); setBandAmplitudePlotData([]); setResponseInfo(null); setPlotBandDefinitions({});
      }
    } else {
      setSelectedFile(null); setFileName("No file chosen");
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) { setError("Please select an EEG file first."); return; }
    const formData = new FormData();
    formData.append("eeg_file", selectedFile);
    setIsLoading(true); setError(""); setPredictions([]); setPsdPlotData([]); setBandAmplitudePlotData([]); setResponseInfo(null); setPlotBandDefinitions({});

    try {
      const response = await axios.post("http://localhost:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" }, timeout: 180000,
      });
      if (response.data) {
        let hasAnyData = false;
        if (response.data.error) {
          setError(`Server error: ${response.data.error}`);
        } else {
          if (response.data.predictions && Array.isArray(response.data.predictions)) {
            setPredictions(response.data.predictions);
            if (response.data.predictions.length > 0) hasAnyData = true;
          }
          if (response.data.psd_plot_data && Array.isArray(response.data.psd_plot_data)) {
            setPsdPlotData(response.data.psd_plot_data);
            if (response.data.psd_plot_data.length > 0) hasAnyData = true;
          }
          if (response.data.band_amplitude_data && Array.isArray(response.data.band_amplitude_data)) {
            setBandAmplitudePlotData(response.data.band_amplitude_data);
            const hasAmpData = response.data.band_amplitude_data.some(b => b.traces?.some(t => t.amplitude?.length > 0));
            if (hasAmpData) hasAnyData = true;
          }
          setResponseInfo({
            processed_channels: response.data.processed_channels,
            emotions_legend: response.data.emotions_legend,
            model_feature_columns: response.data.model_feature_columns,
          });
          if (response.data.band_definitions_for_plots) {
            setPlotBandDefinitions(response.data.band_definitions_for_plots);
          }
          if (!hasAnyData && !response.data.error) {
            setError("Processing complete, but no specific predictions or plot data returned.");
          }
        }
      } else { setError("Received an empty or unexpected response format from the server."); }
    } catch (err) {
      // ... (keep existing error handling)
      console.error("Error uploading or processing file:", err);
      if (err.code === "ECONNABORTED") { setError("Request timed out."); }
      else if (err.response?.data?.error) { setError(`Server error: ${err.response.data.error}`); }
      else if (err.response?.status) { setError(`Server error: ${err.response.status} ${err.response.statusText || ''}`); }
      else if (err.request) { setError("Cannot connect to server."); }
      else { setError("Client-side error."); }
    } finally { setIsLoading(false); }
  };

  // Prepare data for PowerFrequencyBandChart: data for the first epoch from psdPlotData
  const firstEpochPsdDataForAllChannels = useMemo(() => {
    if (!psdPlotData || psdPlotData.length === 0 || !responseInfo?.processed_channels) return [];
    
    // Get data for epoch 1 for all processed channels
    return responseInfo.processed_channels.map(channelName => {
        return psdPlotData.find(p => p.channel_name === channelName && p.epoch_number === 1);
    }).filter(p => p); // Filter out undefined if a channel had no epoch 1 data (shouldn't happen ideally)

  }, [psdPlotData, responseInfo]);


  return (
    <div className="min-h-screen bg-slate-100 flex flex-col items-center justify-start p-4 sm:p-6 selection:bg-blue-200 selection:text-blue-900">
      <div className="w-full max-w-4xl xl:max-w-7xl bg-white shadow-xl rounded-xl p-6 md:p-10 my-8 sm:my-12">
        <header className="mb-10 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold text-slate-800 pb-2">EEG Emotion Predictor 🧠</h1>
          <p className="text-slate-600 text-lg sm:text-xl">Upload EEG CSV for multi-channel emotion prediction & spectrum analysis.</p>
        </header>
        {/* Form (as before) */}
        <form onSubmit={handleSubmit} className="space-y-8">
            <div className="group"> <label htmlFor="file-upload" className={`flex flex-col items-center justify-center w-full h-60 sm:h-72 border-2 border-dashed rounded-lg cursor-pointer transition-all duration-300 ease-in-out ${ isLoading ? "border-slate-300 bg-slate-50 animate-pulse" : "border-slate-300 hover:border-blue-500 bg-slate-50 hover:bg-blue-50" }`} > <div className="flex flex-col items-center justify-center pt-5 pb-6 text-center"> <UploadIcon /> <p className="mb-2 text-lg font-medium text-slate-700 group-hover:text-blue-600 transition-colors"> {isLoading ? "Uploading & Processing..." : fileName !== "No file chosen" ? "File Selected" : "Click to upload or drag & drop"} </p> <p className="text-sm text-slate-500"> {fileName !== "No file chosen" ? fileName : "CSV files only"} </p> </div> <input id="file-upload" type="file" accept=".csv,text/csv" onChange={handleFileChange} className="hidden" disabled={isLoading} /> </label> </div>
            <button type="submit" disabled={isLoading || !selectedFile} className="w-full text-white font-semibold py-3.5 px-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 focus:ring-opacity-50 focus:ring-blue-300 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-slate-400 disabled:hover:bg-slate-400" > {isLoading ? ( <div className="flex items-center justify-center"> <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle> <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg> Processing EEG Data... </div> ) : ( "Analyze EEG & Predict Emotion" )} </button>
        </form>
        {/* Error display (as before) */}
        {error && ( <div className="mt-8 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-md shadow-sm" role="alert"> <div className="flex"> <div className="py-1"> <svg className="fill-current h-6 w-6 text-red-500 mr-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zM10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm-1-5a1 1 0 0 1 1-1h.01a1 1 0 1 1 0 2H10a1 1 0 0 1-1-1zm0-3a1 1 0 0 1 1-1h.01a1 1 0 1 1 0 2H10a1 1 0 0 1-1-1z" /></svg> </div> <div> <p className="font-semibold">Error</p> <p className="text-sm">{error}</p> </div> </div> </div> )}

        {/* Results section */}
        {(!isLoading && !error && (predictions.length > 0 || psdPlotData.length > 0 || bandAmplitudePlotData.length > 0 || responseInfo)) && (
          <div className="mt-12 space-y-12">
            {/* Analysis Info (as before) */}
            {responseInfo && ( <section className="p-6 bg-slate-50 rounded-xl shadow"> <h3 className="text-2xl font-semibold text-slate-700 mb-4 text-center"> Analysis Information </h3> <div className="space-y-2 text-sm text-slate-600"> {responseInfo.processed_channels && Array.isArray(responseInfo.processed_channels) && ( <p><strong>Processed EEG Channels:</strong> {responseInfo.processed_channels.join(', ')}</p> )} {responseInfo.emotions_legend && responseInfo.emotions_legend !== "N/A (model has no classes_ attribute)" && ( <p><strong>Model Emotion Classes:</strong> {Array.isArray(responseInfo.emotions_legend) ? responseInfo.emotions_legend.join(', ') : responseInfo.emotions_legend}</p> )} {responseInfo.model_feature_columns && Array.isArray(responseInfo.model_feature_columns) && responseInfo.model_feature_columns.length > 0 && ( <p className="text-xs text-gray-500">({responseInfo.model_feature_columns.length} features used by model e.g., {responseInfo.model_feature_columns[0]})</p> )} </div> </section> )}
            {/* Predictions (as before) */}
            {predictions.length > 0 && ( <section className="p-6 bg-slate-50 rounded-xl shadow"> <h3 className="text-2xl font-semibold text-slate-700 mb-6 text-center"> Emotion Predictions per Epoch </h3> <ul className="space-y-3 max-h-[30rem] overflow-y-auto pr-2 custom-scrollbar"> {predictions.map((prediction, index) => ( <li key={`pred-${index}`} className="p-4 bg-white border border-slate-200 rounded-lg shadow-sm flex justify-between items-center hover:shadow-md transition-shadow duration-200" > <span className="text-slate-600 font-medium"> Epoch {index + 1}: </span> <span className={`px-3 py-1.5 text-sm font-semibold rounded-full ${getEmotionTagColor(prediction)}`} > {String(prediction).toUpperCase()} </span> </li> ))} </ul> </section> )}
            {/* 3D PSD Plot (as before) */}
            {psdPlotData.length > 0 && ( <section> <h3 className="text-2xl font-semibold text-slate-700 mb-8 text-center"> 3D Frequency Spectrum Analysis (Selected Epochs) </h3> <EEG3DVisualization plotDataList={psdPlotData} processedChannels={responseInfo?.processed_channels} /> </section> )}
            
            {/* Power vs Frequency (per band) - For first epoch */}
            {firstEpochPsdDataForAllChannels.length > 0 && Object.keys(plotBandDefinitions).length > 0 && (
                <section>
                    <h3 className="text-2xl font-semibold text-slate-700 mt-12 mb-8 text-center">
                        Power vs. Frequency (First Epoch)
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {Object.entries(plotBandDefinitions).map(([bandKey, bandRange]) => (
                            <PowerFrequencyBandChart
                                key={`power-band-${bandKey}`}
                                title={`${bandKey.charAt(0).toUpperCase() + bandKey.slice(1)} Band (${bandRange[0]}-${bandRange[1]} Hz)`}
                                channelsPsdData={firstEpochPsdDataForAllChannels} // Pass PSD data for all channels (for the first epoch)
                                targetBandName={bandKey}
                                targetBandRange={bandRange}
                            />
                        ))}
                    </div>
                </section>
            )}

            {/* Amplitude vs Time (per band) - For first epoch */}
            {bandAmplitudePlotData.length > 0 && (
              <section>
                <h3 className="text-2xl font-semibold text-slate-700 mt-12 mb-8 text-center">
                  Amplitude vs. Time (First Epoch)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {bandAmplitudePlotData.map((bandDataEntry) => ( // bandDataEntry is { band_name, freq_range_hz, time_vector, traces }
                    <BandAmplitudeChart
                      key={`amp-band-${bandDataEntry.band_name}`}
                      bandData={bandDataEntry}
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