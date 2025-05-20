// src/components/FrequencySpectrumChart.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler, // Needed for area fills if you try that later
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin // Register the annotation plugin
);

const FrequencySpectrumChart = ({ plotData }) => {
  if (!plotData || !plotData.freqs || !plotData.psd) {
    return <p className="text-center text-gray-500 dark:text-gray-400">No plot data available for this epoch.</p>;
  }

  const { epoch_number, channel_name, freqs, psd, bands } = plotData;

  const data = {
    labels: freqs.map(f => f.toFixed(2)), // X-axis labels (frequencies)
    datasets: [
      {
        label: `PSD for ${channel_name}`,
        data: psd, // Y-axis data (power spectral density)
        borderColor: 'rgb(54, 162, 235)', // Blue line
        backgroundColor: 'rgba(54, 162, 235, 0.2)', // Light blue fill under the line
        tension: 0.1,
        fill: 'origin', // Fill from the line to the x-axis
        pointRadius: 1,
        borderWidth: 1.5,
      },
    ],
  };

  const bandColors = {
    delta: 'rgba(0, 0, 255, 0.1)',   // Blue
    theta: 'rgba(0, 128, 0, 0.1)',  // Green
    alpha: 'rgba(255, 0, 0, 0.1)',  // Red
    beta:  'rgba(128, 0, 128, 0.1)',// Purple
    gamma: 'rgba(255, 165, 0, 0.1)' // Orange
  };

  const annotations = {};
  if (bands) {
    Object.keys(bands).forEach((bandName, index) => {
      const [lowFreq, highFreq] = bands[bandName];
      annotations[`band${index}`] = {
        type: 'box',
        xMin: lowFreq,
        xMax: highFreq,
        yMin: 0, // Or adjust if your y-axis has a different scale base
        yMax: Math.max(...psd) * 1.1 || 1, // Adjust based on max PSD or a sensible upper limit
        backgroundColor: bandColors[bandName] || 'rgba(128, 128, 128, 0.1)',
        borderColor: 'transparent',
        label: {
          content: bandName.charAt(0).toUpperCase() + bandName.slice(1),
          display: true,
          position: 'start',
          color: 'black',
          font: { size: 10 }
        }
      };
    });
  }


  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Frequency (Hz)',
          color: '#666',
          font: { size: 14 }
        },
        ticks: {
            color: '#666',
            maxTicksLimit: 15 // Limit number of x-axis ticks for readability
        },
        grid: {
            color: 'rgba(200, 200, 200, 0.2)'
        }
      },
      y: {
        type: 'logarithmic', // Using logarithmic scale as in your original matplotlib example (semilogy)
        title: {
          display: true,
          text: 'Power Spectral Density',
          color: '#666',
          font: { size: 14 }
        },
        ticks: { color: '#666' },
        grid: {
            color: 'rgba(200, 200, 200, 0.2)'
        }
      },
    },
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#333' }
      },
      title: {
        display: true,
        text: `Frequency Spectrum - Epoch ${epoch_number} (${channel_name})`,
        color: '#333',
        font: { size: 18, weight: 'bold' }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
      annotation: { // Annotation plugin configuration
        annotations: annotations
      }
    },
    interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
    }
  };

  return (
    <div className="bg-white dark:bg-slate-800 p-4 md:p-6 rounded-lg shadow-lg h-96"> {/* Set a fixed height or aspect ratio */}
      <Line options={options} data={data} />
    </div>
  );
};

export default FrequencySpectrumChart;