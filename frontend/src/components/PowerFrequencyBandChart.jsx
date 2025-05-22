// frontend/src/components/PowerFrequencyBandChart.jsx
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
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const channelColors = [
  'rgb(255, 99, 132)',  // Red
  'rgb(54, 162, 235)',  // Blue
  'rgb(75, 192, 192)',  // Green
  'rgb(255, 205, 86)',  // Yellow
  'rgb(153, 102, 255)', // Purple
  'rgb(255, 159, 64)'   // Orange
];

const PowerFrequencyBandChart = ({ title, channelsPsdData, targetBandName, targetBandRange }) => {
  if (!channelsPsdData || channelsPsdData.length === 0 || !targetBandName || !targetBandRange) {
    return <p>No data for {targetBandName} power spectrum.</p>;
  }

  const datasets = channelsPsdData.map((channelData, index) => {
    const { freqs, psd, channel_name } = channelData;
    if (!freqs || !psd) return null;

    const filteredData = [];
    const filteredLabels = [];

    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= targetBandRange[0] && freqs[i] <= targetBandRange[1]) {
        filteredLabels.push(freqs[i].toFixed(2));
        filteredData.push(psd[i]);
      }
    }
    if (filteredData.length === 0) return null; // No data in this band for this channel

    return {
      label: `${channel_name}`,
      data: filteredData,
      borderColor: channelColors[index % channelColors.length],
      backgroundColor: 'rgba(0,0,0,0)',
      tension: 0.1,
      pointRadius: 1,
      borderWidth: 1.5,
    };
  }).filter(dataset => dataset !== null); // Remove null datasets (e.g., if a channel had no data in band)
  
  // Use the labels from the first valid dataset (assuming all have same filtered frequency points)
  // This might need adjustment if frequency points differ significantly after filtering per channel.
  const labels = datasets.length > 0 ? 
    channelsPsdData.find(cd => cd.freqs && cd.psd)?.freqs
        .filter(f => f >= targetBandRange[0] && f <= targetBandRange[1])
        .map(f => f.toFixed(2))
    : [];


  if (datasets.length === 0) {
    return <p>No data within {targetBandName} range for any channel.</p>;
  }

  const chartData = { labels, datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Frequency (Hz)' },
      },
      y: {
        title: { display: true, text: 'Power (μV²/Hz)' }, // Assuming units
        type: 'logarithmic', // PSD is often viewed on a log scale
      },
    },
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: title || `${targetBandName.toUpperCase()} Band Power Spectrum (${targetBandRange[0]}-${targetBandRange[1]} Hz)`,
        font: { size: 16 },
      },
    },
  };

  return (
    <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-md h-72">
      <Line options={options} data={chartData} />
    </div>
  );
};

export default PowerFrequencyBandChart;