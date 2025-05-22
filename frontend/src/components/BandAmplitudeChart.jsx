// frontend/src/components/BandAmplitudeChart.jsx
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


const BandAmplitudeChart = ({ bandData }) => {
  // bandData is expected to be: { band_name, freq_range_hz, time_vector, traces: [{channel_name, amplitude}, ...] }
  if (!bandData || !bandData.time_vector || !bandData.traces || bandData.traces.length === 0) {
    return <p>No amplitude data for {bandData?.band_name || 'this band'}.</p>;
  }

  const chartData = {
    labels: bandData.time_vector.map(t => t.toFixed(3)), 
    datasets: bandData.traces.map((trace, index) => ({
      label: `${trace.channel_name}`,
      data: trace.amplitude, 
      borderColor: channelColors[index % channelColors.length],
      backgroundColor: 'rgba(0,0,0,0)', 
      tension: 0.1,
      pointRadius: 0, 
      borderWidth: 1.5,
    })),
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Time (s)' },
      },
      y: {
        title: { display: true, text: 'Amplitude (µV)' }, 
      },
    },
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: `Band-Filtered Amplitude: ${bandData.band_name.toUpperCase()} (${bandData.freq_range_hz?.[0]}-${bandData.freq_range_hz?.[1]} Hz)`,
        font: { size: 16 },
      },
    },
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-md h-72">
      <Line options={options} data={chartData} />
    </div>
  );
};

export default BandAmplitudeChart;