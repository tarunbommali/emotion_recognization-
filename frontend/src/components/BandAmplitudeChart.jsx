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

const channelColors = [ // Define some colors for channels
  'rgb(255, 99, 132)',  // Red
  'rgb(54, 162, 235)',  // Blue
  'rgb(75, 192, 192)',  // Green
  'rgb(255, 205, 86)',  // Yellow
  'rgb(153, 102, 255)', // Purple
  'rgb(255, 159, 64)'   // Orange
];


const BandAmplitudeChart = ({ bandData, bandDetails }) => {
  if (!bandData || !bandData.time_vector || bandData.traces.length === 0) {
    return <p>No amplitude data for {bandData?.band_name || 'this band'}.</p>;
  }

  const chartData = {
    labels: bandData.time_vector.map(t => t.toFixed(3)), // X-axis: time
    datasets: bandData.traces.map((trace, index) => ({
      label: `${trace.channel_name}`,
      data: trace.amplitude, // Y-axis: amplitude
      borderColor: channelColors[index % channelColors.length],
      backgroundColor: 'rgba(0,0,0,0)', // Transparent fill
      tension: 0.1,
      pointRadius: 0, // No points for cleaner line
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
        title: { display: true, text: 'Amplitude (µV)' }, // Assuming µV
      },
    },
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: `Band-Filtered Amplitude: ${bandData.band_name.toUpperCase()} (${bandDetails[bandData.band_name]?.join('-')} Hz)`,
        font: { size: 16 },
      },
    },
  };

  return (
    <div className="bg-white  p-4 rounded-lg shadow-md h-72">
      <Line options={options} data={chartData} />
    </div>
  );
};

export default BandAmplitudeChart;