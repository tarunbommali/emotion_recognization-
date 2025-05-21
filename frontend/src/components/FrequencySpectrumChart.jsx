/* eslint-disable react-hooks/rules-of-hooks */

import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale, 
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { useEffect, useRef } from 'react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

const FrequencySpectrumChart = ({ plotData }) => {
  const chartRef = useRef(null); 

  // Destructure plotData safely
  const {
    epoch_number = 'N/A',
    channel_name = 'N/A',
    freqs = [],
    psd = [],
    bands = {}
  } = plotData || {};

  if (!plotData || freqs.length === 0 || psd.length === 0) {
    return <p className="text-center text-gray-500 dark:text-gray-400">No plot data available for this epoch.</p>;
  }

  const chartId = `spectrum-chart-${channel_name}-${epoch_number}`;

  // Memoize data and options or ensure they are stable if not changing
  // For simplicity here, we are redefining them on each render, which is
  // usually fine if plotData itself is a new object on change.

  const data = {
    labels: freqs.map(f => f.toFixed(2)),
    datasets: [
      {
        label: `PSD for ${channel_name}`,
        data: psd,
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1,
        fill: 'origin',
        pointRadius: 1,
        borderWidth: 1.5,
      },
    ],
  };

  const bandColors = {
    delta: 'rgba(0, 0, 255, 0.1)',
    theta: 'rgba(0, 128, 0, 0.1)',
    alpha: 'rgba(255, 0, 0, 0.1)',
    beta:  'rgba(128, 0, 128, 0.1)',
    gamma: 'rgba(255, 165, 0, 0.1)'
  };

  const annotations = {};
  if (bands && Object.keys(bands).length > 0) {
    Object.keys(bands).forEach((bandName, index) => {
      const [lowFreq, highFreq] = bands[bandName];
      let yAxisMax = 1;
      const positivePsdValues = psd.filter(p => p > 0);
      if (positivePsdValues.length > 0) {
        yAxisMax = Math.max(...positivePsdValues) * 1.1;
      }

      annotations[`band${index}`] = {
        type: 'box',
        xMin: lowFreq,
        xMax: highFreq,
        yMin: 0,
        yMax: yAxisMax,
        backgroundColor: bandColors[bandName] || 'rgba(128, 128, 128, 0.1)',
        borderColor: 'transparent',
        borderWidth: 0,
        label: {
          content: bandName.charAt(0).toUpperCase() + bandName.slice(1),
          display: true,
          position: 'start',
          color: 'rgba(0,0,0,0.7)',
          font: { size: 9, weight: 'bold' },
          xAdjust: 5,
          yAdjust: -5,
        }
      };
    });
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false, // Disable animation to see if it helps with rapid re-renders
    scales: {
      x: { /* ... your x-axis config ... */ 
        title: { display: true, text: 'Frequency (Hz)', color: '#666', font: { size: 14 } },
        ticks: { color: '#666', maxTicksLimit: 15 },
        grid: { color: 'rgba(200, 200, 200, 0.2)' }
      },
      y: { /* ... your y-axis config ... */ 
        type: 'logarithmic',
        title: { display: true, text: 'Power Spectral Density', color: '#666', font: { size: 14 } },
        ticks: { color: '#666' },
        grid: { color: 'rgba(200, 200, 200, 0.2)' }
      },
    },
    plugins: { /* ... your plugins config ... */ 
      legend: { position: 'top', labels: { color: '#333' } },
      title: { display: true, text: `Spectrum - Epoch ${epoch_number} (${channel_name})`, color: '#333', font: { size: 18, weight: 'bold' } },
      tooltip: { mode: 'index', intersect: false, },
      annotation: { annotations: annotations }
    },
    interaction: { /* ... your interaction config ... */ 
        mode: 'nearest', axis: 'x', intersect: false
    }
  };

  useEffect(() => {
  
    return () => {
      // Cleanup function: called when the component unmounts
      // or before the effect runs again if dependencies like `plotData` change.
      if (chartRef.current) {
        // console.log(`Destroying chart instance associated with canvas ID: ${chartRef.current.canvas?.id}, Chart ID: ${chartRef.current.id}`);
        chartRef.current.destroy();
        chartRef.current = null; // Clear the ref after destroying
      }
    };
  }, [plotData]); // Re-run this effect (and its cleanup) if plotData changes

  return (
    <div className="bg-white  p-4 md:p-6 rounded-lg shadow-lg h-96">
      <Line
        ref={chartRef} // Assign the ref to the Line component
        id={chartId}   // Keep the unique ID for the canvas element
        options={options}
        data={data}
      />
    </div>
  );
};

export default FrequencySpectrumChart;